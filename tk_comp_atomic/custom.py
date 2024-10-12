import sys, time, copy
sys.path.append("..")
# from simtransformer.model_bank import GPT2Standard
from simtransformer.module_base import ConfigBase, DataModuleBase, PipelineBase
import torch
import torch.nn as nn
from torch.utils.data import random_split
from simtransformer.manager import TrainingManagerBase
from simtransformer.model_bank import GPT2Standard
from simtransformer.utils import MRR_fn, EasyDict
from simtransformer.model_base import LinearWithChannel

probe_pos_len = 3

class Config(ConfigBase):
    pass

class TrainingManager(TrainingManagerBase):
    def get_training_name(self):
        training_name = f'atomic_OnlyDst_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # default
        # training_name = f'debugRelOnlyM_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # dst only
        print(f"Current training run: {training_name}")
        return training_name
    
    def config_pipeline(self):
        training_model = GPT2Standard(self.model_config)
        if self.train_config.loss_type == "cross_entropy":
            loss_p_model = nn.CrossEntropyLoss()
            loss_n_model = None
        return  {
            "train_config": self.train_config,
            "training_model": training_model,
            "loss_p_model": loss_p_model,
            "loss_n_model": loss_n_model
        }
        
    def config_probepipeline(self):
        training_model = self.pipeline.training_model
        
        added_probe_target = EasyDict({'encoder': {'blocks':{}}})
        block_dict_pointer = added_probe_target.encoder.blocks
        for name, block in training_model.encoder.blocks.items():
            block_dict_pointer.update({
                name: {
                    # "input": None,
                    "attn": {"input": None, "output": None}, 
                    # "attn_res_output": None,
                    "mlp": {"input": None, "output": None}, 
                    "output": None,
                }
            })
        added_probe_target_key = added_probe_target.flatten().keys()
        
        channel_size = len(added_probe_target_key) * probe_pos_len
        
        probe_layer = LinearWithChannel(
            input_size=self.model_config.hidden_size, 
            output_size=self.model_config.vocab_size,
            channel_size=channel_size)
        
        added_vis_target = EasyDict({'encoder': {'blocks':{}}})
        vis_dict_pointer = added_vis_target.encoder.blocks
        for name, block in training_model.encoder.blocks.items():
            vis_dict_pointer.update({
                name: {
                    "attn": {
                        "attn_prob": None, 
                        "logits_query_pos": None,
                        "logits_pos_key": None,
                        },
                }
            })
        added_vis_target_key = added_vis_target.flatten().keys()
        
        return {
            "probe_config": self.train_config,
            "training_model": training_model,
            "probe_layer": probe_layer,
            "probe_loss_model": nn.CrossEntropyLoss(),
            "added_probe_target_key": added_probe_target_key,
            "added_vis_target_key": added_vis_target_key
        }

class Pipeline(PipelineBase):
    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs['loss']
        loss_p = training_step_outputs['loss_p']
        loss_n = training_step_outputs['loss_n']
        output = training_step_outputs['output']
        batch = training_step_outputs['batch']
        x, y, mask, batch_info = self._unpack_batch(batch)
        
        y_msk_p = self._mask_select(y, mask)
        output_msk_p = self._mask_select(output, mask)
        
        mrr = MRR_fn(output_msk_p, y_msk_p)
        # log mrr
        self.log('train_mrr', mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        # # append x, y, output to three files for debugging
        # with open('x_train.txt', 'a') as f_x, open('y_train.txt', 'a') as f_y, open('output_train.txt', 'a') as f_output, open('loss_train.txt', 'a') as f_loss:
        #     print(x[0, 0:6].tolist(), file=f_x)
        #     print(y[0, 0:6].tolist(), file=f_y)
        #     print(output[0, 0:6, 1].tolist(), file=f_output)
        #     print(loss, file=f_loss)
        
    def validation_step_end(self, validation_step_outputs):
        loss = validation_step_outputs['loss']
        loss_p = validation_step_outputs['loss_p']
        loss_n = validation_step_outputs['loss_n']
        output = validation_step_outputs['output']
        batch = validation_step_outputs['batch']
        x, y, mask, batch_info = self._unpack_batch(batch)
        
        y_msk_p = self._mask_select(y, mask)
        output_msk_p = self._mask_select(output, mask)
        
        mrr = MRR_fn(output_msk_p, y_msk_p)
        # log mrr
        self.log('val_mrr', mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        # # append x, y, output to three files for debugging
        # with open('x_val.txt', 'a') as f_x, open('y_val.txt', 'a') as f_y, open('output_val.txt', 'a') as f_output, open('loss_val.txt', 'a') as f_loss:
        #     print(x[0, 0:6].tolist(), file=f_x)
        #     print(y[0, 0:6].tolist(), file=f_y)
        #     print(output[0, 0:6, 1].tolist(), file=f_output)
        #     print(loss, file=f_loss)
        
        

class DataModule(DataModuleBase):
    def train_val_test_split(self, data):
        # split data into train, validation, and test sets by ratio 90:5:5
        data_train, data_test = random_split(data, [int(0.9*len(data)), len(data)-int(0.9*len(data))])
        data_train, data_val = random_split(data_train, [int(0.9*len(data_train)), len(data_train)-int(0.9*len(data_train))])
        return data_train, data_val, data_test
    
    def transform_batch(self, batch, dataloader_idx):
        """
        Here, each sample consists of a dictionary with "pos", "sentence" and "reasoning_path" keys.
        """
        max_seq_len = max([len(sample['sentence']) for sample in batch])
        x_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long).fill_(self.vocab["<pad>"])
        y_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long)
        batch_info = []
        msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
        probe_msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
        probe_label = torch.zeros(len(batch), dtype=torch.long).fill_(self.vocab["<pad>"])
        
        for i, sample in enumerate(batch):
            sentence, reasoning_path, pos = sample['sentence'], sample['reasoning_path'], sample['pos']
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[:-1]))
            y_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[1:]))
            
            # find the position of '<pre_rel>' and set the token next to it to be '<Q>'
            pre_rel_pos = sentence.index('<pre_rel>')
            rel_pos = pre_rel_pos + 1
            pre_dst_pos = sentence.index('<pre_dst>')
            dst_pos = pre_dst_pos + 1
            
            # x_tensor[i, rel_pos] = self.vocab['<Q>']

            # for positions of '<pre_rel>' and '<pre_dst>', set the mask to True
            # msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_rel>'])] = True
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_dst>'])] = True
            
            # the following two is for probing
            probe_msk_tensor[i, [pre_rel_pos, rel_pos, pre_dst_pos]] = True

            # print(y_tensor.shape, dst_pos)

            # minus 1 because the target is the next token
            probe_label[i] = y_tensor[i, dst_pos - 1]
            
            batch_info.append(sample)
            
        return x_tensor, y_tensor, msk_tensor, probe_label, probe_msk_tensor, batch_info # x, y, msk, probe_label, probe_msk, batch_info
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        x_tensor, y_tensor, msk_tensor, probe_label, probe_msk_tensor, batch_info = batch
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        msk_tensor = msk_tensor.to(device)
        probe_label = probe_label.to(device)
        probe_msk_tensor = probe_msk_tensor.to(device)
        return x_tensor, y_tensor, msk_tensor, probe_label, probe_msk_tensor, batch_info

class TrainingManagerRelandDst(TrainingManager):
    def get_training_name(self):
        training_name = f'atomic_Rel_Dst_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # default
        # training_name = f'debugRelOnlyM_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # dst only
        print(f"Current training run: {training_name}")
        return training_name

class DataModuleRelandDst(DataModule):

    def transform_batch(self, batch, dataloader_idx):
        """
        Here, each sample consists of a dictionary with "pos", "sentence" and "reasoning_path" keys.
        """
        max_seq_len = max([len(sample['sentence']) for sample in batch])
        x_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long).fill_(self.vocab["<pad>"])
        y_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long)
        batch_info = []
        msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
        probe_msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
        probe_label = torch.zeros(len(batch), dtype=torch.long).fill_(self.vocab["<pad>"])
        
        for i, sample in enumerate(batch):
            sentence, reasoning_path, pos = sample['sentence'], sample['reasoning_path'], sample['pos']
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[:-1]))
            y_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[1:]))
            
            # find the position of '<pre_rel>' and set the token next to it to be '<Q>'
            pre_rel_pos = sentence.index('<pre_rel>')
            rel_pos = pre_rel_pos + 1
            pre_dst_pos = sentence.index('<pre_dst>')
            dst_pos = pre_dst_pos + 1
            
            # x_tensor[i, rel_pos] = self.vocab['<Q>']

            # for positions of '<pre_rel>' and '<pre_dst>', set the mask to True
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_rel>'])] = True
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_dst>'])] = True
            
            # the following two is for probing
            probe_msk_tensor[i, [pre_rel_pos, rel_pos, pre_dst_pos]] = True

            probe_label[i] = y_tensor[i, dst_pos-1] # minus 1 because the target is the next token
            
            batch_info.append(sample)
            
        return x_tensor, y_tensor, msk_tensor, probe_label, probe_msk_tensor, batch_info # x, y, msk, probe_label, probe_msk, batch_info
    