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
from typing import Optional

probe_pos_len = 3

class Config(ConfigBase):
    pass

class TrainingManager(TrainingManagerBase):
    def __init__(self, dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule, only_dst=False):
        self.only_dst = only_dst
        super(TrainingManager, self).__init__(dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule)
        

    def get_training_name(self):
        if self.only_dst:
            training_name = f'atomic_OnlyDst_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # default
        else:
            training_name = f'atomic_Rel_Dst_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S")
        # training_name = f'atomic_OnlyDst_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # default
        # training_name = f'debugRelOnlyM_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # dst only
        print(f"Current training run: {training_name}")
        return training_name
    
    def config_pipeline(self):
        training_model = GPT2Standard(self.model_config, input_size=self.data_config.vocab_size, output_size=self.data_config.vocab_size)
        loss_p_model = nn.CrossEntropyLoss()
        loss_n_model = None
        return  {
            "train_config": self.train_config,
            "training_model": training_model,
            "loss_p_model": loss_p_model,
            "loss_n_model": loss_n_model,
            "only_dst": self.only_dst
        }
    
    def config_datamodule(self):
        if "batch_size" not in self.data_config.to_dict().keys():
            self.data_config.batch_size = self.train_config.batch_size
        return {
            "data_config": self.data_config, 
            "dir_handler": self.dir_handler,
            "only_dst": self.only_dst
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
        
        in_channel_size_ls = (len(added_probe_target_key),  probe_pos_len)
        out_channel_size_ls = [1]
        
        probe_layer = LinearWithChannel(
            input_size=self.model_config.hidden_size, 
            output_size=self.model_config.vocab_size,
            in_channel_size_ls=in_channel_size_ls,
            out_channel_size_ls=out_channel_size_ls
        )
        
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
            "probe_config": self.probe_config,
            "pipeline": self.pipeline,
            "probe_layer": probe_layer,
            "probe_loss_model": nn.CrossEntropyLoss(reduction='none'),
            "added_probe_target_key": added_probe_target_key,
            "added_vis_target_key": added_vis_target_key
        }

class Pipeline(PipelineBase):

    def __init__(self, train_config, training_model, loss_p_model, loss_n_model, only_dst=False):
        """
        add only_dst parameter to Pipeline class
        """
        super(Pipeline, self).__init__(train_config, training_model, loss_p_model, loss_n_model)
        self.only_dst = only_dst # if True, only consider the destination entity

    def _split_mask(self, mask):
        """
        Splits the mask into two masks:
        - mask_rel: for the first True value in each sequence
        - mask_dst: for the second True value in each sequence
        """
        mask_rel = torch.zeros_like(mask, dtype=torch.bool)
        mask_dst = torch.zeros_like(mask, dtype=torch.bool)

        for i in range(mask.size(0)):  # Iterate over batch dimension
            true_indices = torch.nonzero(mask[i]).squeeze()  # Get the indices where mask is True
            assert len(true_indices) == 2, f"Expected 2 True values in mask, got {len(true_indices)}"
            mask_rel[i, true_indices[0]] = True  # First True for rel
            mask_dst[i, true_indices[1]] = True  # Second True for dst

        return mask_rel, mask_dst


    def _Step(self, batch, batch_idx, step_type: Optional[str] = None):
        ## --------- forward pass --------- ##
        
        # print("batch", batch)
        train_batch, _, _ = self._unpack_batch(batch)
        x, y, mask = train_batch
        # x (batch_size, seq_len, Optional)
        # y (batch_size, seq_len, Optional)
        # mask (batch_size, seq_len)

        output = self.training_model(x)

        # compute the loss for the masked position
        # y_msk_p = self._mask_select(y, mask)

        if self.only_dst:
            y_msk_p_dst = self._mask_select(y, mask)
            output_msk_p_dst = self._mask_select(output, mask)

            loss_p_dst = self.loss_p_model(output_msk_p_dst, y_msk_p_dst)
            loss_p_rel = 0.0
        else:
            # (assuming each seq_len has exactly two True values)
            mask_rel, mask_dst = self._split_mask(mask)

            # Select the targets and outputs corresponding to the relation (rel) and destination (dst)
            y_msk_p_rel = self._mask_select(y, mask_rel)  # Select for relation
            output_msk_p_rel = self._mask_select(output, mask_rel)  # Model's output for relation

            y_msk_p_dst = self._mask_select(y, mask_dst)  # Select for destination
            output_msk_p_dst = self._mask_select(output, mask_dst) 

            loss_p_rel = self.loss_p_model(output_msk_p_rel, y_msk_p_rel)
            loss_p_dst = self.loss_p_model(output_msk_p_dst, y_msk_p_dst)

        loss_p = loss_p_rel + loss_p_dst

        # compute the loss for the non-masked position
        y_msk_n = self._mask_select(y, ~mask)
        output_msk_n = self._mask_select(output, ~mask)

        if len(y_msk_n) > 0 and self.loss_n_model is not None:
            loss_n = self.loss_n_model(output_msk_n, y_msk_n)
        else:
            loss_n = 0.0

        ## --------- log training loss --------- ##
        if step_type is not None:
            if self.loss_n_model is not None:
                self.log(step_type + "_loss_n", loss_n, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
                self.log(step_type + "_loss_p", loss_p, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            self.log(step_type + "_loss", loss_p + loss_n, prog_bar=True, on_epoch=True, logger=True, batch_size=self.len_batch(batch)) # should always log the total loss as 'val_loss' is used for ckpt saving

            if not self.only_dst:
                self.log(step_type + "_loss_rel", loss_p_rel, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            self.log(step_type + "_loss_dst", loss_p_dst, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            # don't change the log name step_type + '_loss' as it is used for ckpt saving

        return loss_p, loss_n, output

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs['loss']
        loss_p = training_step_outputs['loss_p']
        loss_n = training_step_outputs['loss_n']
        output = training_step_outputs['output']
        batch = training_step_outputs['batch']
        train_batch, _, _ = self._unpack_batch(batch)
        x, y, mask = train_batch

        if self.only_dst:
            y_msk_p_dst = self._mask_select(y, mask)
            output_msk_p_dst = self._mask_select(output, mask)

            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)
        
        else:
            mask_rel, mask_dst = self._split_mask(mask)

            y_msk_p_rel = self._mask_select(y, mask_rel)
            output_msk_p_rel = self._mask_select(output, mask_rel)

            y_msk_p_dst = self._mask_select(y, mask_dst)
            output_msk_p_dst = self._mask_select(output, mask_dst)

            # y_msk_p = self._mask_select(y, mask)
            # output_msk_p = self._mask_select(output, mask)
            
            # mrr = MRR_fn(output_msk_p, y_msk_p)
            mrr_rel = MRR_fn(output_msk_p_rel, y_msk_p_rel)
            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)

        # log mrr
        # self.log('train_mrr', mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        if not self.only_dst:
            self.log('train_mrr_rel', mrr_rel, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log('train_mrr_dst', mrr_dst, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

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
        train_batch, _, _ = self._unpack_batch(batch)
        x, y, mask = train_batch

        if self.only_dst:
            y_msk_p_dst = self._mask_select(y, mask)
            output_msk_p_dst = self._mask_select(output, mask)

            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)
        
        else:
            mask_rel, mask_dst = self._split_mask(mask)

            y_msk_p_rel = self._mask_select(y, mask_rel)
            output_msk_p_rel = self._mask_select(output, mask_rel)

            y_msk_p_dst = self._mask_select(y, mask_dst)
            output_msk_p_dst = self._mask_select(output, mask_dst)
            
            # y_msk_p = self._mask_select(y, mask)
            # output_msk_p = self._mask_select(output, mask)
            
            # mrr = MRR_fn(output_msk_p, y_msk_p)
            mrr_rel = MRR_fn(output_msk_p_rel, y_msk_p_rel)
            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)

        # log mrr
        # self.log('val_mrr', mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        if not self.only_dst:
            self.log('val_mrr_rel', mrr_rel, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log('val_mrr_dst', mrr_dst, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        # # append x, y, output to three files for debugging
        # with open('x_val.txt', 'a') as f_x, open('y_val.txt', 'a') as f_y, open('output_val.txt', 'a') as f_output, open('loss_val.txt', 'a') as f_loss:
        #     print(x[0, 0:6].tolist(), file=f_x)
        #     print(y[0, 0:6].tolist(), file=f_y)
        #     print(output[0, 0:6, 1].tolist(), file=f_output)
        #     print(loss, file=f_loss)
        
        

class DataModule(DataModuleBase):
    def __init__(self, data_config, dir_handler, only_dst=False):
        super(DataModule, self).__init__(data_config, dir_handler)
        self.only_dst = only_dst

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
        probe_label = torch.zeros(len(batch), 1, dtype=torch.long).fill_(self.vocab["<pad>"])
        
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
            if not self.only_dst:
                msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_rel>'])] = True
            # msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_rel>'])] = True
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_dst>'])] = True
            
            # the following two is for probing
            probe_msk_tensor[i, [pre_rel_pos, rel_pos, pre_dst_pos]] = True

            # print(y_tensor.shape, dst_pos)

            # minus 1 because the target is the next token
            probe_label[i, 0] = sentence_idx_padded[dst_pos]
            
            batch_info.append(sample)
            
        return EasyDict({
            "prompt": x_tensor,
            "label": y_tensor,
            "mask": msk_tensor,
            "probe_label": probe_label,
            "probe_mask": probe_msk_tensor,
            "batch_info": batch_info
        })
    
    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     for val in batch.values():
    #         if isinstance(val, torch.Tensor):
    #             val = val.to(device)
    #     return batch

# class TrainingManagerRelandDst(TrainingManager):
#     def get_training_name(self):
#         training_name = f'atomic_Rel_Dst_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # default
#         # training_name = f'debugRelOnlyM_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # dst only
#         print(f"Current training run: {training_name}")
#         return training_name

# class DataModuleRelandDst(DataModule):

#     def transform_batch(self, batch, dataloader_idx):
#         """
#         Here, each sample consists of a dictionary with "pos", "sentence" and "reasoning_path" keys.
#         """
#         max_seq_len = max([len(sample['sentence']) for sample in batch])
#         x_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long).fill_(self.vocab["<pad>"])
#         y_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long)
#         batch_info = []
#         msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
#         probe_msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
#         probe_label = torch.zeros(len(batch), dtype=torch.long).fill_(self.vocab["<pad>"])
        
#         for i, sample in enumerate(batch):
#             sentence, reasoning_path, pos = sample['sentence'], sample['reasoning_path'], sample['pos']
#             sentence_idx = [self.vocab[word] for word in sentence]
#             sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
#             x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[:-1]))
#             y_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[1:]))
            
#             # find the position of '<pre_rel>' and set the token next to it to be '<Q>'
#             pre_rel_pos = sentence.index('<pre_rel>')
#             rel_pos = pre_rel_pos + 1
#             pre_dst_pos = sentence.index('<pre_dst>')
#             dst_pos = pre_dst_pos + 1
            
#             # x_tensor[i, rel_pos] = self.vocab['<Q>']

#             # for positions of '<pre_rel>' and '<pre_dst>', set the mask to True
#             msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_rel>'])] = True
#             msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_dst>'])] = True
            
#             # the following two is for probing
#             probe_msk_tensor[i, [pre_rel_pos, rel_pos, pre_dst_pos]] = True

#             probe_label[i] = sentence_idx_padded[dst_pos] # minus 1 because the target is the next token
            
#             batch_info.append(sample)
            
#         return EasyDict({
#             "prompt": x_tensor,
#             "label": y_tensor,
#             "mask": msk_tensor,
#             "probe_label": probe_label,
#             "probe_mask": probe_msk_tensor,
#             "batch_info": batch_info
#         })
    