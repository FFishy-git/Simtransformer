# Basic Format Setting

Here, we introduce some basic format in Simtransformer.

## Configuration Setting

## Data Setting

batch (tuple): A tuple containing the batch data. The batch can have:

* 2 elements: `(x, y)`
* 3 elements: `(x, y, msk)`
* More than 3 elements: `(x, y, msk, batch_info)`

Here, the shapes are given by

* `x.shape`= (batch_size, seq_len, Optional), where `Optional` can either be `None` for token index or `embedding_dim` if `x` is supplied as the embedding.
* `y.shape` =  (batch_size, seq_len, Optional), where `Optional` is usually the `logits`, i.e., log probability, or the output dimension.
* `msk.shape` = (batch_size, seq_len), a bool tensor indicating where the losses are computed.

In fact, for the position `msk` == `True`, `loss_p_model` in method **config_pipeline()** is applied and for the position `msk` == `False`, `loss_n_model` is applied. The final loss will be the the summation `loss` = `loss_p` + `loss_n`.

### Customizable Methods for Supplying a Batch in a Data

The methods related to batch are majorly in class `DataModuleBase`:

* **train_val_test_split(self, data)**: return (`data_train`, `data_val`, `data_test`)

* **transform_batch(self, batch, dataloader_idx)**: to pad the sequences in the batch and make it ready to be fed into the forward process. The return is a tuple (`x_tensor`, `y_tensor`, `msk_tensor`, `batch_info`)

* **transfer_batch_to_device(self, batch, device, dataloader_idx)**: `batch` is the batch after calling **transform_batch()**. You are expected to specify how to move the batch on the device. If you do not have `batch_info`, you can use the default method without overriding. The return is a tuple (`x_tensor`, `y_tensor`, `msk_tensor`, `batch_info`)

* **prepare_data()**: used for downloading dataset, or you can use it for regenerating the dataset. 

Here is an example: 

```python
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
        
        for i, sample in enumerate(batch):
            sentence, reasoning_path, pos = sample['sentence'], sample['reasoning_path'], sample['pos']
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = torch.tensor(sentence_idx_padded[:-1])
            y_tensor[i, :] = torch.tensor(sentence_idx_padded[1:])
            
            # for positions of '<pre_rel>' and '<pre_dst>', set the mask to True
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_rel>'])] = True
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_dst>'])] = True
            
            batch_info.append(sample)
            
        return x_tensor, y_tensor, msk_tensor, batch_info
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        x, y, msk, batch_info = batch
        x = x.to(device)
        y = y.to(device)
        msk = msk.to(device)
        return x, y, msk, batch_info
```

### DataModuleBase Configuration
The DataModuleBase is initialized by data_config