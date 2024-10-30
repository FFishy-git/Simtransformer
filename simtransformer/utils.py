from collections.abc import MutableMapping
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import yaml, json, os, csv
from lightning.pytorch.utilities.parsing import AttributeDict
import random
from typing import Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def shuffle_with_indices(data: list, indices: Union[range, list]):
    combined = list(zip(data, indices))
    random.shuffle(combined)
    new_data, original_indices = zip(*combined)
    original_indices = sorted(range(len(original_indices)), key=lambda k: original_indices[k])
    return new_data, original_indices

def flatten_dict(dictionary, prefix='', separator='.'):
    """
    Flattens a nested dictionary by concatenating keys.

    Args:
        dictionary (dict): The dictionary to flatten.
        prefix (str, optional): A prefix to add to the keys. Defaults to ''.
        separator (str, optional): The separator to use between concatenated keys. Defaults to '.'.

    Returns:
        dict: A new dictionary with flattened keys.
    """
    items = []
    for key, value in dictionary.items():
        key_prefix = prefix + separator + key if prefix else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, key_prefix, separator=separator).items())
        else:
            items.append((key_prefix, value))
    return dict(items)

class CosineAnnealingWarmup(_LRScheduler):
    """
    Implements a learning rate scheduler that combines a warmup phase with a cosine annealing decay.
    The learning rate increases linearly from 0 to the specified learning rate during the warmup phase.
    After the warmup phase, the learning rate follows a cosine decay schedule down to a minimum learning rate.
    Attributes:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for the warmup phase.
        learning_rate (float): Maximum learning rate after warmup.
        min_lr (float): Minimum learning rate after decay.
        lr_decay_steps (int): Total number of steps for the decay phase.
        verbose (bool): If True, prints a message to stdout for each update.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            learning_rate: float,
            min_lr: float,
            lr_decay_steps: int,
            verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer=optimizer, last_epoch=-1, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self.learning_rate * self._step_count / self.warmup_steps
                    for group in self.optimizer.param_groups]
        if self._step_count > self.lr_decay_steps:
            return [self.min_lr for group in self.optimizer.param_groups]
        
        decay_ratio = (
            (self._step_count - self.warmup_steps)
            / (self.lr_decay_steps - self.warmup_steps)
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.learning_rate - self.min_lr)
                for group in self.optimizer.param_groups]
        
        
def MRR_fn(
    y_hat: torch.Tensor,
    y: torch.Tensor,
): 
    """
    Compute mean reciprocal rank, which is the average of the reciprocal ranks of the true labels.
    
    Args:
    - y_hat: tensor of shape (batch_size, num_classes)
    - y: tensor of shape (batch_size, )
    
    Returns:
    - mrr: tensor of shape (1, )
    """
    _, indices = torch.topk(y_hat, k=y_hat.shape[1]) # shape: (batch_size, num_classes), indices of top k values
    ranks = (indices == y.unsqueeze(1)).float() # indices == y.unsqueeze(1) will give a true value at the ranking position for the true label
    ranks = torch.argmax(ranks, dim=1) + 1 # get the rank of the true label
    return torch.mean(1.0 / ranks.float())


class EasyDict(AttributeDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = EasyDict(v)
            else:
                self[k] = v

    def update(self, other_dict):
        """This function overwrites the original update method of the dictionary class. It updates the dictionary with another dictionary. The hierarchical construction of EasyDict is called recursively.

        Args:
            other_dict: a dictionary to update the current dictionary with.
        """
        # if other_dict is empty, return
        if not other_dict:
            return
        for k, v in other_dict.items():
            if isinstance(v, dict):
                self[k] = EasyDict(v)
            else:
                self[k] = v
                
    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, EasyDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out
    
    def update_from_file(self, path):
        """This function updates the dictionary with the contents of a file using the `update` method.

        Args:
            path (_type_): path to the file to load the dictionary from.

        Raises:
            NotImplementedError: if the file extension is not supported.
        """
        self.update(clever_load(path)) # update the current dictionary with the loaded dictionary
        
        
    def save_to_file(self, path):
        out = self.to_dict()
        clever_save(out, path)
    
    def flatten(self, prefix='', separator='.'):
        """Return a flattened version of the dictionary.

        Args:
            prefix (str, optional): the prefix to the be added to each key. Defaults to ''.
            separator (str, optional): Defaults to '.'.

        Returns:
            EasyDict: a flattened version of the dictionary.
        """
        return AttributeDict(flatten_dict(self, prefix, separator))
    
    def setattr_with_string(self, key, value):
        keys = key.split(".")
        current = self
        for i, k in enumerate(keys[:-1]):
            if isinstance(current, EasyDict):
                if k not in current.keys():
                    current[k] = EasyDict()
                current = current[k]
            else:
                raise ValueError(f"Cannot set attribute {'.'.join(keys[:i])} as {'.'.join(keys[:i-1])} has value that is not an EasyDict.")
        k = keys[-1]
        if isinstance(current, EasyDict):
            current.update({k: value})
        else:
            raise ValueError(f"Cannot set attribute {key} as {'.'.join(keys[:-1])} has value that is not an EasyDict.")
        

def clever_load(file_path):
    """Support loading from both .yaml, .json, and .pth files.

    Args:
        file_path (str): path to the file to load.
    """
    if file_path.endswith(".yaml"):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".pth"):
        return torch.load(file_path)
    elif file_path.endswith(".csv"):
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f)
            obj =[row for row in reader]
            return obj
                
    else:
        raise NotImplementedError(f"File extension {file_path.split('.')[-1]} is not supported!")
    
def clever_save(obj, file_path):
    """Support saving to both .yaml, .json, and .pth files.

    Args:
        obj (_type_): the object to save.
        file_path (str): path to save the object to.
    """
    # check if the file directory exists
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if file_path.endswith(".yaml"):
        with open(file_path, "w") as f:
            yaml.dump(obj, f)
    elif file_path.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(obj, f)
    elif file_path.endswith(".pth"):
        torch.save(obj, file_path)
    elif file_path.endswith(".csv"):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(obj)
    else:
        raise NotImplementedError(f"File extension {file_path.split('.')[-1]} is not supported!")
    
    
    
def estimate_mfu(self, fwdbwd_per_iter, dt, num_layers, num_heads, dim_per_head, seq_len):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS 
    
    first estimate the number of flops we do per iteration.
    see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    """
    N = self.get_num_params()
    flops_per_token = 6 * N + 12 * num_layers * num_heads * dim_per_head * seq_len
    flops_per_fwdbwd = flops_per_token * seq_len
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu

def token_accuracy(y_hat, y):
    """
    Computes the token prediction accuracy.

    Parameters:
    y_hat (torch.Tensor): Predictions from the model of shape (batch_size, num_classes).
    y (torch.Tensor): Ground truth labels of shape (batch_size,).

    Returns:
    float: The accuracy of token predictions.
    """
    # Get the predicted class by taking the argmax over the class dimension
    predicted = torch.argmax(y_hat, dim=1)
    
    # Compare predictions with the ground truth labels
    correct_predictions = (predicted == y).sum().item()
    
    # Compute accuracy
    accuracy = correct_predictions / y.size(0)
    
    return accuracy

def check_cosine_similarity(embedding, target_embedding=None, verbose=False, emb_label=None, target_label=None, max_size=16, title=None):
    # check the cosine similarity between the embeddings
    embedding_np = embedding.cpu().detach().numpy()
    if target_embedding is None:
        cos_sim = cosine_similarity(embedding_np, embedding_np)
    else:
        target_embedding_np = target_embedding.cpu().detach().numpy()
        cos_sim = cosine_similarity(embedding_np, target_embedding_np)
    if verbose:
        if target_embedding is None:
            plt.figure(figsize=(max_size + 2, max_size))
            sns.heatmap(cos_sim, annot=False)
            if emb_label is not None:
                fontsize = max(6, 12 - len(emb_label) // 10)
                plt.xticks(ticks=np.arange(len(emb_label)), labels=emb_label, rotation=90, fontsize=fontsize)
                plt.yticks(ticks=np.arange(len(emb_label)), labels=emb_label, rotation=0, fontsize=fontsize)
            if title is not None:
                plt.title(title)
            plt.show()
        else:
            # select a size according to the relative length of embedding and target_embedding
            size = (len(embedding_np), len(target_embedding_np))
            # control the maximum size of the heatmap to be 16 while keeping the aspect ratio
            if size[0] > max_size:
                size = (max_size, int(max_size * size[1] / size[0]))
            elif size[1] > max_size:
                size = (int(max_size * size[0] / size[1]), max_size)
            size = (size[1] + 2, size[0])
            plt.figure(figsize=size)
            sns.heatmap(cos_sim, annot=False)
            if emb_label is not None:
                fontsize = max(6, 12 - len(emb_label) // 10)
                plt.xticks(ticks=np.arange(len(emb_label)), labels=emb_label, rotation=90, fontsize=fontsize)
            if target_label is not None:
                fontsize = max(6, 12 - len(target_label) // 10)
                plt.yticks(ticks=np.arange(len(target_label)), labels=target_label, rotation=0, fontsize=fontsize)
            if title is not None:
                plt.title(title)
            plt.show()
    return cos_sim