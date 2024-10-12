import os
from lightning.pytorch.utilities.parsing import AttributeDict
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from copy import deepcopy


# get the current file directory
current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)
# check if there exists a directory named 'data' below the current directory
if os.path.exists(os.path.join(current_dir, 'data')):
    print('The directory "data" exists')
else:
    # if not, create the directory
    os.makedirs(os.path.join(current_dir, 'data'))
data_path = os.path.join(current_dir, 'data')

# add parent directory to the system path
import sys
sys.path.append("..")
from simtransformer.module_base import Vocab
from simtransformer.utils import clever_save, clever_load

class DataMethod:
    """
    The data is constructed based on a graph upon `entities` and `relations`. We build a vocabulary with `entities`, `relations`, `placeholders`, `variables`, and `specials`.
        - The `entities` are the nodes in the graph, and the `relations` are the edges in the graph.
        - The `placeholders` are randomly sampled entities that are not connected to any relation.
        - The `variables` are used to represent intermediate entities in the reasoning path, e.g., `x` in `f(a)=x`.
        - The `specials` are used to represent special tokens, e.g., question marks, `<pad>`, `<eos>`, etc.
        
    A reasoning step is a triple `(a, f, b)`, which means `f(a)=b`. Here, `a` and `b` are entities (nodes), and `f` is a relation (edge). Such a triple is also referred to as an atomic fact. We call `a` as the entity connected to the relation `f`, and `b` as the destination entity or destination for short.
    
    Each data piece contains three reasoning steps, which could be either of the following types with equal probability:
        - `data_type`=1: `a->f->b`, `b->g->c`, `b->h->d`.
        - `data_type`=2: `a->f->b`, `b->g->c`, `c->h->d`.
    
    For each reasoning step, e.g., `a->f->b`, we wrap the reasoning path with surrounding context. We have parameter `win_size` to control the window size of the surrounding context. In particular:
        - For the first entity `a`, we sample a window with fixed size `win_size` with `a` appearing at a random position. The other tokens in the window are randomly sampled from `entities` and `placeholders` conditioned on not being a connected entity to the relation `f`. In the other word, this `a` is the ***unique*** entity connected to the relation `f` in the window.
        - For the relation `f`, we sample a window of size randomly chosen from [1, `win_size`] from `entities` and `placeholders` with `f` appearing at the beginning of the window. In this way, we ensure that `a` always appears in the window of size `win_size` before `f`. Moreover, there is only one relation `f` in the window.
        - The last token in the reasoning path is the destination entity `b`.
    Note that the last entity `b` will be replaced by a variable, say `x`, in the data. There is in fact only a ***unique*** way for the model to reason about the erased destination entity `b` by the following:
        - At the position of `x`, look for the only relation `f` token in the past window of size `win_size`.
        - At the position of `f`, look for the only entity `a` connected to the relation `f` in the past window of size `win_size`.
        
    Between two reasoning paths and at the end of the sentence, we add random sequences of tokens. The length of the random sequence is sampled from `[0, win_size - 1]` with tokens randomly sampled from `entities`, `relations`, and `placeholders`.
    
    At the end of the sequence, we add a special token, `specials[0]`, to indicate the question mark.
    Following the question mark, users are expected to append a question to the sentence. The question and the answer can be found as follows:
    | Variable | Question          |
    |----------|-------------------|
    | `x`      | `variables[0]`    |
    | `y`      | `variables[1]`    |
    | `z`      | `variables[2]`    |
    """
    def __init__(self, 
                 num_entities: int, 
                 num_relations: int,
                 num_placeholders: int,
                 max_degree: int, 
                 win_size: int = 5,
                 **kwargs):
        """
        Initialize the data method. Here we require `num_variables` >= 3 and `num_specials` >= 1.
        
        Args:
            num_entities: the number of entities
            num_relations: the number of relations
            num_placeholders: the number of placeholders
            num_variables: the number of variables
            num_specials: the number of special tokens
            max_degree: the maximum degree of the graph
            win_size: the window size for the surrounding context
            kwargs: other arguments
            
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.kwargs = kwargs

        # build the knowledge graph
        entities = [f'ent_{i}' for i in range(num_entities)]
        relations = [f'rel_{i}' for i in range(num_relations)]
        placeholders = [f'phd_{i}' for i in range(num_placeholders)]
        specials = ['<pre_ent>', '<pre_rel>', '<pre_dst>', '<Q>', '<A>', '<pad>', '<dst>', 'x', 'y', 'z']
        
        atomic_ls = [] # used for storing the triples
        atomic_dict = {} # used for lookup of all (rel, dst) pairs for an entity
        rel2ent_dict = {} # used for inverse lookup of all entities connected to a relation
        assert max_degree <= num_relations, f'max_degree={max_degree} should be less than or equal to num_relations={num_relations}'
        for ent in entities:
            # sample the number of relations for entity `ent` 
            num_rel = np.random.randint(1, max_degree)
            # sample the relations for entity `ent`
            rels = random.sample(relations, num_rel)
            
            atomic_dict[ent] = []
            # for each relation, sample the connected entities
            for rel in rels:
                # sample the connected entity
                dst = random.choice(entities)
                
                # save the triple in the atomic list
                atomic_ls.append((ent, rel, dst))
                # save the triple in the lookup dictionary
                atomic_dict[ent].append((rel, dst))
                # save the reverse lookup
                if rel not in rel2ent_dict.keys():
                    rel2ent_dict[rel] = []
                rel2ent_dict[rel].append(ent)
        
        self.atomic_ls = atomic_ls
        self.atomic_dict = atomic_dict
        self.rel2ent_dict = rel2ent_dict
        self.entities = list(entities)
        self.relations = list(relations)
        self.placeholders = list(placeholders)
        self.specials = list(specials)
        
        self.win_size = win_size
        
        self.ent_phd = self.entities + self.placeholders
        self.ent_rel_phd = self.entities + self.relations + self.placeholders
        

        vocab_ls = self.entities + self.relations + self.placeholders + self.specials
        # counter = Counter(vocab_ls)
        self.vocab = Vocab(vocab_ls)

        self.vocab_size = len(self.vocab)


    
    def _single_reasoning_data_wrapped(self, 
                                       reasoning_path: list | tuple,
                                       sentence: list=[],
                                       pos: list=[]):
        """
        Given a reasoning path `a, f, b`, fill in the sample with surrounding context. 
        The last token is `b`

        Args:
            reasoning_path: a list of 3 elements, [a, f, b]
            pos: a list of previous positions of the reasoning path
            sentence: a list of tokens in the previous sentence

        Returns:
            sentence: extended sentence with the reasoning path
            pos: a list of positions of the reasoning path in the current sentence
        """
        a, f, b= reasoning_path
        tail_idx = len(sentence)
        
        # choose the window size for `a`
        a_win_size = self.win_size
        # fill the windows with ent_phd unrelated to `f` by sampling with replacement
        a_win_tokens = random.choices(list(set(self.ent_phd).difference(set(self.rel2ent_dict[f]))), k=a_win_size)
        # let `a` appear in the window at a random position
        a_pos_in_win = np.random.choice(a_win_size)
        a_win_tokens[a_pos_in_win] = a
        # save the position of `a` in the sentence
        pos.append(a_pos_in_win + tail_idx)
        tail_idx += a_win_size
        sentence += a_win_tokens
        
        # add a special token for the relation
        sentence += ['<pre_rel>']
        tail_idx += 1
        # pos.append(tail_idx + 0)
        
        # choose the window size for `f`
        f_win_size = np.random.choice(self.win_size) + 1
        # fill the windows with ent_phd by sampling with replacement
        f_win_tokens = random.choices(list(self.ent_phd), k=f_win_size)
        # let `f` appear at the beginning of the window
        f_win_tokens[0] = f
        # save the position of `f` in the sentence
        pos.append(tail_idx + 0)
        tail_idx += f_win_size
        sentence += f_win_tokens
        
        # add a special token for the destination entity
        sentence += ['<pre_dst>']
        tail_idx += 1
        # pos.append(tail_idx + 0)
        
        # append `b`
        pos.append(tail_idx + 0)
        tail_idx += 1
        sentence.append(b)
        
        return sentence, pos
        
    def _add_random_tokens(self, sentence: list):
        """
        Add random tokens to the sentence
        """
        random_win_size = np.random.choice(self.win_size)
        random_tokens = random.choices(self.ent_rel_phd, k=random_win_size)
        sentence += random_tokens
        return sentence

    def __generate_sample__(self):
        """
        generate one data sample
        """
        sentence = []
        pos = []
        
        for _ in range(1):
            a, f, b = random.choice(self.atomic_ls)
            sentence, pos = self._single_reasoning_data_wrapped([a, f, b], sentence=sentence, pos=pos)
            sentence = self._add_random_tokens(sentence)

        reasoning_path = [sentence[i] for i in pos] # pos: [a, f, x]

        return AttributeDict({
            'sentence': sentence, 
            'pos':pos, # indices for important tokens in the sentence
            'reasoning_path': reasoning_path,
            })

    def generate_data(self, 
                        num_samples: int, 
                        data_path: str, 
                        vocab_path: str):
        """
        generate data samples, and save them to a file if `save_path` is provided

        Args:
            num_samples: the number of samples to generate
            save_path: the path to save the generated data

        Returns:
            data: a list of generated samples
        """
        data = []
        for _ in range(num_samples):
            sample = self.__generate_sample__()
            data.append(sample)
            
        clever_save(data, data_path)
        # also save the vocabulary
        clever_save(self.vocab.vocab, vocab_path)
        return data

# def split_data(
#             data_dict: dict,
#             data_config: AttributeDict,
#             ):
#     """
#     Split the data into training, validation, and test sets.

#     Args:
#         data_dict: the data dictionary
#         train_ratio: the ratio of training data
#         second_step_ratio: the ratio of the second reasoning step
#         allow_Qz: whether to allow question about the second reasoning step

#     Returns:
#         train_data: the training data
#         val_data: the validation data
#         test_data: the test data
#         vocab: the vocabulary
#     """
#     train_ratio=data_config.training_data_ratio
#     # second_step_ratio=data_config.second_step_ratio
#     # allow_Qz=data_config.allow_Qz
#     # second_step_enabled=data_config.second_step_enabled
    
#     vocab = data_dict['vocab']
#     data = data_dict['data']
    
    
    
#     # let's append question to the data
#     for sample in data:
#         # with probability second_step_ratio ask question about the second reasoning step
#         question_type = np.random.choice(['x', 'y', 'z'])
#         if question_type == 'x':
#             sample['Q'] = sample['x_var_idx']
#             sample['A'] = sample['x_value_idx']
#             sample['Q_type'] = 'x'
#         elif question_type == 'y':
#             sample['Q'] = sample['y_var_idx']
#             sample['A'] = sample['y_value_idx']
#             sample['Q_type'] = 'y'
#         else:
#             sample['Q'] = sample['z_var_idx']
#             sample['A'] = sample['z_value_idx']
#             sample['Q_type'] = 'z'
        
#     # split the data
#     num_train = int(len(data) * train_ratio)
#     train_data = data[:num_train]
    
#     num_val = int((len(data) - num_train) / 2)
#     val_data = data[num_train:num_train + num_val]
    
#     test_data = data[num_train + num_val:]

#     return train_data, val_data, test_data, vocab

# def batch_to_model_input(batch, padding_value=0, device='cpu'):
#     """
#     Convert a batch of samples to model input

#     Args:
#         batch: a list of samples
#         padding_value: the padding value

#     Returns:
#         x_tensor: the input tensor
#         y_tensor: the output tensor
#         batch_ls: the list of samples
#     """
#     max_seq_len = max([len(sample[0]) for sample in batch])
#     x_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.long).fill_(padding_value)
#     y_tensor = torch.zeros(len(batch), dtype=torch.long)
#     batch_ls = []
#     for i, sample in enumerate(batch):
#         x, y, sample_dict = sample
#         x_padded = x + [padding_value] * (max_seq_len - len(x))
#         x_tensor[i, :] = torch.tensor(x_padded)
#         y_tensor[i] = y
#         batch_ls.append(sample_dict)
#     return x_tensor.to(device), y_tensor.to(device), batch_ls

# def load_data(data_config: AttributeDict):
#     # get the current file directory
#     currentdir = os.path.dirname(os.path.realpath(__file__))
#     data_path = os.path.join(currentdir, "data", data_config.data_file_name)
#     with open(data_path, "r") as f:
#         data_dict = json.load(f)
#     return data_dict

# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         x = deepcopy(sample['sentence']) # deep copy the sentence to avoid changing the original data
#         x.append(sample['Q'])
#         y = sample['A']
#         return x, y, sample

#     def load_data(data_path: str):
#         with open(data_path, 'r', encoding='utf-8') as f:
#             data_dict = json.load(f)
#         return data_dict
    

        



        
