import os
import json
import torch
from torch.utils.data import Dataset
from pytorch_transformers import GPT2Tokenizer

from utils import add_special_tokens


class GPT21024Dataset(Dataset):

    def __init__(self, root_dir, ids_file, mode='train',length=None):
        self.root_dir = root_dir
        self.tokenizer = add_special_tokens()
        with open(ids_file,'r') as f:
            if mode=='train':
                self.idxs = json.load(f)['train_ids']
            elif mode=='valid':
                self.idxs = json.load(f)['valid_ids']
            else:
                self.idxs = json.load(f)['test_ids']
        if len == None:
            self.len = len(self.idxs)
        else:
            self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        idx = self.idxs[idx]
        file_name = os.path.join(self.root_dir,str(idx)+".json")
        with open(file_name,'r') as f:
              data = json.load(f)
        text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'article': text, 'sum_idx': len(data['article'])}
        return sample