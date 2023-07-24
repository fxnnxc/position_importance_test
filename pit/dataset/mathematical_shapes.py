import torch
import numpy as np 
from typing import List 

from .base_shapes import BaseShapesDataset, BaseRule

class MathematicalShapesDataset(BaseShapesDataset):
    def __init__(self, 
                 train,
                 rule_indices:List,
                 num_shapes:int=10,
                 num_colors:int=1,
                 num_textures:int=1,
                 num_samples:int=1000,
                 min_seq_len:int=6,
                 max_seq_len:int=8,
                 is_one_hot_color:bool=True,
                 is_one_hot_texture:bool=True,
                 return_rule_label=False,
                 seed=0,
                 **kwargs,
                 ):
        
        self.train = train
        
        
        super().__init__(num_shapes,
                        num_colors,
                        num_textures,
                        rule_indices,
                        num_samples,
                        min_seq_len,
                        max_seq_len,
                        is_one_hot_color,
                        is_one_hot_texture,
                        return_rule_label,
                        seed,
        )
        self.shape_vocab_len = self.vocab_len
        self.vocab_len += 1  # equal
        self.vocab_len += 1  # modular equal 
        self.vocab_len += 1  # additive
        self.vocab_len += 1  # multiplication
        self.vocab_len += 1  # subtraction
        self.vocab_len += 1  # division
        self.vocab_len += 1  # negative
        self.vocab_len += 1  # eos_token

        self.equal_token     = self.shape_vocab_len
        self.mod_equal_token = self.shape_vocab_len + 1 
        self.add_token       = self.shape_vocab_len + 2
        self.multiply_token  = self.shape_vocab_len + 3
        self.subtract_token  = self.shape_vocab_len + 4
        self.divide_token    = self.shape_vocab_len + 5
        self.negative_token  = self.shape_vocab_len + 5
        self.eos_token = self.vocab_len - 1


        self.rules = [
                        globals()[f"Rule{i}"](
                                            num_shapes,
                                            num_colors,
                                            num_textures,
                                            self.vocab_len,
                                            min_seq_len,
                                            max_seq_len,
                                            is_one_hot_color,
                                            is_one_hot_texture,
                                            self.eos_token,
                                            self.equal_token,
                                            self.mod_equal_token,
                                            self.add_token,
                                            self.multiply_token,
                                            self.subtract_token,
                                            self.divide_token,
                                            negative_token = self.negative_token,
                                            ) 
                        for i in self.rule_indices
                    ] 
        self._make_data()

    def __getitem__(self, idx):
        input_ids = self.data[idx][:,0].long()
        return {"input_ids":input_ids}

    def _make_data(self):

        self.data = []
        # Rule 0 
        for a in range(100):
            for b in range(100):
                if self.train:
                    if (not (a>b)) or ((a+b)==100):
                        continue
                else: #test
                    if not ((a+b)==100):
                        continue
                inputs = (a,b)
                        
                sequence = self.rules[0].make_sample(None, inputs=inputs) 
                self.data.append(sequence)
        # Rule 4 
        for a in range(100):
            for b in range(100):
                if self.train:
                    if ((abs(a)+abs(b))==100):
                        continue 
                else:
                    if not ((abs(a)+abs(b))==100):
                        continue 
                inputs = (a,b)
                sequence = self.rules[1].make_sample(None, inputs=inputs)
                self.data.append(sequence)
    
        
        self.num_samples = len(self.data)


class Rule0(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool,
                 eos_token, equal_token, mod_equal_token,  add_token, multiply_token, subtract_token, divide_token, negative_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )

        self.eos_token = eos_token
        self.equal_token = equal_token
        self.mod_equal_token = mod_equal_token
        self.add_token = add_token
        self.multiply_token = multiply_token
        self.subtract_token = subtract_token
        self.divide_token = divide_token


    def _check_requirement(self):
        assert self.min_seq_len > 6 

    def make_sample(self, i:int, inputs=None):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        # a = np.random.randint(1, self.num_shapes-1)
        # b = np.random.randint(1, self.num_shapes - a)
        a,b = inputs
        c = a+b
        # assert c < self.num_shapes        
        sample[1, 0] = a 
        sample[2, 0] = self.add_token
        sample[3, 0] = b
        sample[4, 0] = self.equal_token
        sample[5, 0] = c
        return sample


class Rule4(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool,
                 eos_token, equal_token, mod_equal_token,  add_token, multiply_token, subtract_token, divide_token, negative_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )

        self.eos_token = eos_token
        self.equal_token = equal_token
        self.mod_equal_token = mod_equal_token
        self.add_token = add_token
        self.multiply_token = multiply_token
        self.subtract_token = subtract_token
        self.divide_token = divide_token
        self.negative_token = negative_token


    def _check_requirement(self):
        assert self.min_seq_len > 6 

    def make_sample(self, i:int, inputs=None):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        # a = np.random.randint(1, self.num_shapes)
        # b = np.random.randint(1, self.num_shapes)
        a,b = inputs
        c = a-b
        sample[1, 0] = a 
        sample[2, 0] = self.subtract_token
        sample[3, 0] = b
        sample[4, 0] = self.equal_token
        if c >= 0:
            sample[5, 0] = c
        else:
            sample[5, 0] = self.negative_token
            sample[6, 0] = -c

        return sample
    