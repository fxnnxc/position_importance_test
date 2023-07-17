import torch
import numpy as np 
from typing import List 

class BaseShapesDataset:
    def __init__(self, 
                 num_shapes:int=10,
                 num_colors:int=1,
                 num_textures:int=1,
                 rule_indices:List=[],
                 num_samples:int=1000,
                 min_seq_len:int=1,
                 max_seq_len:int=10,
                 is_one_hot_color:bool=True,
                 is_one_hot_texture:bool=True,
                 return_rule_label=False,
                 seed=0,
                 **kwargs,
                 ):
        

        self.seed = seed 
        np.random.seed(self.seed)

        self.num_shapes=num_shapes
        self.num_colors=num_colors
        self.num_textures=num_textures
        self.rule_indices = rule_indices
        self.num_rules = len(rule_indices)
        self.num_samples = num_samples

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.is_one_hot_color = is_one_hot_color
        self.is_one_hot_texture = is_one_hot_texture
        self.return_rule_label = return_rule_label

        if self.is_one_hot_color and self.is_one_hot_texture:
            self.vocab_len = self.num_shapes * self.num_colors * self.num_textures
        elif self.is_one_hot_color:
            self.vocab_len = self.num_shapes * self.num_colors 
        elif self.is_one_hot_texture:
            self.vocab_len = self.num_shapes * self.num_textures
        else:
            self.vocab_len = self.num_shapes

    def __getitem__(self, idx:int):
        x, y = self.data[idx]
        return x, y

    def __len__(self):
        return len(self.data)

    def _make_data(self):
        if self.num_rules > len(self.rules):
            print(f" changed the number of labels : {self.num_rules} --> {len(self.rules)} ")
            self.num_rules = len(self.rules)

        num_samples_per_rule = self.num_samples // len(self.rules)
        self.data = []
        for rule_idx, rule in enumerate(self.rules):
            for i in range(num_samples_per_rule):
                sequence = rule.make_sample(i) 
                if self.return_rule_label:
                    rule_idx =  rule.__class__.__name__
                self.data.append((sequence, rule_idx))
        self.num_samples = num_samples_per_rule * len(self.rules)


class BaseRule:
    def __init__(self, 
                 num_shapes:int,
                 num_colors:int,
                 num_textures:int,
                 vocab_len:int,
                 min_seq_len:int,
                 max_seq_len:int,
                 is_one_hot_color:bool,
                 is_one_hot_texture:bool,
                 **kwargs,
                 ):
        self.num_shapes=num_shapes
        self.num_colors=num_colors
        self.num_textures=num_textures
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.is_one_hot_color = is_one_hot_color 
        self.is_one_hot_texture = is_one_hot_texture
        self.vocab_len = vocab_len

    def _check_requirement(self):
        raise NotImplementedError() 

    def make_sample(self, i:int):
        raise NotImplementedError()


    def make_one_hot_form(self, x):
        if self.is_one_hot_color and self.is_one_hot_texture:
            color_incr = self.num_shapes
            texture_incr = self.num_shapes * self.num_colors

            x[0] += (color_incr * x[1]) + (texture_incr *x[2])
            x[1] = self.vocab_len
            x[2] = self.vocab_len
        elif self.is_one_hot_color:
            color_incr = self.num_shapes
            x[0] += (color_incr * x[1]) 
            x[1] = self.vocab_len

        elif self.is_one_hot_texture:
            texture_incr = self.num_shapes
            x[0] += (texture_incr * x[2]) 
            x[2] = self.vocab_len
        else:
            pass 
        return x 
