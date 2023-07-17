import torch
import numpy as np 
from typing import List 

from .base_shapes import BaseShapesDataset, BaseRule

class AutoRegressiveShapesDataset(BaseShapesDataset):
    def __init__(self, 
                 rule_indices:List,
                 num_shapes:int=10,
                 num_colors:int=1,
                 num_textures:int=1,
                 num_samples:int=1000,
                 min_seq_len:int=5,
                 max_seq_len:int=10,
                 is_one_hot_color:bool=True,
                 is_one_hot_texture:bool=True,
                 return_rule_label=False,
                 seed=0,
                 **kwargs,
                 ):
        
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

        self.vocab_len += 1  # eos_token
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
                                            eos_token=self.eos_token,
                                            ) 
                        for i in self.rule_indices
                    ] 
        self._make_data()


class Rule0(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 1
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 3

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i!=1])  
        random_loc = np.random.randint(1, seq_len-2)
        sample[random_loc,  0] = 1 
        sample[random_loc+1,0] = 1 
        return sample 


class Rule1(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 2
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 3

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i!=1])  
        random_loc = np.random.randint(1, seq_len-2)
        sample[random_loc,  0] = 1
        sample[random_loc+1,0] = 2 
        return sample 
    
class Rule2(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 0
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 3

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i!=2])  
        random_loc = np.random.randint(1, seq_len-2)
        sample[random_loc,  0] = 2
        sample[random_loc+1,0] = 3 
        return sample 
    
class Rule3(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 4
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 3

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i!=3])  
        random_loc = np.random.randint(1, seq_len-2)
        sample[random_loc,  0] = 3
        sample[random_loc+1,0] = 4 
        return sample 
    
    
class Rule4(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 5
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 3

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i!=4])  
        random_loc = np.random.randint(1, seq_len-2)
        sample[random_loc,  0] = 4
        sample[random_loc+1,0] = 5 
        return sample 
    
class Rule5(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 3
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 5

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,2]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 2
        sample[random_locs[2], 0] = 2
        sample[random_locs[2]+1, 0] = 1
        return sample 
    
class Rule6(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 4
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 5

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,3,4]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 3
        sample[random_locs[2], 0] = 3
        sample[random_locs[2]+1, 0] = 4
        return sample 

    
class Rule7(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 4
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 5

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,4,5]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 4
        sample[random_locs[2], 0] = 4
        sample[random_locs[2]+1, 0] = 5
        return sample 
    
class Rule8(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 7
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 6

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,5,6,7]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 5
        sample[random_locs[2], 0] = 5
        sample[random_locs[2]+1, 0] = 6
        sample[random_locs[2]+2, 0] = 7
        return sample 
    

class Rule9(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 7
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 6

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,5]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 5
        sample[random_locs[2], 0] = 1
        sample[random_locs[2]+1, 0] = 5 
        return sample 
    

    
class Rule10(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 7
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 6

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,5]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 5
        sample[random_locs[2], 0] = 1
        sample[random_locs[2]+2, 0] = 5 
        return sample 
    

class Rule11(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 7
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 6

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,2,3,4]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-2)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 2
        sample[random_locs[2], 0] = 2
        sample[random_locs[2]+1, 0] = 3
        sample[random_locs[2]+3, 0] = 4 
        return sample 
    

class Rule12(BaseRule):
    def __init__(self, 
                 num_shapes:int, num_colors:int, num_textures:int, vocab_len:int, 
                 min_seq_len:int, max_seq_len:int, is_one_hot_color:bool, is_one_hot_texture:bool, eos_token,
                 **kwargs,
                 ):
        super().__init__(
                    num_shapes, num_colors, num_textures, vocab_len,
                    min_seq_len, max_seq_len, is_one_hot_color, is_one_hot_texture,
            )
        self.eos_token = eos_token

    def _check_requirement(self):
        assert self.num_shapes > 7
        assert self.num_colors  == 0 
        assert self.num_textures == 0 
        assert self.min_seq_len > 6

    def make_sample(self, i:int):
        sample = torch.zeros(self.max_seq_len, 3).fill_(self.eos_token)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        words = [1,2,3,4]
        for s in range(1, seq_len):
            sample[s,0] = np.random.choice([i for i in range(1, self.num_shapes) if i not in words])  
        random_locs = sorted(np.random.choice([i for i in range(1, seq_len-3)], 3, replace=False))

        sample[random_locs[0], 0] = 1
        sample[random_locs[1], 0] = 2
        sample[random_locs[2], 0] = 2
        sample[random_locs[2]+2, 0] = 3
        sample[random_locs[2]+3, 0] = 4 
        return sample 
    
