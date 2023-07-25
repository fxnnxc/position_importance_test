import numpy as np

def converter(mathmathicalshapes, tensor):
    converted = []
    vocab_len = mathmathicalshapes.shape_vocab_len 
    for element in tensor:
        if (element == vocab_len):
            converted.append('=')
        elif (element == vocab_len+2):
            converted.append('+')
        elif (element == vocab_len+4):
            converted.append('-')
        elif (element == vocab_len+6):
            converted.append('neg')
        elif (element == vocab_len+7):
            continue
        else:
            converted.append(str(element.item()))

    return ' '.join(converted)
        

        # self.equal_token     = self.shape_vocab_len
        # self.mod_equal_token = self.shape_vocab_len + 1 
        # self.add_token       = self.shape_vocab_len + 2
        # self.multiply_token  = self.shape_vocab_len + 3
        # self.subtract_token  = self.shape_vocab_len + 4
        # self.divide_token    = self.shape_vocab_len + 5
        # self.negative_token  = self.shape_vocab_len + 6