from pit.dataset.mathematical_shapes import MathematicalShapesDataset
from converter import converter
import numpy as np
import pandas as pd

def save_vector_as_csv(vector, output_file):
    with open(output_file, 'a') as f:
        f.write(' '.join(map(str, vector)) + '\n')

def process_matrix_and_save_as_csv(matrix, output_file_train, output_file_test, num_shapes):
    

    save_vector_as_csv(matrix, output_file_train)
    # print(f"vector: {vector}")
num_shapes=102
train_dataset = MathematicalShapesDataset( 
                                      train=True,
                                      rule_indices=[0,4], 
                                      num_shapes=num_shapes,
                                      return_rule_label=True)

test_dataset = MathematicalShapesDataset(  
                                      train=False,
                                      rule_indices=[0,4], 
                                      num_shapes=num_shapes,
                                      return_rule_label=True)
c_y = 0
# for d in dataset:
#     x, y = d 
#     print(f"----- {y} -----")
#     print(x)
    
output_file_train = "train.csv"
output_file_test = "test.csv"

print(f"number of trainset samples : {train_dataset.num_samples}")
print(f"number of testset samples : {test_dataset.num_samples}")


for d in train_dataset:
    x = d['input_ids']
    print("converted: ")
    print(converter(train_dataset, x))
    matrix = np.array(x)  
    process_matrix_and_save_as_csv(matrix, output_file_train, output_file_test, num_shapes )
for d in test_dataset:
    x = d['input_ids'] 
    print("converted: ")
    print(converter(train_dataset, x))
    matrix = np.array(x)  
    process_matrix_and_save_as_csv(matrix, output_file_test, output_file_test, num_shapes )

print("saved")