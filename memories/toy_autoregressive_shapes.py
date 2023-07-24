from pit.dataset.auto_regressive_shapes import AutoRegressiveShapesDataset
import numpy as np
import pandas as pd

def save_vector_as_csv(vector, output_file):
    with open(output_file, 'a') as f:
        f.write(','.join(map(str, vector)) + '\n')

def process_matrix_and_save_as_csv(matrix, output_file):
    vector = []
    for row in matrix:
        element = int(row[0:1]) 
        vector.append(element)
    save_vector_as_csv(vector, output_file)
    print(f"vector: {vector}")

dataset = AutoRegressiveShapesDataset(rule_indices=[5,6,7,8,9,10,11,12], 
                                      num_samples=1000, 
                                      min_seq_len=7, 
                                      max_seq_len=16, 
                                      return_rule_label=True)
c_y = 0
# for d in dataset:
#     x, y = d 
#     print(f"----- {y} -----")
#     print(x)
    
output_file = "autoregressive_shape_test.csv"

for d in dataset:
    x, y = d 
    matrix = np.array(x)  
    process_matrix_and_save_as_csv(matrix, output_file)
print("saved")