from pit.dataset.mathematical_shapes import MathematicalShapesDataset
import numpy as np
import pandas as pd

def save_vector_as_csv(vector, output_file):
    with open(output_file, 'a') as f:
        f.write(' '.join(map(str, vector)) + '\n')

def process_matrix_and_save_as_csv(matrix, output_file_train, output_file_test, num_shapes):
    vector = []
    for row in matrix:
        element = int(row[0:1]) 
        vector.append(element)
    if (vector[2]==(num_shapes+2) and (vector[1]+vector[3]==100) and vector[1]>vector[3] and vector[3]<51): #rule 0
        save_vector_as_csv(vector, output_file_test)
    if (vector[4]==(num_shapes+4) and (vector[1]-vector[3]==50 or vector[1]-vector[3]==-50) and vector[1]<100): #rule 4
        save_vector_as_csv(vector, output_file_test)
    save_vector_as_csv(vector, output_file_train)
    print(f"vector: {vector}")
num_shapes=102
dataset = MathematicalShapesDataset(
                                      rule_indices=[0], 
                                      num_shapes=num_shapes,
                                      num_samples=1000000, 
                                      return_rule_label=True)
c_y = 0
# for d in dataset:
#     x, y = d 
#     print(f"----- {y} -----")
#     print(x)
    
output_file_train = "mathematical_shapes_100_train.csv"
output_file_test = "mathematical_shapes_100_test.csv"

for d in dataset:
    x, y = d 
    matrix = np.array(x)  
    process_matrix_and_save_as_csv(matrix, output_file_train, output_file_test, num_shapes )
print("saved")