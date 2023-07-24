from pit.dataset.auto_regressive_shapes import AutoRegressiveShapesDataset



dataset = AutoRegressiveShapesDataset(rule_indices=[12], 
                                      num_samples=10, 
                                      min_seq_len=7, 
                                      max_seq_len=16, 
                                      return_rule_label=True)
c_y = 0
for d in dataset:
    x, y = d 
    print(f"----- {y} -----")
    print(x)
    
