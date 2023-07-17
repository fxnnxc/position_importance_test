from pit.dataset.mathematical_shapes import MathematicalShapesDataset



dataset = MathematicalShapesDataset(
                                      rule_indices=[0,1,2,3,4,5,6,7], 
                                      num_shapes=20,
                                      num_samples=100, 
                                      return_rule_label=True)
c_y = 0
for d in dataset:
    x, y = d 
    print(f"----- {y} -----")
    print(x)
    
