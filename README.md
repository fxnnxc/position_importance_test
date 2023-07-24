# position_importance_test
testing the position importance with synthetic dataset





```
- mdoel 
- dataset
   - base_shapes --> Make dataset (rules --> dataset)
   - mathematical_shapes --> _make_dataset (상속해서 구현 Rule0, Rule4 합해서 100이 되는 숫자들)


```


### Mathematical Shapes 

0. Addition `a` + `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
1. Addition `a` + `b` & `c` (mod `m`).  [`a + b & m = c`]
2. Multiplication `a` * `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
3. Multiplication `a` * `b` & `c` (mod `m`).  [`a * b & m = c`]
4. Subtraction `a` - `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
5. Subtraction `a` - `b` & `c` (mod `m`).  [`a - b & m = c`]
6. Division `a` / `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
7. Division `a` / `b` & `c` (mod `m`).  [`a / b & m = c`]