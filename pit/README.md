# Toy Models with Shapes 

<img src="../../../assets/shapes.png" width=200px>

# Autoregressive Shapes 

star `*` is any number and we use `0` as a token for padding, start, and end. 

0. `[...11...]`
1. `[...12...]`
2. `[...23...]`
3. `[...34...]`
4. `[...45...]`
5. `[..1...2...21..]`
6. `[..1...3...34..]`
7. `[..1...4...45..]`
8. `[..1...5...567..]`
9. `[..1...5...15..]`
10. `[..1...5...1*5..]`
11. `[..1...2...23*4..]`
12. `[..1...2...2*34..]`

# Mathematical Shapes 

0. Addition `a` + `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
1. Addition `a` + `b` & `c` (mod `m`).  [`a + b & m = c`]
2. Multiplication `a` * `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
3. Multiplication `a` * `b` & `c` (mod `m`).  [`a * b & m = c`]
4. Subtraction `a` - `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
5. Subtraction `a` - `b` & `c` (mod `m`).  [`a - b & m = c`]
6. Division `a` / `b` = `c` (in numeric manner). Note the `a`,`b`,`c` are chosen to hold the maximum operation. 
7. Division `a` / `b` & `c` (mod `m`).  [`a / b & m = c`]