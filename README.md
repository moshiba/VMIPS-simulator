# VMIPS-functional-simulator

## Test

```
DEBUG=2 python3 -W ignore::ResourceWarning -m unittest discover tests "test_*.py" -v
```

Instruction execution is delegated to the ```ALU``` class, in which I grouped instructions with operation type:
| handler name | operation type |
|--- | ---|
| ```vector_op```    | vector operations |
| ```vec_mask_reg``` | vector mask register operations |
| ```vec_len_reg```  | vector length register (flag register) operations |
| ```mem_op```       | memory operations |
| ```scalar_op```    | scalar operations |
| ```control```      | flow control |
| ```stop```         | halt |

And most instruction handlers are structured with these following stages:
1. Set common aliases
2. Map operation name to standard operators
3. Get operands
4. Do operation and store the result

Step 2 and 3 are optional, but they looks really clean on arithmetics and branching,
where I directly map ```add/sub/mul/div/lshift/rshift``` and ```eq/ne/lt/le/gt/ge``` to native operators.
I'm invoking the built-in arithmetic and ordering operators by name, which is dynamically selected by parsing the name of the instruction.

