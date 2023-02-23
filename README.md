# VMIPS functional simulator [![test](https://github.com/HsuanTingLu/VMIPS-functional-simulator/actions/workflows/test.yml/badge.svg)](https://github.com/HsuanTingLu/VMIPS-functional-simulator/actions/workflows/test.yml)

## Dependencies
No third party libraries needed, although pytest could be used to select test cases with regular expression.

Developed and tested with Python 3.11

## Test

Tests are written with the built-in ```unittest``` module, and the scripts are stored in the ```tests``` directory.
The ```tests/NoOp``` directory stores example input and output files, along with a ```Code.asm``` containing all instructions which will be used to provide test coverage information.
**All commands provided should be executed from project root.**

### Instruction set
These test cases are in ```tests/test_isa.py```.

The ```TestIntegratedSmallProgram``` class holds tests that test instructions with small programs, which you can find in the ```isa_test``` directory. They include 
ADD, BLE, BLT, HALT, LS, LV, SS, SUB, SUBVV, and SV,
covering 10 (21%) instructions.

The ```TestSingleInstruction``` class holds tests that focus on testing one instruction and one instruction only. They generate code and manipulate internal state directly during runtime to isolate itself from bugs in other instructions. They include
ADDVS, ADDVV, AND, BEQ, BGE, BGT, BLE, BLT, BNE, CVM, DIVVS, DIVVV, HALT, LVI, LVWS, MFCL, MTCL, MULVS, MULVV, OR, POP, SEQVS, SEQVV, SGEVS, SGEVV, SGTVS, SGTVV, SLEVS, SLEVV, SLL, SLTVS, SLTVV, SNEVS, SNEVV, SRA, SRL, SUBVS, SUBVV, SVI, SVWS, and XOR,
covering 41 (87%) instructions. (yes they overlap a little)

An report of a previous run can be found at ```instruction_test_results.txt```.
Run ```python3 -Wignore -m unittest discover tests test_isa.py -v``` to test it yourself.

Use environmental variable ```DEBUG``` to change the verbosity of runtime logs. Defaults to level 0 and will not bother you with too much detail. Try level 1 and you'll get instruction parsing info. Level 2 gives you transaction logs, including all scalar/vector register-file/memory read/writes, with reads colored in red and writes colored in blue.
We use ```DEBUG=2 python3 -Wignore -m unittest discover tests test_isa.py -v``` a lot when we're debugging our assembly, hope you like it too.

### Dot product
You can test this either with standard ```python3 main.py --iodir dot_product_test``` command and check the outputs,

or run ```python3 -Wignore -m unittest discover tests test_dot_product.py -v``` and let the program test assertions for you.

## Code

Syntax sugar has been added to let users check internal state more easily.

Access ```vcore.scalar_register_file[6]``` with ```vcore.SR7```, and the same goes for ```vector_register_file```.
Access ```vcore.vector_data_mem[0]``` with ```vcore.SR0```, and the same goes for ```scalar_data_mem```.

e.g.
- scalar register file
    - ```vcore.SR1```
    - ```vcore.SR2```
    - ```vcore.SR3```
- vector register file
    - ```vcore.VR5```
    - ```vcore.VR6```
    - ```vcore.VR7```
- scalar memory
    - ```vcore.SM0```
    - ```vcore.SM2```
    - ```vcore.SM4```
- vector memory
    - ```vcore.VM0```
    - ```vcore.VM3```
    - ```vcore.VM6```

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
where I directly map ```add/sub/mul/div``` and ```eq/ne/lt/le/gt/ge``` to native operators.
