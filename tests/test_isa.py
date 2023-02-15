"""Test instruction functionalities
"""
import inspect
import pathlib
import re
import unittest

from main import Core, DataMemory, InstructionMemory, RegisterFile
from tests.utils import BaseTestWithTempDir

golden_dir = pathlib.Path("isa_test")
assert golden_dir.is_dir()

instructions_used = set()  # Log instruction used


def setUpModule():
    pass


def tearDownModule():
    # Report what instructions are tested and what aren't
    instruction_container = InstructionMemory(load_path="tests/NoOp/Code.asm")
    instruction_container.load()
    whole_instr = set(
        Core.decode(instr)["instruction"]
        for instr in instruction_container.instructions)
    missing_instructions = whole_instr - instructions_used
    print("-" * 80)
    print("INSTRUCTION TEST COVERAGE REPORT")
    print("-" * 80)
    print(f"{sorted(instructions_used) = }")
    print(f"{sorted(missing_instructions) = }")
    print("-" * 80)
    print(f"coverage: {len(instructions_used)/len(whole_instr)*100:2.3}%")
    print("-" * 80)


def gather_stats(core: Core):
    """Gather instruction test coverage
    """
    instructions_used.update(
        set(instr for instr in [
            core.decode(line)["instruction"]
            for line in core.instruction_mem.instructions
        ] if instr is not None))


def get_core(in_dir: pathlib.Path, out_dir: pathlib.Path, file_prefix: str):
    """Creates a processor
    with instruction and data memory loaded from designated locations
    """
    assert in_dir.is_dir() and out_dir.is_dir()

    vcore = Core(
        scalar_register_file=RegisterFile(
            dump_path=out_dir / f"{file_prefix}_SRF.txt",
            **Core.common_params.scalar_regfile,
        ),
        vector_register_file=RegisterFile(
            dump_path=out_dir / f"{file_prefix}_VRF.txt",
            **Core.common_params.vector_regfile,
        ),
        instruction_mem=InstructionMemory(load_path=in_dir /
                                          f"{file_prefix}.asm"),
        scalar_data_mem=DataMemory(
            load_path=in_dir / f"{file_prefix}_SDMEM.txt",
            dump_path=out_dir / f"{file_prefix}_SDMEMOP.txt",
            **Core.common_params.scalar_datamem,
        ),
        vector_data_mem=DataMemory(
            load_path=in_dir / f"{file_prefix}_VDMEM.txt",
            dump_path=out_dir / f"{file_prefix}_VDMEMOP.txt",
            **Core.common_params.vector_datamem,
        ),
    )
    return vcore


class TestSingleInstruction(BaseTestWithTempDir):
    """Test each instructions, one at a time
    """
    _parse_test_name_regex = re.compile(r"^test_\d+_(?P<instr>\w+)$", re.ASCII)

    @classmethod
    def current_instruction(cls):
        return cls._parse_test_name_regex.match(
            inspect.currentframe().f_back.f_code.co_name)["instr"]

    @classmethod
    def generate(cls, tempdir: pathlib.Path, instruction: str, code: str,
                 scalar_mem: list[int], vector_mem: list[int]) -> str:
        """Generates code with template
        """
        assert instruction in code, f"instruction {instruction} is not in the code: '{instruction}'"
        file_prefix = f"single_instr_test_{instruction}"

        code_file = (tempdir / f"{file_prefix}.asm").open(mode="w")
        code_file.write(code + "\n")
        code_file.flush()

        scalar_mem_file = (tempdir / f"{file_prefix}_SDMEM.txt").open(mode="w")
        scalar_mem_file.writelines([str(i) for i in scalar_mem])
        scalar_mem_file.flush()

        vector_mem_file = (tempdir / f"{file_prefix}_VDMEM.txt").open(mode="w")
        vector_mem_file.writelines([str(i) for i in vector_mem])
        vector_mem_file.flush()

        return code_file, scalar_mem_file, vector_mem_file

    def test_1_ADDVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="ADDVV VR3 VR1 VR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = list(range(64))
        vcore.vector_register_file[1] = list(reversed(range(64)))
        vcore.run()
        self.assertEqual([63] * 64, vcore.VR3)
        gather_stats(vcore)

    def test_1_SUBVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="SUBVV VR3 VR1 VR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = list(range(0, 64))
        vcore.vector_register_file[1] = list(range(1, 65))
        vcore.run()
        self.assertEqual([-1] * 64, vcore.VR3)
        gather_stats(vcore)

    def test_2_ADDVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="ADDVS VR3 VR1 SR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = list(range(3, 67))
        vcore.scalar_register_file[1] = -2
        vcore.run()
        self.assertEqual(list(range(1, 65)), vcore.VR3)
        gather_stats(vcore)

    def test_2_SUBVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="SUBVS VR3 VR1 SR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = list(range(0, 64))
        vcore.scalar_register_file[1] = 63
        vcore.run()
        self.assertEqual(list(range(-63, 1)), vcore.VR3)
        gather_stats(vcore)

    def test_3_MULVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="MULVV VR3 VR1 VR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = list(range(64))
        vcore.vector_register_file[1] = list(range(64))
        vcore.run()
        self.assertEqual([i * i for i in range(64)], vcore.VR3)
        gather_stats(vcore)

    def test_3_DIVVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="DIVVV VR3 VR1 VR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = [i * i for i in range(1, 65)]
        vcore.vector_register_file[1] = list(range(1, 65))
        vcore.run()
        self.assertEqual(list(range(1, 65)), vcore.VR3)
        gather_stats(vcore)

    def test_4_MULVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="MULVS VR3 VR1 SR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = list(range(64))
        vcore.scalar_register_file[1] = 2
        vcore.run()
        self.assertEqual([i * 2 for i in range(64)], vcore.VR3)
        gather_stats(vcore)

    def test_4_DIVVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="DIVVS VR3 VR1 SR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.vector_register_file[0] = [i for i in range(64)]
        vcore.scalar_register_file[1] = 2
        vcore.run()
        self.assertEqual([i // 2 for i in range(64)], vcore.VR3)
        gather_stats(vcore)

    @unittest.skip("TODO")
    def test_5_SEQVV(self):
        pass  # @todo Test SEQVV

    @unittest.skip("TODO")
    def test_5_SNEVV(self):
        pass  # @todo Test SNEVV

    @unittest.skip("TODO")
    def test_5_SGTVV(self):
        pass  # @todo Test SGTVV

    @unittest.skip("TODO")
    def test_5_SLTVV(self):
        pass  # @todo Test SLTVV

    @unittest.skip("TODO")
    def test_5_SGEVV(self):
        pass  # @todo Test SGEVV

    @unittest.skip("TODO")
    def test_5_SLEVV(self):
        pass  # @todo Test SEQVV

    @unittest.skip("TODO")
    def test_6_SEQVS(self):
        pass  # @todo Test SEQVS

    @unittest.skip("TODO")
    def test_6_SNEVS(self):
        pass  # @todo Test SNEVS

    @unittest.skip("TODO")
    def test_6_SGTVS(self):
        pass  # @todo Test SGTVS

    @unittest.skip("TODO")
    def test_6_SLTVS(self):
        pass  # @todo Test SLTVS

    @unittest.skip("TODO")
    def test_6_SGEVS(self):
        pass  # @todo Test SGEVS

    @unittest.skip("TODO")
    def test_6_SLEVS(self):
        pass  # @todo Test SLEVS

    @unittest.skip("TODO")
    def test_7_CVM(self):
        pass  # @todo Test CVM

    @unittest.skip("TODO")
    def test_8_POP(self):
        pass  # @todo Test POP

    @unittest.skip("TODO")
    def test_9_MTCL(self):
        pass  # @todo Test MTCL

    @unittest.skip("TODO")
    def test_10_MFCL(self):
        pass  # @todo Test MFCL

    @unittest.skip("TODO")
    def test_11_LV(self):
        pass  # @todo Test LV

    @unittest.skip("TODO")
    def test_12_SV(self):
        pass  # @todo Test SV

    @unittest.skip("TODO")
    def test_13_LVWS(self):
        pass  # @todo Test LVWS

    @unittest.skip("TODO")
    def test_14_SVWS(self):
        pass  # @todo Test SVWS

    @unittest.skip("TODO")
    def test_15_LVI(self):
        pass  # @todo Test LVI

    @unittest.skip("TODO")
    def test_16_SVI(self):
        pass  # @todo Test SVI

    @unittest.skip("TODO")
    def test_17_LS(self):
        pass  # @todo Test LS

    @unittest.skip("TODO")
    def test_18_SS(self):
        pass  # @todo Test SS

    @unittest.skip("TODO")
    def test_19_ADD(self):
        pass  # @todo Test ADD

    @unittest.skip("TODO")
    def test_19_SUB(self):
        instruction = self.current_instruction()
        code, scalar_mem = self.generate(
            self.temp_dir,
            instruction,
            code="SUB SR3 SR1 SR2",
            scalar_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vcore.scalar_register_file[0] = list(range(0, 64))
        vcore.scalar_register_file[1] = list(range(1, 65))
        vcore.run()
        self.assertEqual([-1]*64, vcore.SR3)
        gather_stats(vcore)

    @unittest.skip("TODO")
    def test_20_AND(self):
        pass  # @todo Test AND

    @unittest.skip("TODO")
    def test_20_OR(self):
        pass  # @todo Test OR

    @unittest.skip("TODO")
    def test_20_XOR(self):
        pass  # @todo Test XOR

    @unittest.skip("TODO")
    def test_21_SLL(self):
        pass  # @todo Test SLL

    @unittest.skip("TODO")
    def test_21_SRL(self):
        pass  # @todo Test SRL

    @unittest.skip("TODO")
    def test_22_SRA(self):
        pass  # @todo Test SRA

    def test_23_BEQ(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BEQ SR1 SR2 1"
                  "\n"
                  "ABRACADABRA"
                  "\n"
                  "HALT"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: SR1 > SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 3
        vcore.scalar_register_file[1] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 2
        vcore.scalar_register_file[1] = 2
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 1
        vcore.scalar_register_file[1] = 3
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)

    def test_23_BNE(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BNE SR1 SR2 1"
                  "\n"
                  "ABRACADABRA"
                  "\n"
                  "HALT"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: SR1 > SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 3
        vcore.scalar_register_file[1] = 1
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 2
        vcore.scalar_register_file[1] = 2
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 1
        vcore.scalar_register_file[1] = 3
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)

    def test_23_BGT(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BGT SR1 SR2 1"
                  "\n"
                  "ABRACADABRA"
                  "\n"
                  "HALT"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: SR1 > SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 3
        vcore.scalar_register_file[1] = 1
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 2
        vcore.scalar_register_file[1] = 2
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 1
        vcore.scalar_register_file[1] = 3
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)

    def test_23_BLT(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BLT SR1 SR2 1"
                  "\n"
                  "ABRACADABRA"
                  "\n"
                  "HALT"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: SR1 > SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 3
        vcore.scalar_register_file[1] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 2
        vcore.scalar_register_file[1] = 2
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 1
        vcore.scalar_register_file[1] = 3
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)

    def test_23_BGE(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BGE SR1 SR2 1"
                  "\n"
                  "ABRACADABRA"
                  "\n"
                  "HALT"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: SR1 > SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 3
        vcore.scalar_register_file[1] = 1
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 2
        vcore.scalar_register_file[1] = 2
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 1
        vcore.scalar_register_file[1] = 3
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)

    def test_23_BLE(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BLE SR1 SR2 1"
                  "\n"
                  "ABRACADABRA"
                  "\n"
                  "HALT"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: SR1 > SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 3
        vcore.scalar_register_file[1] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 2
        vcore.scalar_register_file[1] = 2
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[0] = 1
        vcore.scalar_register_file[1] = 3
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)

    def test_24_HALT(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("HALT "
                  "\n"
                  "ABRACADABRA"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)


class TestIntegratedSmallProgram(BaseTestWithTempDir):
    """Run small programs that uses multiple instructions
    """

    def test_accumulate_array(self):
        """Test accumulate values in an array
        """
        test_prefix = "accumulate"
        vcore = get_core(golden_dir, self.temp_dir, test_prefix)

        vcore.run()
        self.assertEqual(vcore.SR5, 55)

        gather_stats(vcore)

    def test_scalar_load_store(self):
        """Test scalar load store instructions
        """
        test_prefix = "scalar_load_store"
        vcore = get_core(golden_dir, self.temp_dir, test_prefix)

        # Test load
        # source before operation
        self.assertEqual(vcore.SM0, 3)
        self.assertEqual(vcore.SM1, 5)
        self.assertEqual(vcore.SM2, 7)
        self.assertEqual(vcore.SM3, 11)
        # destination before operation
        self.assertEqual(vcore.SR2, 0)
        self.assertEqual(vcore.SR3, 0)
        self.assertEqual(vcore.SR4, 0)
        self.assertEqual(vcore.SR5, 0)
        vcore.step_instr()
        vcore.step_instr()
        vcore.step_instr()
        vcore.step_instr()
        # destination after operation
        self.assertEqual(vcore.SR2, 3)
        self.assertEqual(vcore.SR3, 5)
        self.assertEqual(vcore.SR4, 7)
        self.assertEqual(vcore.SR5, 11)

        # Test store
        # source before operation is previous destination-after-operation
        # destination before operation
        self.assertEqual(vcore.SM4, 0)
        self.assertEqual(vcore.SM5, 0)
        self.assertEqual(vcore.SM6, 0)
        self.assertEqual(vcore.SM7, 0)
        vcore.step_instr()
        vcore.step_instr()
        vcore.step_instr()
        vcore.step_instr()
        # destination after operation
        self.assertEqual(vcore.SM4, 3)
        self.assertEqual(vcore.SM5, 5)
        self.assertEqual(vcore.SM6, 7)
        self.assertEqual(vcore.SM7, 11)

        gather_stats(vcore)

    def test_scalar_add_sub(self):
        """Test scalar add/subtract
        """
        test_prefix = "add_sub"
        vcore = get_core(golden_dir, self.temp_dir, test_prefix)

        vcore.run()

        self.assertEqual(5, vcore.SR7)
        self.assertEqual(-5, vcore.SR8)

        gather_stats(vcore)

    def test_vector_load_store(self):
        """Test vector load store instructions
        """
        test_prefix = "vector_load_store"
        vcore = get_core(golden_dir, self.temp_dir, test_prefix)

        # Test load
        # source before operation
        self.assertEqual(vcore.vector_data_mem[0:64], list(range(0, 64)))
        # destination before operation
        self.assertEqual(vcore.VR1, [0] * 64)
        vcore.step_instr()
        # destination after operation
        self.assertEqual(vcore.VR1, list(range(0, 64)))

        # Test store
        # source before operation is previous destination-after-operation
        # destination before operation
        self.assertEqual(vcore.vector_data_mem[64:64 * 2], [0] * 64)
        vcore.step_instr()
        vcore.step_instr()
        # destination after operation
        self.assertEqual(vcore.vector_data_mem[64:64 * 2], list(range(0, 64)))

        gather_stats(vcore)

    def test_vector_add_sub(self):
        """Test vector add/subtract
        """
        test_prefix = "vector_add_sub"
        vcore = get_core(golden_dir, self.temp_dir, test_prefix)

        # destination before operation
        self.assertEqual(vcore.vector_data_mem[64 * 2:64 * 3], [0] * 64)

        vcore.run()

        # destination after operation
        self.assertEqual(vcore.vector_data_mem[64 * 2:64 * 3], [1] * 64)

        gather_stats(vcore)

    def test_halt(self):
        """Test if program stops before EOF after a HALT
        """
        test_prefix = "halt"
        vcore = get_core(golden_dir, self.temp_dir, test_prefix)

        # Test load
        # destination before operation
        self.assertEqual(vcore.SR2, 0)

        vcore.run()  # should stop automatically

        # force even some more steps
        vcore.step()
        vcore.step()
        vcore.step()

        # destination after operation: LS->SR2 SHOULDN'T BE EXECUTED
        self.assertEqual(vcore.SR2, 0)

        gather_stats(vcore)
