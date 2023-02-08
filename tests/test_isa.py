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

    vcore = Core(scalar_register_file=RegisterFile(
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
                 ))
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
