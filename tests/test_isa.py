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
