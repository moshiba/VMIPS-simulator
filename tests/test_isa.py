"""Test instruction functionalities
"""
import pathlib
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


class TestProcessorCore(BaseTestWithTempDir):
    """Tests the configurable processor core
    """

    @classmethod
    def get_core(cls, tempdir: pathlib.Path, file_prefix: str):
        """Creates a processor
        with instruction and data memory loaded from designated locations
        """
        assert tempdir.is_dir()

        vcore = Core(scalar_register_file=RegisterFile(
            dump_path=None,
            **Core.common_params.scalar_regfile,
        ),
                     vector_register_file=RegisterFile(
                         dump_path=None,
                         **Core.common_params.vector_regfile,
                     ),
                     instruction_mem=InstructionMemory(load_path=golden_dir /
                                                       f"{file_prefix}.asm"),
                     scalar_data_mem=DataMemory(
                         load_path=golden_dir / f"{file_prefix}_SDMEM.txt",
                         dump_path=None,
                         **Core.common_params.scalar_datamem,
                     ),
                     vector_data_mem=DataMemory(
                         load_path=golden_dir / f"{file_prefix}_VDMEM.txt",
                         dump_path=None,
                         **Core.common_params.vector_datamem,
                     ))
        return vcore

    def test_accumulate_array(self):
        """Test accumulate values in an array
        """
        test_prefix = "accumulate"
        vcore = self.get_core(self.temp_dir, test_prefix)

        instr = ""
        while instr != "HALT":
            current_line = vcore.instruction_mem[vcore.PC]
            instr = vcore.decode(current_line)["instruction"]
            if instr is not None:
                instructions_used.add(instr)

            # Do work
            vcore.step()

        print(vcore.scalar_register_file._data)

        self.assertEqual(vcore.SR5, 55)
