"""Test component functionality like operations and load/dump format
"""
import array
import operator
import pathlib
import unittest

from main import Core, DataMemory, InstructionMemory, RegisterFile
from tests.utils import BaseTestWithTempDir

GoldenFiles = pathlib.Path("tests/NoOp")
assert GoldenFiles.is_dir()


class TestRegisterFile(BaseTestWithTempDir):
    """Tests the configurable register file under scalar and vector settings
    """

    def test_dump_scalar_regfile(self):
        """Tests scalar register file dump format
        """
        golden_output = GoldenFiles / "SRF.txt"
        my_output = self.temp_dir / "SRF.txt"

        regfile = RegisterFile(dump_path=my_output,
                               **Core.common_params.scalar_regfile)
        regfile.dump()

        with golden_output.open() as golden, my_output.open() as result:
            self.assertEqual(golden.read(), result.read())

    def test_dump_vector_regfile(self):
        """Tests vector register file dump format
        """
        golden_output = GoldenFiles / "VRF.txt"
        my_output = self.temp_dir / "VRF.txt"

        regfile = RegisterFile(dump_path=my_output,
                               **Core.common_params.vector_regfile)
        regfile.dump()

        with golden_output.open() as golden, my_output.open() as result:
            self.assertEqual(golden.read(), result.read())


class TestDataMemory(BaseTestWithTempDir):
    """Tests the configurable data memory under scalar and vector settings
    """

    def test_dump_scalar_memory(self):
        """Tests scalar data memory dump format
        """
        test_input = GoldenFiles / "SDMEM.txt"
        golden_output = GoldenFiles / "SDMEMOP.txt"
        my_output = self.temp_dir / "SDMEMOP.txt"

        # 32 KB is 2^15 bytes = 2^13 K 32-bit words
        data_mem = DataMemory(load_path=test_input,
                              dump_path=my_output,
                              **Core.common_params.scalar_datamem)
        data_mem.load()
        data_mem.dump()

        with golden_output.open("rb") as golden, my_output.open("rb") as result:
            self.assertEqual(golden.read(), result.read())

    def test_dump_vector_memory(self):
        """Tests vector data memory dump format
        """
        test_input = GoldenFiles / "VDMEM.txt"
        golden_output = GoldenFiles / "VDMEMOP.txt"
        my_output = self.temp_dir / "VDMEMOP.txt"

        # 512 KB is 2^19 bytes = 2^17 K 32-bit words
        data_mem = DataMemory(load_path=test_input,
                              dump_path=my_output,
                              **Core.common_params.vector_datamem)
        data_mem.load()
        data_mem.dump()

        with golden_output.open("rb") as golden, my_output.open("rb") as result:
            self.assertEqual(golden.read(), result.read())


class TestProcessorCore(BaseTestWithTempDir):
    """Tests the configurable processor core under scalar and vector settings
    """

    @classmethod
    def get_empty_core(cls, tempdir: pathlib.Path):
        """Creates a processor with empty instruction and data memory.

        Intends to be used in pure processor utils tests that don't involve
        components like register files and memories.
        """
        assert tempdir.is_dir()
        with (empty_file := tempdir / "empty.txt").open("w"):
            # Generate an empty file to load into the instruction/data memory
            # since we're not trying to do any operations here
            pass

        vcore = Core(scalar_register_file=RegisterFile(
            dump_path=None,
            **Core.common_params.scalar_regfile,
        ),
                     vector_register_file=RegisterFile(
                         dump_path=None,
                         **Core.common_params.vector_regfile,
                     ),
                     instruction_mem=InstructionMemory(load_path=empty_file),
                     scalar_data_mem=DataMemory(
                         load_path=empty_file,
                         dump_path=None,
                         **Core.common_params.scalar_datamem,
                     ),
                     vector_data_mem=DataMemory(
                         load_path=empty_file,
                         dump_path=None,
                         **Core.common_params.vector_datamem,
                     ))
        return vcore

    def test_program_counter(self):
        """Test add/sub and add/sub-assignment on the program counter
        """
        vcore = self.get_empty_core(self.temp_dir)

        # Test property initialization
        self.assertEqual(0, vcore.PC)
        self.assertEqual(0, vcore.program_counter)

        # Test assignment
        vcore.PC = 3
        self.assertEqual(3, vcore.PC)
        self.assertEqual(3, vcore.program_counter)

        # Test addition assignment
        vcore.PC += 5
        self.assertEqual(8, vcore.PC)
        self.assertEqual(8, vcore.program_counter)

        # Test property read
        vcore.program_counter += 7
        self.assertEqual(15, vcore.PC)


class TestPythonOperators(unittest.TestCase):

    def test_numbers(self):
        # positive
        a = operator.xor(0xFFFF_FFFF, 0x00FF00FF)
        b = 0xFF00FF00
        self.assertEqual(a, b)

        #
        """ for bitwise operations
        can think of it like having a 64 bit container, so it's usually bit-extended way over
        just &0xFFFF_FFFF to get the 32-bit part I want
        """
        # negative
        c = -1 & 0xFFFF_FFFF
        d = 0xFFFF_FFFF
        self.assertEqual(c, d)

    def test_logical_right_shift(self):
        # positive
        a = None
        b = None
        self.assertEqual(a, b)

        # negative
        #c = operator.rshift(-1, 4) & 0xFFFF_FFFF
        #d = operator.rshift(0xFFFF_FFFF, 4) & 0xFFFF_FFFF
        #self.assertEqual(d, 0x0FFF_FFFF)
        #self.assertEqual(c, d)
        c = (-1 % 0x1_0000_0000) >> 4
        self.assertEqual(c, 0x0FFF_FFFF)
        d = operator.rshift(-1 % 0x1_0000_0000, 4)
        self.assertEqual(d, 0x0FFF_FFFF)

        self.assertEqual(-240 % 0x1_0000_0000, 0xFFFF_FF10)
        self.assertEqual(-240 & 0xFFFF_FFFF, 0xFFFF_FF10)
        e = operator.rshift(-240 % 0x1_0000_0000, 4)
        self.assertEqual(e, 0x0FFF_FFF1)

        # TODO: verdict : &0xFFFF_FFFF BEFORE SHIFT to get logical-right-shift

    def test_logical_left_shift(self):
        # negative
        c = operator.lshift(-1, 4) & 0xFFFF_FFFF
        self.assertEqual(c, 0xFFFF_FFF0)
        d = operator.lshift(-240, 4) & 0xFFFF_FFFF
        self.assertEqual(d, 0xFFFF_F100)

    def test_arithmetic_right_shift(self):
        # positive
        a = operator.rshift(0xF0, 4) & 0xFFFF_FFFF  # no need to be masked
        self.assertEqual(a, 0x0000_000F)

        # negative  # NOTE: MUST MASK
        c = operator.rshift(-4, 1) & 0xFFFF_FFFF  # NOTE: MUST MASK
        self.assertEqual(c, 0xFFFF_FFFE)
        # -240 (-0xF0): 2's-comp : 0xFFFF_FF10
        e = operator.rshift(-240, 4) & 0xFFFF_FFFF  # NOTE: MUST MASK
        self.assertEqual(e, 0xFFFF_FFF1)
        f = operator.rshift(0xFFFF_FFFF, 4) & 0xFFFF_FFFF
        self.assertEqual(f, 0x0FFF_FFFF)

        # TODO: verdict: MASK RESULTS OF arithmetic-right-shift


class TestArray(unittest.TestCase):

    def test_array(self):
        self.assertEqual(4, array.array("i").itemsize)
        self.assertEqual(8, array.array("l").itemsize)

        with self.assertRaisesRegex(OverflowError,
                                    "signed integer is greater than maximum"):
            return array.array("i", [2147483647, 2147483648, -1, -2147483648])

        with self.assertRaisesRegex(OverflowError,
                                    "signed integer is less than minimum"):
            return array.array("i", [2147483647, 1, -1, -2147483649])

        sint32_array = array.array("i", [2147483647, 1, -1, -2147483648])
        print()
        print(f"{sint32_array = }")
        print(f"{sint32_array.tolist() = }")
        print(f"{sint32_array.tobytes() = }")
        print(f"{sint32_array[0] = }")
        print(f"{type(sint32_array[0]) = }")
        print(f"{sint32_array[3] = }")
        print(f"{type(sint32_array[3]) = }")
