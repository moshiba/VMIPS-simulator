"""Test component functionality like operations and load/dump format
"""
import pathlib
import tempfile
import unittest

import main

GoldenFiles = pathlib.Path("tests/NoOp")
assert GoldenFiles.is_dir()


class BaseTestWithTempDir(unittest.TestCase):
    """Base class that creates a temporary directory for tests to operate on
    """

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = pathlib.Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


class TestRegisterFile(BaseTestWithTempDir):

    def test_dump_scalar_regfile(self):
        golden_output = GoldenFiles / "SRF.txt"
        my_output = self.temp_dir / "SRF.txt"

        regfile = main.RegisterFile(dump_path=my_output,
                                    n_reg=8,
                                    vec_size=1,
                                    word_size=32)
        regfile.dump()

        with golden_output.open() as golden, my_output.open() as result:
            self.assertEqual(golden.read(), result.read())

    def test_dump_vector_regfile(self):
        golden_output = GoldenFiles / "VRF.txt"
        my_output = self.temp_dir / "VRF.txt"

        regfile = main.RegisterFile(dump_path=my_output,
                                    n_reg=8,
                                    vec_size=64,
                                    word_size=32)
        regfile.dump()

        with golden_output.open() as golden, my_output.open() as result:
            self.assertEqual(golden.read(), result.read())


class TestDataMemory(BaseTestWithTempDir):

    def test_dump_scalar_memory(self):
        test_input = GoldenFiles / "SDMEM.txt"
        golden_output = GoldenFiles / "SDMEMOP.txt"
        my_output = self.temp_dir / "SDMEMOP.txt"

        # 32 KB is 2^15 bytes = 2^13 K 32-bit words
        data_mem = main.DataMemory(load_path=test_input,
                                   dump_path=my_output,
                                   address_length=13)
        data_mem.load()
        data_mem.dump()

        with golden_output.open("rb") as golden, my_output.open("rb") as result:
            self.assertEqual(golden.read(), result.read())

    def test_dump_vector_memory(self):
        test_input = GoldenFiles / "VDMEM.txt"
        golden_output = GoldenFiles / "VDMEMOP.txt"
        my_output = self.temp_dir / "VDMEMOP.txt"

        # 512 KB is 2^19 bytes = 2^17 K 32-bit words
        data_mem = main.DataMemory(load_path=test_input,
                                   dump_path=my_output,
                                   address_length=17)
        data_mem.load()
        data_mem.dump()

        with golden_output.open("rb") as golden, my_output.open("rb") as result:
            self.assertEqual(golden.read(), result.read())
