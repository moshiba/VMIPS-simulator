"""dot product test
sum(square(i)) for i from 0 to 449 = 30273825
"""
import pathlib
import unittest

from main import Core, DataMemory, InstructionMemory, RegisterFile

golden_dir = pathlib.Path("dot_product_test")
assert golden_dir.is_dir()


class TestDotProduct(unittest.TestCase):

    def test_450_dot_product(self):
        vcore = Core(
            scalar_register_file=RegisterFile(
                dump_path=golden_dir / "SRF.txt",
                **Core.common_params.scalar_regfile,
            ),
            vector_register_file=RegisterFile(
                dump_path=golden_dir / "VRF.txt",
                **Core.common_params.vector_regfile,
            ),
            instruction_mem=InstructionMemory(load_path=golden_dir /
                                              "Code.asm"),
            scalar_data_mem=DataMemory(
                load_path=golden_dir / "SDMEM.txt",
                dump_path=golden_dir / "SDMEMOP.txt",
                **Core.common_params.scalar_datamem,
            ),
            vector_data_mem=DataMemory(
                load_path=golden_dir / "VDMEM.txt",
                dump_path=golden_dir / "VDMEMOP.txt",
                **Core.common_params.vector_datamem,
            ),
        )

        vcore.run()

        print(f"{vcore.vector_register_file.vector_length_register = }")
        print(f"{vcore.VM2048 = }")

        self.assertEqual(sum(i * i for i in range(450)),
                         vcore.vector_data_mem[2048])

        vcore.dump()
