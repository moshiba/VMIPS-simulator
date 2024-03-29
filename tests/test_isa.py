"""Test instruction functionalities
"""
import inspect
import itertools
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
    every_instr = set(
        Core.decode(instr)["instruction"]
        for instr in instruction_container.instructions)
    global instructions_used
    instructions_used = instructions_used - set(["ABRACADABRA"])

    missing_instructions = every_instr - instructions_used
    print("-" * 80)
    print("INSTRUCTION TEST COVERAGE REPORT")
    print("-" * 80)
    print(f"{sorted(instructions_used) = }")
    print(f"{sorted(missing_instructions) = }")
    print("-" * 80)
    count_used = len(instructions_used)
    count_every = len(every_instr)
    print("test coverage: "
          f"{count_used}/{count_every}"
          f" ({count_used/count_every*100:.2f}%)"
          " instructions")
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


def twos_complement(value: int) -> int:
    sign = -1 if (value & (1 << 31)) else 1
    return sign * (((value ^ 0xFFFF_FFFF) + 1) & 0xFFFF_FFFF)


max_int32 = 2147483647
min_int32 = -2147483648
assert twos_complement(0) == 0
assert twos_complement(0xFFFF_FFFF) == -1


class TestSingleInstruction(BaseTestWithTempDir):
    """Test each instructions, one at a time
    """
    _parse_test_name_regex = re.compile(r"^test_\d+_(?P<instr>\w+)$", re.ASCII)

    @classmethod
    def current_instruction(cls):
        return cls._parse_test_name_regex.match(
            inspect.currentframe().f_back.f_code.co_name)["instr"]

    @classmethod
    def generate(cls, tempdir: pathlib.Path, instruction: str, *, code: str,
                 scalar_mem: list[int], vector_mem: list[int]) -> str:
        """Generates code with template
        """
        assert instruction in code, f"instruction {instruction} is not in the code: '{instruction}'"
        file_prefix = f"single_instr_test_{instruction}"

        code_file = (tempdir / f"{file_prefix}.asm").open(mode="w")
        code_file.write(code + "\n")
        code_file.flush()

        scalar_mem_file = (tempdir / f"{file_prefix}_SDMEM.txt").open(mode="w")
        scalar_mem_file.writelines("\n".join([str(i) for i in scalar_mem]))
        scalar_mem_file.flush()

        vector_mem_file = (tempdir / f"{file_prefix}_VDMEM.txt").open(mode="w")
        vector_mem_file.writelines("\n".join([str(i) for i in vector_mem]))
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

        vcore.vector_register_file[1] = list(range(64))
        vcore.vector_register_file[2] = list(reversed(range(64)))
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

        vcore.vector_register_file[1] = list(range(0, 64))
        vcore.vector_register_file[2] = list(range(1, 65))
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

        vcore.vector_register_file[1] = list(range(3, 67))
        vcore.scalar_register_file[2] = -2
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

        vcore.vector_register_file[1] = list(range(0, 64))
        vcore.scalar_register_file[2] = 63
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

        vcore.vector_register_file[1] = list(range(64))
        vcore.vector_register_file[2] = list(range(64))
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

        vcore.vector_register_file[1] = [i * i for i in range(1, 65)]
        vcore.vector_register_file[2] = list(range(1, 65))
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

        vcore.vector_register_file[1] = list(range(64))
        vcore.scalar_register_file[2] = 2
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

        vcore.vector_register_file[1] = [i for i in range(64)]
        vcore.scalar_register_file[2] = 2
        vcore.run()
        self.assertEqual([i // 2 for i in range(64)], vcore.VR3)
        gather_stats(vcore)

    def test_5_SEQVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SEQVV VR1 VR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: VR1-0to7 > VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(1, 9))  # (1)~(8)
        vcore.vector_register_file[2][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.run()
        self.assertEqual([0] * 8 + [1] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 2: VR1-0to7 = VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.vector_register_file[2][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.run()
        self.assertEqual([1] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 3: VR1-0to7 < VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.vector_register_file[2][:8] = list(range(1, 9))  # (1)~(8)
        vcore.run()
        self.assertEqual([0] * 8 + [1] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_5_SNEVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SNEVV VR1 VR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: VR1-0to7 > VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(1, 9))  # (1)~(8)
        vcore.vector_register_file[2][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.run()
        self.assertEqual([1] * 8 + [0] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 2: VR1-0to7 = VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.vector_register_file[2][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.run()
        self.assertEqual([0] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 3: VR1-0to7 < VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.vector_register_file[2][:8] = list(range(1, 9))  # (1)~(8)
        vcore.run()
        self.assertEqual([1] * 8 + [0] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_5_SGTVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SGTVV VR1 VR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: VR1-0to7 > VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(1, 9))  # (1)~(8)
        vcore.vector_register_file[2][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.run()
        self.assertEqual([1] * 8 + [0] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 2: VR1-0to7 = VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.vector_register_file[2][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.run()
        self.assertEqual([0] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 3: VR1-0to7 < VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.vector_register_file[2][:8] = list(range(1, 9))  # (1)~(8)
        vcore.run()
        self.assertEqual([0] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_5_SLTVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SLTVV VR1 VR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: VR1-0to7 > VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(1, 9))  # (1)~(8)
        vcore.vector_register_file[2][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.run()
        self.assertEqual([0] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 2: VR1-0to7 = VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.vector_register_file[2][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.run()
        self.assertEqual([0] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 3: VR1-0to7 < VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.vector_register_file[2][:8] = list(range(1, 9))  # (1)~(8)
        vcore.run()
        self.assertEqual([1] * 8 + [0] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_5_SGEVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SGEVV VR1 VR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: VR1-0to7 > VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(1, 9))  # (1)~(8)
        vcore.vector_register_file[2][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.run()
        self.assertEqual([1] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 2: VR1-0to7 = VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.vector_register_file[2][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.run()
        self.assertEqual([1] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 3: VR1-0to7 < VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.vector_register_file[2][:8] = list(range(1, 9))  # (1)~(8)
        vcore.run()
        self.assertEqual([0] * 8 + [1] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_5_SLEVV(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SLEVV VR1 VR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        # case 1: VR1-0to7 > VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(1, 9))  # (1)~(8)
        vcore.vector_register_file[2][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.run()
        self.assertEqual([0] * 8 + [1] * 56,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 2: VR1-0to7 = VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.vector_register_file[2][:8] = list(range(-4, 4))  # (-4)~(3)
        vcore.run()
        self.assertEqual([1] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

        # case 3: VR1-0to7 < VR2-0to7
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:8] = list(range(-1, -9, -1))  # (-1)~(-8)
        vcore.vector_register_file[2][:8] = list(range(1, 9))  # (1)~(8)
        vcore.run()
        self.assertEqual([1] * 64,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_6_SEQVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SEQVS VR1 SR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(1, 65))  # (1)~(64)
        vcore.scalar_register_file[2] = 32
        vcore.run()
        self.assertEqual([0] * 31 + [1] + [0] * 32,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_6_SNEVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SNEVS VR1 SR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(1, 65))  # (1)~(64)
        vcore.scalar_register_file[2] = 32
        vcore.run()
        self.assertEqual([1] * 31 + [0] + [1] * 32,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_6_SGTVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SGTVS VR1 SR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(1, 65))  # (1)~(64)
        vcore.scalar_register_file[2] = 32
        vcore.run()
        self.assertEqual([0] * 31 + [0] + [1] * 32,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_6_SLTVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SLTVS VR1 SR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(1, 65))  # (1)~(64)
        vcore.scalar_register_file[2] = 32
        vcore.run()
        self.assertEqual([1] * 31 + [0] + [0] * 32,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_6_SGEVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SGEVS VR1 SR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(1, 65))  # (1)~(64)
        vcore.scalar_register_file[2] = 32
        vcore.run()
        self.assertEqual([0] * 31 + [1] + [1] * 32,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_6_SLEVS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SLEVS VR1 SR2"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(1, 65))  # (1)~(64)
        vcore.scalar_register_file[2] = 32
        vcore.run()
        self.assertEqual([1] * 31 + [1] + [0] * 32,
                         vcore.vector_register_file.vector_mask_register)
        gather_stats(vcore)

    def test_7_CVM(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="CVM",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        vec_size = vcore.vector_register_file.vec_size
        # Add zeros into the VMR and confirm the contents
        for i in range(vec_size):
            vcore.vector_register_file.vector_mask_register[i] = i % 2
        self.assertTrue(0 in vcore.vector_register_file.vector_mask_register)
        self.assertFalse(
            all(
                map(lambda x: x == 1,
                    vcore.vector_register_file.vector_mask_register)))
        # Run CVM and confirm the results
        vcore.run()
        self.assertTrue(
            all(
                map(lambda x: x == 1,
                    vcore.vector_register_file.vector_mask_register)))
        gather_stats(vcore)

    def test_8_POP(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="POP SR1\nPOP SR2",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vec_size = vcore.vector_register_file.vec_size

        # Case 1: half of the mask is 1
        # Add zeros into half of the VMR and confirm the contents
        for i in range(vec_size):
            vcore.vector_register_file.vector_mask_register[i] = i % 2
        self.assertEqual(
            vec_size // 2,
            vcore.vector_register_file.vector_mask_register.count(1))
        # Run POP and confirm the results
        vcore.step_instr()
        self.assertEqual(vec_size // 2, vcore.SR1)

        # Case 2: two thirds of the mask is 1
        # Add zeros into two thirds of the VMR and confirm the contents
        for i in range(vec_size):
            vcore.vector_register_file.vector_mask_register[i] = int(bool(i %
                                                                          3))
        self.assertEqual(
            2 * vec_size // 3,
            vcore.vector_register_file.vector_mask_register.count(1))
        # Run POP and confirm the results
        vcore.step_instr()
        self.assertEqual(2 * vec_size // 3, vcore.SR2)
        gather_stats(vcore)

    def test_9_MTCL(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="MTCL SR1",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        self.assertEqual(vcore.vector_register_file.vec_size,
                         vcore.vector_register_file.vector_length_register)
        random_number = 37
        self.assertTrue(random_number <= vcore.vector_register_file.vec_size)
        vcore.scalar_register_file[1] = random_number

        vcore.run()
        self.assertEqual(random_number,
                         vcore.vector_register_file.vector_length_register)
        gather_stats(vcore)

    def test_10_MFCL(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="MFCL SR1",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        random_number = 53
        self.assertTrue(random_number <= vcore.vector_register_file.vec_size)
        vcore.vector_register_file.vector_length_register = random_number
        self.assertEqual(0, vcore.SR1)

        vcore.run()
        self.assertEqual(random_number, vcore.SR1)
        gather_stats(vcore)

    @unittest.skip("TODO")
    def test_11_LV(self):
        pass  # @todo Test LV

    @unittest.skip("TODO")
    def test_12_SV(self):
        pass  # @todo Test SV

    def test_13_LVWS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("LVWS VR1 SR2 SR3"),
            scalar_mem=[0],
            vector_mem=[0] + list(
                itertools.chain.from_iterable(
                    [(i, 0, 0) for i in range(0 + 1, 64 * 2 + 1, 2)]))[:-2],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[2] = 1  # base addr
        vcore.scalar_register_file[3] = 3  # stride
        vcore.run()
        self.assertEqual(list(range(0 + 1, 64 * 2 + 1, 2)), vcore.VR1)
        gather_stats(vcore)

    def test_14_SVWS(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SVWS VR1 SR2 SR3"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(0 + 1, 64 * 2 + 1, 2))
        vcore.scalar_register_file[2] = 2  # base addr
        vcore.scalar_register_file[3] = 3  # stride
        vcore.run()
        self.assertEqual([0, 0] + list(
            itertools.chain.from_iterable([
                (i, 0, 0) for i in range(0 + 1, 64 * 2 + 1, 2)
            ]))[:-2], vcore.vector_data_mem[:64 * 3])
        gather_stats(vcore)

    def test_15_LVI(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("LVI VR1 SR2 VR3"),
            scalar_mem=[0],
            vector_mem=list(
                itertools.chain.from_iterable([
                    ([i] + [0] * i) for i in range(0 + 1, 64 * 2 + 1, 2)
                ]))[:-127],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[2] = 0  # base addr
        vcore.vector_register_file[3][:] = list(
            itertools.accumulate([2 * i for i in range(64)]))
        vcore.run()
        self.assertEqual(list(range(0 + 1, 64 * 2 + 1, 2)), list(vcore.VR1))
        gather_stats(vcore)

    def test_16_SVI(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("SVI VR1 SR2 VR3"),
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.vector_register_file[1][:] = list(range(0 + 1, 64 * 2 + 1, 2))
        vcore.scalar_register_file[2] = 0  # base addr
        vcore.vector_register_file[3][:] = list(
            itertools.accumulate([2 * i for i in range(64)]))
        vcore.run()
        self.assertEqual(
            list(
                itertools.chain.from_iterable([
                    ([i] + [0] * i) for i in range(0 + 1, 64 * 2 + 1, 2)
                ]))[:-127], vcore.vector_data_mem[:list(
                    itertools.accumulate([2 * i for i in range(64)]))[-1] + 1])
        gather_stats(vcore)

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
        pass  # @todo Test SUB

    def test_20_AND(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="AND SR1 SR2 SR3\nAND SR4 SR5 SR6",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        max_int32 = 2147483647
        min_int32 = -2147483648

        # test negative
        vcore.scalar_register_file[2] = twos_complement(0xF0F0_F0F0)
        vcore.scalar_register_file[3] = twos_complement(0xFFFF_0000)
        print()
        print(f"{vcore.SR2 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR3 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR1 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR1, twos_complement(0xF0F0_0000))

        # test positive
        vcore.scalar_register_file[5] = 0x0F0F_0F0F
        vcore.scalar_register_file[6] = 0x0000_FFFF
        print()
        print(f"{vcore.SR5 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR6 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        self.assertEqual(vcore.SR4, 0x0000_0F0F)
        print(f"{vcore.SR4 & 0xFFFF_FFFF = :09_X}")

        gather_stats(vcore)

    def test_20_OR(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="OR SR1 SR2 SR3\nOR SR4 SR5 SR6",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        # test negative
        vcore.scalar_register_file[2] = twos_complement(0xF0F0_F0F0)
        vcore.scalar_register_file[3] = twos_complement(0xFF00_0F0F)
        print()
        print(f"{vcore.SR2 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR3 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR1 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR1, twos_complement(0xFFF0_FFFF))

        # test positive
        vcore.scalar_register_file[5] = 0x0F0F_0F0F
        vcore.scalar_register_file[6] = 0x00FF_F0F0
        print()
        print(f"{vcore.SR5 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR6 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR4 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR4, 0x0FFF_FFFF)

        gather_stats(vcore)

    def test_20_XOR(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="XOR SR1 SR2 SR3\nXOR SR4 SR5 SR6",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        # test negative
        vcore.scalar_register_file[2] = min_int32
        vcore.scalar_register_file[3] = twos_complement(0xFFFF_FFFF)
        print()
        print(f"{vcore.SR2 & 0xFFFF_FFFF = :032b}")
        print(f"{vcore.SR3 & 0xFFFF_FFFF = :032b}")
        vcore.step_instr()
        print(f"{vcore.SR1 & 0xFFFF_FFFF = :032b}")
        self.assertEqual(vcore.SR1, max_int32)

        # test positive
        vcore.scalar_register_file[5] = max_int32
        vcore.scalar_register_file[6] = twos_complement(0xFFFF_FFFF)
        print()
        print(f"{vcore.SR5 & 0xFFFF_FFFF = :032b}")
        print(f"{vcore.SR6 & 0xFFFF_FFFF = :032b}")
        vcore.step_instr()
        print(f"{vcore.SR4 & 0xFFFF_FFFF = :032b}")
        self.assertEqual(vcore.SR4, min_int32)

        gather_stats(vcore)

    def test_21_SLL(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="SLL SR1 SR2 SR3\nSLL SR4 SR5 SR6",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        # test negative
        vcore.scalar_register_file[2] = twos_complement(0xFF00_00FF)
        vcore.scalar_register_file[3] = 4
        print()
        print(f"{vcore.SR2 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR3 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR1 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR1, twos_complement(0xF000_0FF0))

        # test positive
        vcore.scalar_register_file[5] = 0x7F00_00FF
        vcore.scalar_register_file[6] = 4
        print()
        print(f"{vcore.SR6 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR5 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR4 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR4, twos_complement(0xF000_0FF0))

        gather_stats(vcore)

    def test_21_SRL(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="SRL SR1 SR2 SR3\nSRL SR4 SR5 SR6",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        # test negative
        vcore.scalar_register_file[2] = twos_complement(0xFF00_00FF)
        vcore.scalar_register_file[3] = 4
        print()
        print(f"{vcore.SR2 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR3 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR1 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR1, 0x0FF0_000F)

        # test positive
        vcore.scalar_register_file[5] = 0x7F00_00FF
        vcore.scalar_register_file[6] = 4
        print()
        print(f"{vcore.SR6 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR5 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR4 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR4, 0x07F0_000F)

        gather_stats(vcore)

    def test_22_SRA(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code="SRA SR1 SR2 SR3\nSRA SR4 SR5 SR6",
            scalar_mem=[0],
            vector_mem=[0],
        )
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")

        # test negative
        vcore.scalar_register_file[2] = twos_complement(0xFF00_00FF)
        vcore.scalar_register_file[3] = 4
        print()
        print(f"{vcore.SR2 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR3 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR1 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR1, twos_complement(0xFFF0_000F))

        # test positive
        vcore.scalar_register_file[5] = 0x7F00_00FF
        vcore.scalar_register_file[6] = 4
        print()
        print(f"{vcore.SR6 & 0xFFFF_FFFF = :09_X}")
        print(f"{vcore.SR5 & 0xFFFF_FFFF = :09_X}")
        vcore.step_instr()
        print(f"{vcore.SR4 & 0xFFFF_FFFF = :09_X}")
        self.assertEqual(vcore.SR4, 0x07F0_000F)

        gather_stats(vcore)

    def test_23_BEQ(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BEQ SR1 SR2 2"
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
        vcore.scalar_register_file[1] = 3
        vcore.scalar_register_file[2] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 2
        vcore.scalar_register_file[2] = 2
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 1
        vcore.scalar_register_file[2] = 3
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)

    def test_23_BNE(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BNE SR1 SR2 2"
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
        vcore.scalar_register_file[1] = 3
        vcore.scalar_register_file[2] = 1
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 2
        vcore.scalar_register_file[2] = 2
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 1
        vcore.scalar_register_file[2] = 3
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)

    def test_23_BGT(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BGT SR1 SR2 2"
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
        vcore.scalar_register_file[1] = 3
        vcore.scalar_register_file[2] = 1
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 2
        vcore.scalar_register_file[2] = 2
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 1
        vcore.scalar_register_file[2] = 3
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)

    def test_23_BLT(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BLT SR1 SR2 2"
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
        vcore.scalar_register_file[1] = 3
        vcore.scalar_register_file[2] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 2
        vcore.scalar_register_file[2] = 2
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 1
        vcore.scalar_register_file[2] = 3
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)

    def test_23_BGE(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BGE SR1 SR2 2"
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
        vcore.scalar_register_file[1] = 3
        vcore.scalar_register_file[2] = 1
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 2
        vcore.scalar_register_file[2] = 2
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 1
        vcore.scalar_register_file[2] = 3
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)

    def test_23_BLE(self):
        instruction = self.current_instruction()
        code, scalar_mem, vector_mem = self.generate(
            self.temp_dir,
            instruction,
            code=("BLE SR1 SR2 2"
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
        vcore.scalar_register_file[1] = 3
        vcore.scalar_register_file[2] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown instruction"):
            vcore.run()
        gather_stats(vcore)
        # case 2: SR1 = SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 2
        vcore.scalar_register_file[2] = 2
        vcore.run()  # asserts that no exception is raised
        gather_stats(vcore)
        # case 3: SR1 < SR2
        vcore = get_core(self.temp_dir, self.temp_dir,
                         f"single_instr_test_{instruction}")
        vcore.scalar_register_file[1] = 1
        vcore.scalar_register_file[2] = 3
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

        self.assertEqual(5, vcore.SR6)
        self.assertEqual(-5, vcore.SR7)

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
