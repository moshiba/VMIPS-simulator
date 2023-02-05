import abc
import copy
import io
import pathlib
import re
import typing


class FileMap(abc.ABC):
    """Base class for internal state persistence.

    Provides load() and dump() like other Python built-in modules to predefined
    or at-load/dump-time designated location.
    """

    def __init__(self, /, load_path: str, dump_path: str = None):
        self.load_path = pathlib.Path(load_path)
        self.dump_path = pathlib.Path(dump_path or load_path)

    def load(self, path: pathlib.Path = None):
        """State importer"""
        path = pathlib.Path(path or self.load_path)
        with path.open(mode="r", encoding="ascii") as file:
            self.internal_state = file

    def dump(self, path: pathlib.Path = None) -> int:
        """State exporter"""
        path = pathlib.Path(path or self.dump_path)
        with path.open(mode="w", encoding="ascii") as file:
            return file.write(self.internal_state)

    @property
    @abc.abstractmethod
    def internal_state(self) -> str:
        """Stringifier and formatter for the exporter
        """
        raise NotImplementedError

    @internal_state.setter
    @abc.abstractmethod
    def internal_state(self, file) -> None:
        """Stringifier and formatter for the importer
        """
        raise NotImplementedError


class RegisterFile(FileMap):
    """Configurable register file
    """

    def __init__(
            self,
            /,
            dump_path: str,
            n_reg: int,  # number of registers
            vec_size: int = 1,  # number of words in a vector register
            word_size: int = 32,  # number of bits in a word
    ):
        super().__init__(load_path="", dump_path=dump_path)
        self.n_reg = n_reg
        self.vec_size = vec_size
        self.word_size = word_size
        # TODO: do get/set min/max value check
        self._data = [
            [0x0 for scalar in range(vec_size)] for reg in range(n_reg)
        ]

    def __getitem__(self, index) -> int | list[int]:
        """Syntax sugar to directly access the underlying registers without
        meddling with internal variables.
        """
        if self.vec_size == 1:
            # scalar register
            return self._data[index][0]
        else:
            # vector register
            return self._data[index]

    @typing.final
    def load(self):
        """Drop support for loading initial state.
        """
        raise NotImplementedError

    @property
    def internal_state(self) -> str:
        lines = []  # stdout buffer

        row_format = "{:<13}" * self.vec_size
        # Print index columns
        lines.append(
            row_format.format(
                *[str(word_idx) for word_idx in range(self.vec_size)]),)

        # Print separator line
        lines.append("-" * (self.vec_size * 13))

        # Print register values
        lines += [
            row_format.format(*[str(val)
                                for val in data])
            for data in self._data
        ]
        return "\n".join(lines) + "\n"

    @typing.final
    @internal_state.setter
    def internal_state(self, file):
        """Drop support for loading initial state.
        """
        raise NotImplementedError


class DataMemory(FileMap):
    """Configurable data memory
    """

    def __init__(
            self,
            /,
            load_path: str,
            dump_path: str,
            address_length: int,  # in bits
    ):
        super().__init__(load_path, dump_path)
        self.size_limit = pow(2, address_length)
        # TODO: do get/set min/max value check
        self._data = [0x0 for word in range(self.size_limit)]

    def __getitem__(self, index) -> int:
        """Syntax sugar to directly access the underlying cells without meddling
        with internal variables.
        """
        return self._data[index]

    @property
    def internal_state(self) -> str:
        lines = [str(word) for word in self._data]  # stdout buffer
        return "\n".join(lines) + "\n"

    @internal_state.setter
    def internal_state(self, file: io.TextIOWrapper):
        mem_words = [int(line.strip()) for line in file.readlines()]
        n_words = len(mem_words)
        assert n_words < self.size_limit, "too much data"
        self._data = mem_words
        # Pad the rest as zero
        self._data.extend(0x0 for word in range(self.size_limit - n_words))


class InstructionMemory(FileMap):
    """Configurable instruction memory
    """

    def __init__(self, /, load_path: str):
        super().__init__(load_path=load_path, dump_path="")
        self.size_limit = pow(2, 16)
        self.instructions = tuple()

    def __getitem__(self, index) -> str:
        """Syntax sugar to directly access the underlying lines without meddling
        with internal variables.
        """
        if index > self.size_limit:
            raise IndexError(f"Invalid memory access at index {index}"
                             f" with memory size {self.size_limit}")
        return self.instructions[index]

    @typing.final
    def dump(self):
        """Drop support for dumping internal state.
        There's no need for persistence and the current state is accessable
        through the attribute 'instructions'.
        """
        raise NotImplementedError

    @property
    def internal_state(self) -> str:
        instructions = copy.copy(self.instructions)
        # Strip trailing empty instructions
        while not instructions[-1]:
            instructions.pop()
        return tuple(instructions)

    @internal_state.setter
    def internal_state(self, file: io.TextIOWrapper):
        instructions = [instr.strip() for instr in file.readlines()]
        assert len(instructions) < self.size_limit, "too many instructions"
        self.instructions = instructions


class Core:
    """Configurable VMIPS core
    """

    def __init__(
        self,
        scalar_register_file: RegisterFile,
        vector_register_file: RegisterFile,
        instruction_mem: InstructionMemory,
        scalar_data_mem: DataMemory,
        vector_data_mem: DataMemory,
    ):

        self.scalar_register_file = scalar_register_file
        self.vector_register_file = vector_register_file
        self.instruction_mem = instruction_mem
        self.scalar_data_mem = scalar_data_mem
        self.vector_data_mem = vector_data_mem

        self.instruction_mem.load()
        self.scalar_data_mem.load()
        self.vector_data_mem.load()
        self.program_counter = 0  # mapped as writable property: PC

    _instruction_decoder_regex = re.compile(
        # Type: Valid statement
        r"(?:^(?P<instruction>\w+)"  #       instruction
        r"(?:[ ]+(?P<operand1>\w+))?"  #     operand-1 (optional)
        r"(?:[ ]+(?P<operand2>\w+))?"  #     operand-2 (optional)
        r"(?:[ ]+(?P<operand3>[-\w]+))?"  #  operand-3 (optional)
        r"[ ]*(?P<inline_comment>#.*)?$)"  # in-line comment (optional)
        # Type: Comment line
        r"|(?:^(?P<comment_line>[ ]*?#.*)$)"
        # Type: Empty line
        r"|(?P<empty_line>^(?<!.)$|(?:^[ ]+$)$)",
        re.ASCII)

    @classmethod
    def decode(cls, statement):
        tokens = cls._instruction_decoder_regex.match(statement)
        return tokens

    @property
    def PC(self):
        """program counter value getter"""
        return self.program_counter

    @PC.setter
    def PC(self, value):
        """program counter value setter"""
        self.program_counter = int(value)

    # Common parameters for components
    class AttrDict(dict):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    common_params = AttrDict(
        scalar_regfile={
            "n_reg": 8,
            "vec_size": 1,
            "word_size": 32,
        },
        vector_regfile={
            "n_reg": 8,
            "vec_size": 64,
            "word_size": 32,
        },
        scalar_datamem={
            "address_length": 13,  # 32 KB is 2^15 bytes = 2^13 K 32-bit words
        },
        vector_datamem={
            "address_length": 17,  # 512 KB is 2^19 bytes = 2^17 K 32-bit words
        },
    )


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2
    assert (assembly_program := pathlib.Path(sys.argv[1])).is_file()
    data_dir = pathlib.PurePath("isa_test_data")
    vcore = Core(
        scalar_register_file=RegisterFile(dump_path=data_dir / "SRF.txt",
                                          n_reg=8,
                                          word_size=32),
        vector_register_file=RegisterFile(dump_path=data_dir / "VRF.txt",
                                          n_reg=8,
                                          vec_size=64,
                                          word_size=32),
        instruction_mem=InstructionMemory(load_path=data_dir / "Code.asm"),
        scalar_data_mem=DataMemory(load_path=data_dir / "SDMEM.txt",
                                   dump_path=data_dir / "SDMEMOP.txt",
                                   address_length=13
                                   # 32 KB is 2^15 bytes = 2^13 K 32-bit words
                                  ),
        vector_data_mem=DataMemory(load_path=data_dir / "VDMEM.txt",
                                   dump_path=data_dir / "VDMEMOP.txt",
                                   address_length=17
                                   # 512 KB is 2^19 bytes = 2^17 K 32-bit words
                                  ))

    for line in vcore.instruction_mem:
        instr, op1, op2, op3 = vcore.decode(line).group(
            "instruction",
            "operand1",
            "operand2",
            "operand3",
        )
        print(instr, op1, op2, op3)

    if dump_all := False:
        # RegFile
        vcore.scalar_register_file.dump()
        vcore.vector_register_file.dump()
        # Memory
        vcore.scalar_data_mem.dump()
        vcore.vector_data_mem.dump()
