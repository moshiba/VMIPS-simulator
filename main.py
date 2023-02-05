import pathlib
import abc
import io
import copy


class FileMap(abc.ABC):
    """Base class for internal state persistence
    """

    def __init__(self, /, load_path: str, dump_path: str = None):
        self.load_path = pathlib.Path(load_path)
        self.dump_path = pathlib.Path(dump_path or load_path)

    def load(self, path: pathlib.Path = None):
        path = pathlib.Path(path or self.load_path)
        with path.open(mode="r", encoding="ascii") as file:
            self.data = file

    def dump(self, path: pathlib.Path = None) -> int:
        path = pathlib.Path(path or self.dump_path)
        with path.open(mode="w", encoding="ascii") as file:
            return file.write(self.data)

    @property
    @abc.abstractmethod
    def data(self) -> str:
        raise NotImplementedError

    @data.setter
    @abc.abstractmethod
    def data(self, file) -> None:
        raise NotImplementedError


class RegisterFile(FileMap):

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
        if self.vec_size == 1:
            # scalar register
            return self._data[index][0]
        else:
            # vector register
            return self._data[index]

    def load(self):
        raise NotImplementedError

    @property
    def data(self) -> str:
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

    @data.setter
    def data(self, file):
        raise NotImplementedError


class DataMemory(FileMap):

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
        return self._data[index]

    @property
    def data(self) -> str:
        lines = [str(word) for word in self._data]  # stdout buffer
        return "\n".join(lines) + "\n"

    @data.setter
    def data(self, file: io.TextIOWrapper):
        mem_words = [int(line.strip()) for line in file.readlines()]
        n_words = len(mem_words)
        assert n_words < self.size_limit, "too much data"
        self._data = mem_words
        # Pad the rest as zero
        self._data.extend(0x0 for word in range(self.size_limit - n_words))


class InstructionMemory(FileMap):

    def __init__(self, /, load_path: str):
        super().__init__(load_path=load_path, dump_path="")
        self.size_limit = pow(2, 16)
        self.instructions = tuple()

    def __getitem__(self, index) -> str:
        if index > self.size_limit:
            raise IndexError(f"Invalid memory access at index {index}"
                             f" with memory size {self.size_limit}")
        return self.instructions[index]

    def dump(self):
        raise NotImplementedError

    @property
    def data(self) -> str:
        instructions = copy.copy(self.instructions)
        # Strip trailing empty instructions
        while not instructions[-1]:
            instructions.pop()
        return tuple(instructions)

    @data.setter
    def data(self, file: io.TextIOWrapper):
        instructions = [instr.strip() for instr in file.readlines()]
        assert len(instructions) < self.size_limit, "too many instructions"
        self.instructions = instructions
