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
