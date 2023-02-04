import pathlib
import abc


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
