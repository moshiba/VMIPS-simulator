import abc
import argparse
import copy
import functools
import io
import itertools
import operator
import os
import pathlib
import re
import typing

DEBUG = os.environ.get("DEBUG", False)  # debug flag


def dprint(*args, debug_level: int = 1, **kwargs):
    """Level filtered debug logger
    """
    log_level = int(DEBUG)
    if log_level >= debug_level:
        print(*args, **kwargs)


def color(color_str: str, background=False):
    """Provide colorful output with ANSI escape codes
    """
    csi = "\033["  # control sequence introducer
    reset = csi + "0m"
    color_map = ("black", "red", "green", "yellow", "blue", "magenta", "cyan",
                 "white")
    bright_tone = color_str.startswith("bright_")
    color_str = color_str.removeprefix("bright_")
    color_code = color_map.index(
        color_str) + 30 + background * 10 + bright_tone * 60
    color_seq = csi + str(color_code) + "m"

    def formatter(string: str):
        return color_seq + string + reset

    return formatter


bgcolor = functools.partial(color, background=True)

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
        # get type assuming standard naming scheme: S/V+RF for scalar/vector
        self.type = str(dump_path)[-7] if len(str(dump_path)) >= 7 else ""
        self.type = self.type if self.type in "SV" else ""

    def __getitem__(self, index) -> int | list[int]:
        """Syntax sugar to directly access the underlying registers without
        meddling with internal variables.
        """
        dprint(color("blue")(f"{self.type}reg read "),
               f"R{index+1}",
               end="",
               debug_level=2)
        assert 0 <= index, "index too small"

        if self.vec_size == 1:
            # scalar register
            assert index < self.n_reg, "index too large"
            dprint(f" = {self._data[index][0]}", debug_level=2)
            return self._data[index][0]
        else:
            # vector register
            assert index < self.vec_size, "index too large"
            dprint(f" = {self._data[index]}", debug_level=2)
            return self._data[index]

    def __setitem__(self, index, value):
        """Syntax sugar to directly set the underlying registers without
        meddling with internal variables.
        """
        dprint(color("red")(f"{self.type}reg write"),
               f"R{index+1} = {value}",
               debug_level=2)
        assert 0 <= index, "index too small"

        if self.vec_size == 1:
            # scalar register
            assert index < self.n_reg, "index too large"
            self._data[index][0] = value
        else:
            # vector register
            assert index < self.vec_size, "index too large"
            self._data[index] = value

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
        # get type assuming standard naming scheme: S/V+DMEM for scalar/vector
        self.type = str(load_path)[-9] if len(str(load_path)) >= 9 else ""
        self.type = self.type if self.type in "SV" else "?"

    def __getitem__(self, key) -> int:
        """Syntax sugar to directly access the underlying cells without meddling
        with internal variables.
        """
        # TODO: consider raising IndexError properly
        # TODO: add tests for illegal address access
        if isinstance(key, slice):  # Access a slice
            lower_index = key.start or 0
            upper_index = key.stop - 1
        elif isinstance(key, int):  # Access a single element
            lower_index = upper_index = key
        else:
            raise TypeError(f"bad key type: {type(key)}")

        assert 0 <= lower_index, "address too small"
        assert upper_index < self.size_limit, "address too large"

        dprint(bgcolor("blue")(f"{self.type}mem read "),
               f"{lower_index:010_d} = {self._data[key]}",
               debug_level=2)
        return self._data[key]

    def __setitem__(self, key, value) -> int:
        """Syntax sugar to directly set the underlying cells without meddling
        with internal variables.
        """
        if isinstance(key, slice):  # Access a slice
            lower_index = key.start or 0
            upper_index = key.stop - 1
        elif isinstance(key, int):  # Access a single element
            lower_index = upper_index = key
        else:
            raise TypeError(f"bad key type: {type(key)}")

        assert 0 <= lower_index, "address too small"
        assert upper_index < self.size_limit, "address too large"

        dprint(bgcolor("blue")(f"{self.type}mem write"),
               f"0x{lower_index:010_d} = {self._data[key]}",
               debug_level=2)
        self._data[key] = value

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
        dprint(bgcolor("bright_black")(f"{index+1}" + color("bright_green")
                                       (self.instructions[index])),
               debug_level=1)
        dprint(bgcolor("white")(color("green")("instr read")) +
               bgcolor("white")(color("black")(f" {index+1} ")),
               list(i for i in Core.decode(self.instructions[index]).groups()
                    if i is not None),
               debug_level=2)

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


class ALU:
    # function_table
    _match_func_type_regex = re.compile(
        r"^"
        r"(?:(?P<vector_op>ADD|SUB|MUL|DIV)V(?P<vec_op_type>[VS]))"
        r"|(?P<vec_mask_reg>(?:S(?P<mask_condition>EQ|NE|GT|LT|GE|LE)V(?P<mask_type>[VS]))|(?P<clear_mask>CVM)|(?P<count_mask>POP))"
        r"|(?P<vec_len_reg>M[TF]CL)"
        r"|(?P<mem_op>[LS])(?P<mem_type>VWS|VI|V|S)"
        r"|(?P<scalar_op>ADD|SUB|AND|OR|XOR)"
        r"|(?P<control>B(?P<branch_condition>EQ|NE|GT|LT|GE|LE))"
        r"|(?P<stop>HALT)"
        r"$", re.ASCII)

    def __init__(self, core):
        self.core = core

    def do(self, parsed_instruction: re.Match):
        instruction = parsed_instruction["instruction"]
        if instruction is None:
            return  # Skip no-op statements such as comments

        functionality = self._match_func_type_regex.match(instruction)
        # As all these instruction-name-format groups are disjoint, only one
        # type would match. So if there are some match then they must all belong
        # to the same instruction.
        # And since instruction operation type is the first capture group for
        # all instruction formats, finding the first non-empty match would give
        # the functionality group of the instruction
        func_group = tuple({
            k: v for k, v in functionality.groupdict().items() if v is not None
        })[0]
        getattr(self, func_group)(functionality, parsed_instruction)

    @classmethod
    def reg_index(cls, register_token):
        """Helper method to extract register index from assembly
        for example: SR2->1
        """
        return int(register_token[2:]) - 1

    def vector_op(self, functionality, instruction):
        # TODO: implement vector length/mask later
        srf = self.core.scalar_register_file
        vrf = self.core.vector_register_file

        operation_code = functionality["vector_op"].lower()
        # scalar, vector, strided, scatter/gather
        operand_type = functionality["vec_op_type"]

        if operand_type == "V":
            operand2 = vrf[self.reg_index(instruction["operand2"])]
            operand3 = vrf[self.reg_index(instruction["operand3"])]

            vrf[self.reg_index(instruction["operand1"])] = list(
                map(getattr(operator, operation_code), operand2, operand3))
        elif operand_type == "S":
            operand2 = vrf[self.reg_index(instruction["operand2"])]
            operand3 = srf[self.reg_index(instruction["operand3"])]
            raise NotImplementedError
        else:
            raise RuntimeError("Unknown vector arithmetic instruction:",
                               instruction.groupdict())

    def vec_mask_reg(self, functionality, instruction):
        raise NotImplementedError

    def vec_len_reg(self, functionality, instruction):
        raise NotImplementedError

    def mem_op(self, functionality, instruction):
        srf = self.core.scalar_register_file
        vrf = self.core.vector_register_file
        smem = self.core.scalar_data_mem
        vmem = self.core.vector_data_mem

        # load or save
        action = functionality["mem_op"]
        # scalar, vector, strided, scatter/gather
        mem_type = functionality["mem_type"]

        if mem_type == "S":  # scalar
            immediate = int(instruction["operand3"])

            if action == "L":  # load
                address_value = srf[self.reg_index(instruction["operand2"])]
                # reg = mem[reg + imm]
                srf[self.reg_index(
                    instruction["operand1"])] = smem[address_value + immediate]
            else:  # store
                address_value = srf[self.reg_index(instruction["operand2"])]
                value = srf[self.reg_index(instruction["operand1"])]
                # mem[reg + imm] = reg
                smem[address_value + immediate] = value
        elif mem_type == "V":  # vector
            # TODO: implement vector length/mask later
            if action == "L":  # load
                address_value = srf[self.reg_index(instruction["operand2"])]
                mem_value = vmem[address_value:address_value + vrf.vec_size]
                # reg = mem[reg]
                vrf[self.reg_index(instruction["operand1"])] = mem_value
            else:  # store
                address_value = srf[self.reg_index(instruction["operand2"])]
                value = vrf[self.reg_index(instruction["operand1"])]
                # mem[reg] = reg
                vmem[address_value:address_value + vrf.vec_size] = value
        elif mem_type == "VWS":  # strided
            raise NotImplementedError
        elif mem_type == "VI":  # scatter/gather
            raise NotImplementedError
        else:
            raise RuntimeError("Unknown memory operation instruction:",
                               instruction.groupdict())

    def scalar_op(self, functionality, instruction):
        operation_code = functionality["scalar_op"].lower()

        srf = self.core.scalar_register_file
        operand2 = srf[self.reg_index(instruction["operand2"])]
        operand3 = srf[self.reg_index(instruction["operand3"])]

        operation = operator.methodcaller(operation_code, operand2, operand3)
        srf[self.reg_index(instruction["operand1"])] = operation(operator)

    def control(self, functionality, instruction):
        condition = functionality["branch_condition"].lower()

        srf = self.core.scalar_register_file
        operand1 = srf[self.reg_index(instruction["operand1"])]
        operand2 = srf[self.reg_index(instruction["operand2"])]
        immediate = int(instruction["operand3"])

        operation = operator.methodcaller(condition, operand1, operand2)
        compare_result = operation(operator)

        if compare_result is True:
            # Branch taken
            self.core.PC += immediate
        else:
            # Branch not taken
            pass

    def stop(self, functionality, instruction):
        self.core.freeze = True


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
        self.alu = ALU(self)
        self.freeze = False

    _attr_shorthand_regex = re.compile(
        r"^(?P<value_type>[SV])(?P<mem_type>[RM])(?P<index>\d+)$", re.ASCII)
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

    def __getattr__(self, name: str):
        """Shorthands for accessing register files and memories
        """
        if match_result := self._attr_shorthand_regex.match(name):
            value_type, mem_type, reg_idx = match_result.groups()
            cell = self.__getattribute__(
                f"{dict(S='scalar', V='vector')[value_type]}"
                "_"
                f"{dict(R='register_file',M='data_mem')[mem_type]}")

            # Designate registers by index starting from 1
            # and memory cells starting from 0
            return cell[int(reg_idx) - int(mem_type == "R")]
        else:
            return self.__getattribute__(name)

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

    def step(self):
        """Execute one line of code
        """
        current_line = self.instruction_mem[self.PC]
        self.alu.do(self.decode(current_line))
        self.PC += 1

    def step_instr(self):
        """Executes one instruction
        Step over empty or comment lines
        """
        # Skip non-statement lines
        next_line = self.instruction_mem.instructions[self.PC]
        next_instruction = self.decode(next_line)["instruction"]
        while next_instruction is None:
            self.step()
            next_line = self.instruction_mem.instructions[self.PC]
            next_instruction = self.decode(next_line)["instruction"]

        self.step()  # Execute the next line, which will not by empty or comment

    def run(self):
        while self.freeze is not True:
            self.step()

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
    # Parse arguments for input file location
    parser = argparse.ArgumentParser(
        description='Vector Core Performance Model')
    parser.add_argument(
        '--iodir',
        default="",
        type=str,
        help=
        'Path to the folder containing the input files - instructions and data.'
    )
    parsed_args = parser.parse_args()

    io_dir = pathlib.Path(parsed_args.iodir).absolute()
    print("IO Directory:", io_dir)

    vcore = Core(
        scalar_register_file=RegisterFile(dump_path=io_dir / "SRF.txt",
                                          n_reg=8,
                                          word_size=32),
        vector_register_file=RegisterFile(dump_path=io_dir / "VRF.txt",
                                          n_reg=8,
                                          vec_size=64,
                                          word_size=32),
        instruction_mem=InstructionMemory(load_path=io_dir / "Code.asm"),
        scalar_data_mem=DataMemory(load_path=io_dir / "SDMEM.txt",
                                   dump_path=io_dir / "SDMEMOP.txt",
                                   address_length=13
                                   # 32 KB is 2^15 bytes = 2^13 K 32-bit words
                                  ),
        vector_data_mem=DataMemory(load_path=io_dir / "VDMEM.txt",
                                   dump_path=io_dir / "VDMEMOP.txt",
                                   address_length=17
                                   # 512 KB is 2^19 bytes = 2^17 K 32-bit words
                                  ))

    vcore.run()

    # Dump all internal state
    # RegFile
    vcore.scalar_register_file.dump()
    vcore.vector_register_file.dump()
    # Memory
    vcore.scalar_data_mem.dump()
    vcore.vector_data_mem.dump()
