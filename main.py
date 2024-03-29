"""VMIPS functional simulator
"""
import abc
import argparse
import array
import collections.abc
import copy
import functools
import io
import itertools
import operator
import os
import pathlib
import pprint
import queue
import re
import typing

DEBUG = os.environ.get("DEBUG", False)  # verbose level flag


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


@functools.total_ordering
class StaticLengthArray(collections.abc.Sequence):
    """List, but with a static size
    """

    def __init__(self, iterable, *, container_type=list) -> None:
        self.__data = container_type(iterable)
        self.__size = len(self.__data)

    @property
    def size(self):
        return self.__size

    def __getitem__(self, index):
        assert self.size == len(self.__data), f"{self.size}!={len(self.__data)}"
        return self.__data[index]

    def __len__(self):
        assert self.size == len(self.__data), f"{self.size}!={len(self.__data)}"
        return len(self.__data)

    def __setitem__(self, index, value):
        # Provide this but not all other 'MutableSequence' methods
        assert self.size == len(self.__data), f"{self.size}!={len(self.__data)}"
        self.__data[index] = value

    def __eq__(self, other):
        return self.__data == other

    def __lt__(self, other):
        return self.__data < other

    def __str__(self):
        return str(self.__data)


@functools.total_ordering
class SignedInt32Array(StaticLengthArray):
    """Static sized array with SignedInt32 value format
    read/write value range check included
    """

    def __init__(self, iterable) -> None:
        # Choose the right type that gives a 4-byte signed integer
        if array.array("i").itemsize == 4:
            self.type_code = "i"
        elif array.array("l").itemsize == 4:
            self.type_code = "l"
        else:
            print(
                "Warning: "
                "Failed to find a native type that represents 32-bit signed int"
                ", falling back to normal lists instead")
            super().__init__(iterable)
            return
        s32_array = array.array(self.type_code, iterable)
        super().__init__(s32_array, container_type=lambda x: x)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            value = array.array(self.type_code, value)  # type cast
        super().__setitem__(index, value)
        assert self.size == len(self), f"{self.size}!={len(self)}"

    def __eq__(self, other):
        return self._StaticLengthArray__data.tolist() == other

    def __lt__(self, other):
        return self._StaticLengthArray__data.tolist() < other


class FileMap(abc.ABC):
    """Base class for internal state persistence.

    Provides load() and dump() like other Python built-in modules to predefined
    or at-load/dump-time designated location.
    """

    def __init__(self, *, load_path: str, dump_path: str = None):
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
            *,
            dump_path: str,
            n_reg: int,  # number of registers
            vec_size: int = 1,  # number of words in a vector register
            word_size: int = 32,  # number of bits in a word
    ):
        super().__init__(load_path="", dump_path=dump_path)
        self.n_reg = n_reg
        self.vec_size = vec_size
        self.word_size = word_size
        self.__data = StaticLengthArray(
            SignedInt32Array(0x0
                             for scalar in range(vec_size))
            for reg in range(n_reg))
        self.vector_mask_register = StaticLengthArray([1] * vec_size)
        self.vector_length_register = vec_size
        # get type assuming standard naming scheme: S/V+RF for scalar/vector
        self.type = str(dump_path)[-7] if len(str(dump_path)) >= 7 else ""
        self.type = self.type if self.type in "SV" else ""

    def __getitem__(self, index) -> typing.Union[int, list[int]]:
        """Syntax sugar to directly access the underlying registers without
        meddling with internal variables.
        """
        dprint(color("blue")(f"{self.type}reg read "),
               f"R{index}",
               end="",
               debug_level=2)
        assert 0 <= index, "index too small"

        if self.vec_size == 1:
            # scalar register
            assert index < self.n_reg, "index too large"
            dprint(f" = {self.__data[index][0]}", debug_level=2)
            return self.__data[index][0]
        else:
            # vector register
            assert index < self.vec_size, "index too large"
            dprint(f" = {self.__data[index]}", debug_level=2)
            return self.__data[index]

    def __setitem__(self, index, value):
        """Syntax sugar to directly set the underlying registers without
        meddling with internal variables.
        """
        dprint(color("red")(f"{self.type}reg write"),
               f"R{index} = {value}",
               debug_level=2)
        assert 0 <= index, "index too small"

        if self.vec_size == 1:
            # scalar register
            assert index < self.n_reg, "index too large"
            self.__data[index][0] = value
        else:
            # vector register
            assert index < self.vec_size, "index too large"
            self.__data[index] = value

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
            for data in self.__data
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
            *,
            load_path: str,
            dump_path: str,
            address_length: int,  # in bits
    ):
        super().__init__(load_path=load_path, dump_path=dump_path)
        self.size_limit = pow(2, address_length)
        self.__data = SignedInt32Array(0x0 for word in range(self.size_limit))
        # get type assuming standard naming scheme: S/V+DMEM for scalar/vector
        self.type = str(load_path)[-9] if len(str(load_path)) >= 9 else ""
        self.type = self.type if self.type in "SV" else "?"

    def __getitem__(self, key) -> int:
        """Syntax sugar to directly access the underlying cells without meddling
        with internal variables.
        """
        # TODO: consider raising IndexError properly
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
               f"{lower_index:010_d} = {self.__data[key]}",
               debug_level=2)
        return self.__data[key]

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

        self.__data[key] = value
        dprint(bgcolor("red")(f"{self.type}mem write"),
               f"0x{lower_index:010_d} = {self.__data[key]}",
               debug_level=2)

    @property
    def internal_state(self) -> str:
        lines = [str(word) for word in self.__data]  # stdout buffer
        return "\n".join(lines) + "\n"

    @internal_state.setter
    def internal_state(self, file: io.TextIOWrapper):
        mem_words = [int(line.strip()) for line in file.readlines()]
        n_words = len(mem_words)
        assert n_words < self.size_limit, "too much data"
        self.__data = mem_words
        # Pad the rest as zero
        self.__data.extend(0x0 for word in range(self.size_limit - n_words))


class InstructionType:

    def __init__(self, operands, result):
        # Registers that will be read by this instruction
        self.operands: set[str] = set(operands)

        # Registers that will be written by this instruction
        self.result: str = result


class ScalarInstruction(InstructionType):
    pass


class ControlInstruction(ScalarInstruction):
    pass


class VectorComputeInstruction(InstructionType):
    # TODO: depends on VLR and VMR but latches them upon dispatch, doesn't need them after this
    def __init__(self, operands, result, vec_len):
        super().__init__(operands, result)
        self.vec_len = vec_len


class VectorMemoryInstruction(InstructionType):

    def __init__(self, operands, result, target_addrs):
        super().__init__(operands, result)
        self.target_addrs: list[int] = list(target_addrs)


class VectorMultiplyInstruction(VectorComputeInstruction):
    pass


class VectorAddSubInstruction(VectorComputeInstruction):
    pass


class VectorDivideInstruction(VectorComputeInstruction):
    pass


class InstructionMemory(FileMap):
    """Configurable instruction memory
    """

    def __init__(self, *, load_path: str):
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
    """The actual operation handler
    Uses regular expression to parse instructions and extract operands
    """
    # function_table
    _match_func_type_regex = re.compile(
        r"^"
        r"(?:(?P<vector_op>ADD|SUB|MUL|DIV)V(?P<vec_op_type>[VS]))"
        r"|(?P<vec_mask_reg>(?:S(?P<mask_condition>EQ|NE|GT|LT|GE|LE)V(?P<mask_type>[VS]))|(?P<clear_mask>CVM)|(?P<count_mask>POP))"
        r"|(?P<vec_len_reg>M(?P<vlr_mode>[TF])CL)"
        r"|(?P<mem_op>[LS])(?P<mem_type>VWS|VI|V|S)"
        r"|(?P<scalar_op>ADD|SUB|AND|OR|XOR|SLL|SRL|SRA)"
        r"|(?P<control>B(?P<branch_condition>EQ|NE|GT|LT|GE|LE))"
        r"|(?P<stop>HALT)"
        r"$", re.ASCII)

    def __init__(self, core):
        self.core = core
        self.instruction_stream = []  # stores program trace

    def do(self, parsed_instruction: re.Match):
        instruction = parsed_instruction["instruction"]
        if instruction is None:
            return  # Skip no-op statements such as comments

        functionality = self._match_func_type_regex.match(instruction)
        if functionality is None:
            raise RuntimeError(
                f"Unknown instruction: {parsed_instruction.string}")
        # As all these instruction-name-format groups are disjoint, only one
        # type would match. So if there are some match then they must all belong
        # to the same instruction.
        # And since instruction operation type is the first capture group for
        # all instruction formats, finding the first non-empty match would give
        # the functionality group of the instruction
        func_group = tuple({
            k: v for k, v in functionality.groupdict().items() if v is not None
        })[0]
        valid_line = getattr(self, func_group)(functionality,
                                               parsed_instruction)
        assert valid_line is not None
        self.instruction_stream.append(valid_line)

    @classmethod
    def reg_index(cls, register_token):
        """Helper method to extract register index from assembly
        for example: SR2->1
        """
        return int(register_token[2:]) - 0

    def vector_op(self, functionality, instruction):
        # Aliases
        srf = self.core.scalar_register_file
        vrf = self.core.vector_register_file

        # Map operation name to standard operators
        op_code = functionality["vector_op"].lower()
        op_code = "floordiv" if op_code == "div" else op_code
        operand_type = functionality["vec_op_type"]
        operation = getattr(operator, op_code)

        # Get operands
        operand2 = vrf[self.reg_index(instruction["operand2"])]
        if operand_type == "V":
            operand3 = vrf[self.reg_index(instruction["operand3"])]
        elif operand_type == "S":
            operand3 = itertools.repeat(
                srf[self.reg_index(instruction["operand3"])], vrf.vec_size)
        else:
            raise RuntimeError("Unknown vector arithmetic instruction:",
                               instruction.groupdict())

        # Do operation and store the result
        result = list(map(operation, operand2, operand3))
        # Handle vector length and vector mask effects
        original_values = vrf[self.reg_index(instruction["operand1"])]
        masked_values = [
            val if mask else orig for mask, orig, val in tuple(
                zip(vrf.vector_mask_register, original_values, result))
            [:vrf.vector_length_register]
        ]
        vrf[self.reg_index(instruction["operand1"]
                          )][:vrf.vector_length_register] = masked_values

        if op_code == "mul":
            instr_cls = VectorMultiplyInstruction
        elif op_code == "floordiv":
            instr_cls = VectorDivideInstruction
        elif op_code in ["add", "sub"]:
            instr_cls = VectorAddSubInstruction
        else:
            raise RuntimeError(f"unknown op_code: {op_code}")
        return instr_cls(
            ["VLR", "VMR", instruction["operand2"], instruction["operand3"]],
            instruction["operand1"], vrf.vector_length_register)

    def vec_mask_reg(self, functionality, instruction):
        # Aliases
        srf = self.core.scalar_register_file
        vrf = self.core.vector_register_file

        # Act differently according to instruction parsing results
        if functionality["clear_mask"]:  # CVM
            # Use slice assignment instead of direct assignment
            # in order to retain custom 'StaticLengthArray' type
            vrf.vector_mask_register[:] = [1] * vrf.vec_size
            assert isinstance(vrf.vector_mask_register, StaticLengthArray)
            return ScalarInstruction([], "VMR")

        elif functionality["count_mask"]:  # POP
            srf[self.reg_index(
                instruction["operand1"])] = vrf.vector_mask_register.count(1)
            return ScalarInstruction(["VMR"], instruction["operand1"])

        elif (op_code := functionality["mask_condition"]) and (
                mask_type := functionality["mask_type"]):  # S__VV and S__VS
            # Map operation name to standard operators
            op_code = op_code.lower()
            operation = getattr(operator, op_code)

            # Get operands
            operand1 = vrf[self.reg_index(instruction["operand1"])]
            if mask_type == "V":
                operand2 = vrf[self.reg_index(instruction["operand2"])]
            elif mask_type == "S":
                operand2 = [srf[self.reg_index(instruction["operand2"])]
                           ] * vrf.vec_size

            # Do operation and store the result
            # Use slice assignment instead of direct assignment
            # in order to retain custom 'StaticLengthArray' type
            vrf.vector_mask_register[:] = [
                int(bool(value)) for value in map(operation, operand1, operand2)
            ]
            assert isinstance(vrf.vector_mask_register, StaticLengthArray)
            return ScalarInstruction(
                [instruction["operand1"], instruction["operand2"]], "VMR")

        else:
            raise RuntimeError("Unknown vector mask register instruction:",
                               instruction.groupdict())

    def vec_len_reg(self, functionality, instruction):
        # Aliases
        srf = self.core.scalar_register_file
        vrf = self.core.vector_register_file

        # Act differently according to instruction parsing results
        mode = functionality["vlr_mode"]
        if mode == "T":  # MTCL
            value = srf[self.reg_index(instruction["operand1"])]
            assert 0 <= value <= vrf.vec_size, "illegal vector length"
            vrf.vector_length_register = value
            return ScalarInstruction([instruction["operand1"]], "VLR")

        elif mode == "F":  # MFCL
            srf[self.reg_index(
                instruction["operand1"])] = vrf.vector_length_register
            return ScalarInstruction(["VLR"], instruction["operand1"])

        else:
            raise RuntimeError("Unknown vector length register instruction:",
                               instruction.groupdict())

    def mem_op(self, functionality, instruction):
        # Aliases
        srf = self.core.scalar_register_file
        vrf = self.core.vector_register_file
        smem = self.core.scalar_data_mem
        vmem = self.core.vector_data_mem
        vector_length = vrf.vector_length_register

        # Classify instruction
        action = functionality["mem_op"]  # load or save
        # scalar, vector, strided, scatter/gather
        mem_type = functionality["mem_type"]

        # Act differently according to instruction parsing results
        if mem_type == "S":  # scalar
            immediate = int(instruction["operand3"])

            if action == "L":  # load: reg = mem[reg + imm]
                base_address = srf[self.reg_index(instruction["operand2"])]
                srf[self.reg_index(
                    instruction["operand1"])] = smem[base_address + immediate]
            else:  # store: mem[reg + imm] = reg
                base_address = srf[self.reg_index(instruction["operand2"])]
                value = srf[self.reg_index(instruction["operand1"])]
                smem[base_address + immediate] = value

            return ScalarInstruction([instruction["operand2"]],
                                     instruction["operand1"])

        elif mem_type.startswith("V"):  # vector
            # Generalize all vector load-stores as scatter/gather
            if (strided := mem_type == "VWS") or mem_type == "V":
                # Generalize basic vector load-stores as stride=1
                stride = srf[self.reg_index(
                    instruction["operand3"])] if strided else 1
                if stride == 0:
                    # special case that spreads a single value from vector mem to the entire vector register
                    offsets = list([0] * vrf.vec_size)
                else:
                    offsets = list(range(0, vrf.vec_size * stride, stride))
            elif mem_type == "VI":  # gather/scatter
                offsets = vrf[self.reg_index(instruction["operand3"])]
            else:
                raise NotImplementedError

            base_address = srf[self.reg_index(instruction["operand2"])]

            if action == "L":  # load/gather
                values = [vmem[base_address + offset] for offset in offsets]
                # Handle vector length and vector mask effects
                original_values = vrf[self.reg_index(instruction["operand1"])]
                masked_values = [
                    val if mask else orig for mask, orig, val in tuple(
                        zip(vrf.vector_mask_register, original_values, values))
                    [:vector_length]
                ]

                vrf[self.reg_index(
                    instruction["operand1"])][:vector_length] = masked_values

            else:  # store/scatter
                values = vrf[self.reg_index(
                    instruction["operand1"])][:vector_length]

                for lane in range(vector_length):
                    offset = offsets[lane]
                    # Handle vector length and vector mask effects
                    enable = vrf.vector_mask_register[lane]
                    if enable:
                        vmem[base_address + offset] = values[lane]

            dependencies = ["VLR", "VMR", instruction["operand2"]]
            if mem_type != "V":
                dependencies.append(instruction["operand3"])
            return VectorMemoryInstruction(
                dependencies,
                instruction["operand1"],
                [(base_address + offset) for offset in offsets][:vector_length],
            )

        else:
            raise RuntimeError("Unknown memory operation instruction:",
                               instruction.groupdict())

    def scalar_op(self, functionality, instruction):
        # Aliases
        srf = self.core.scalar_register_file

        # Get operands
        operand2 = srf[self.reg_index(instruction["operand2"])]
        operand3 = srf[self.reg_index(instruction["operand3"])]

        # Map operation name to standard operators
        op_code = functionality["scalar_op"].lower()
        # Slightly alter operation name to get the bitwise version
        if op_code in ("and", "or"):  # xor stays the same
            op_code += "_"
        elif op_code in ("sll", "sra"):
            op_code = f"{op_code[1]}shift"
        operation = operator.methodcaller(op_code, operand2, operand3)

        # Do operation and store the result
        # Python stores integers in large containers, so native >> behaves like
        # arithmetic right shift, and it needs special steps to get signed-int32
        # logical right shift behavior
        if op_code == "srl":
            result = (operand2 & 0xFFFF_FFFF) >> operand3
        else:
            result = operation(operator)

        # handle OverflowError
        if op_code == "lshift":
            negative = result & (1 << 31)
            if negative:

                def twos_complement(x):
                    sign = -1 if (x & (1 << 31)) else 1
                    # Python needs the sign even if the underlying container is
                    # designated as signed 32-bit integers
                    return sign * (((
                        (x & 0xFFFF_FFFF) ^ 0xFFFF_FFFF) + 1) & 0xFFFF_FFFF)

                result = twos_complement(result)
            else:
                result &= 0xFFFF_FFFF
        srf[self.reg_index(instruction["operand1"])] = result
        return ScalarInstruction(
            [instruction["operand2"], instruction["operand3"]],
            instruction["operand1"])

    def control(self, functionality, instruction):
        # Aliases
        srf = self.core.scalar_register_file

        # Get operands
        operand1 = srf[self.reg_index(instruction["operand1"])]
        operand2 = srf[self.reg_index(instruction["operand2"])]
        immediate = int(instruction["operand3"])
        assert immediate <= pow(2, 20), "immediate too large"
        assert -pow(2, 20) <= immediate, "immediate too small"

        # Map operation name to standard operators
        condition = functionality["branch_condition"].lower()
        operation = operator.methodcaller(condition, operand1, operand2)
        compare_result = operation(operator)

        # Do operation
        if compare_result is True:
            # Branch taken
            self.core.PC += immediate - 1
        else:
            # Branch not taken
            pass

        return ControlInstruction(
            [instruction["operand1"], instruction["operand2"]], None)

    def stop(self, functionality, instruction):
        # Halt
        self.core.freeze = True
        return ScalarInstruction([], None)


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

            return cell[int(reg_idx)]
        else:
            return self.__getattribute__(name)

    @classmethod
    def decode(cls, statement):
        # TODO: test: parse then reconstruct, to check if operand parsing order is correct (if operand-1 is parsed as operand-3)
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
        if self.freeze is not True:
            current_line = self.instruction_mem[self.PC]
            self.alu.do(self.decode(current_line))
            if self.PC + 1 < len(self.instruction_mem.instructions):
                self.PC += 1
            else:
                # Automatically stops after stepping over the last line
                self.freeze = True

    def step_instr(self):
        """Executes one instruction
        Step over empty or comment lines
        """
        # Skip non-statement lines
        next_line = self.instruction_mem.instructions[self.PC]
        next_instruction = self.decode(next_line)["instruction"]
        # FIXME: does prefetch/predecode next line causes index problems?
        while next_instruction is None:
            self.step()
            next_line = self.instruction_mem.instructions[self.PC]
            next_instruction = self.decode(next_line)["instruction"]

        self.step()  # Execute the next line, which will not be empty or comment

    def run(self):
        while self.freeze is not True:
            # TODO: count instructions as they flow pass
            self.step_instr()
        if "config" in globals():
            self.measure_time_spent()

    def measure_time_spent(self):
        """Reconstruct timing simulation after halting
        Schedule instructions at the DecodeStage instead of FetchStage, and
        drives functional units with an event based framework
        """
        instruction_stream = self.alu.instruction_stream
        n_banks = config.parameters["vdmNumBanks"]
        n_lanes = config.parameters["numLanes"]

        print(f"instructions executed: {len(instruction_stream)}")

        # Scoreboard: instruction/functional units/register status
        class Scoreboard:
            register: dict[str, bool] = {
                "VLR": False,
                "VMR": False
            } | {
                f"SR{key}": False
                for key in range(self.scalar_register_file.n_reg)
            } | {
                f"VR{key}": False
                for key in range(self.vector_register_file.n_reg)
            }

            functional_unit: dict[str, bool] = {
                key: None
                for key in ("vec_addsub", "vec_mul", "vec_div", "vec_mem")
            }

            instruction: dict[int, InstructionType] = {}

        class DispatchQueue:
            scalar_q = queue.Queue(maxsize=1)
            vector_compute_q = queue.Queue(
                maxsize=config.parameters["computeQueueDepth"])
            vector_mem_q = queue.Queue(
                maxsize=config.parameters["dataQueueDepth"])

        ScalarFunctionalUnit = queue.Queue(maxsize=1)

        class VectorFunctionalUnit:
            addsub = queue.Queue(maxsize=config.parameters["pipelineDepthAdd"])
            multiply = queue.Queue(
                maxsize=config.parameters["pipelineDepthMul"])
            divide = queue.Queue(maxsize=config.parameters["pipelineDepthDiv"])

        class MemoryController:
            banks = {
                idx: queue.Queue(maxsize=config.parameters["vlsPipelineDepth"])
                for idx in range(n_banks)
            }
            busy_counter = {idx: 0 for idx in range(n_banks)}

        # Prefill with empty bubbles
        for pipeline in [
                VectorFunctionalUnit.addsub,
                VectorFunctionalUnit.multiply,
                VectorFunctionalUnit.divide,
                ScalarFunctionalUnit,
        ]:
            for i in range(pipeline.maxsize):
                pipeline.put_nowait(None)
        for bank_idx in MemoryController.banks:
            for i in range(config.parameters["vlsPipelineDepth"]):
                MemoryController.banks[bank_idx].put_nowait(None)

        cycle_counter = 1  # start from 1 in 'Decode' because it's already fetched
        pseudo_PC = 0
        retired_instruction_count = 0
        stall_reason = {
            DispatchQueue.scalar_q: 0,
            DispatchQueue.vector_compute_q: 0,
            DispatchQueue.vector_mem_q: 0,
            "other": 0,
        }

        while retired_instruction_count < len(instruction_stream):
            if pseudo_PC < len(instruction_stream):
                instruction = instruction_stream[pseudo_PC]
            else:
                # Insert bubbles until the simulation finishes
                instruction = ScalarInstruction([], None)
                dprint(bgcolor("cyan")("(insert EMPTY instruction)"))
            dprint(
                bgcolor("bright_black")(f"{pseudo_PC}" + color("bright_cyan")
                                        (instruction.__class__.__name__)))
            dprint("\toperands:", sorted(instruction.operands))
            dprint("\tresult:", instruction.result)
            if isinstance(instruction, VectorComputeInstruction):
                dprint("\tvecLen:", instruction.vec_len)
            if isinstance(instruction, VectorMemoryInstruction):
                dprint("\tvecLen:", len(instruction.target_addrs))
                dprint("\taddr:", instruction.target_addrs)
                dprint("\tmem bank:",
                       [x % n_banks for x in instruction.target_addrs])

            # Start updating status from the end of the pipeline first
            # Update functional unit pipelines
            if (scalar_result := ScalarFunctionalUnit.get_nowait()) is not None:
                # Do useful work
                # Then
                dprint(bgcolor("green")(f"retire {scalar_result}"))
                retired_instruction_count += 1

            vec_addsub_result = VectorFunctionalUnit.addsub.get_nowait()
            if (vec_addsub_instr_id := Scoreboard.functional_unit["vec_addsub"]
               ) is not None:  # in use
                # Check if work is done
                pipe_tasks = set(VectorFunctionalUnit.addsub.queue)
                if vec_addsub_result == vec_addsub_instr_id and len(
                        pipe_tasks) == 1 and None in pipe_tasks:
                    Scoreboard.functional_unit["vec_addsub"] = None
                    dprint(bgcolor("green")(f"retire {vec_addsub_instr_id}"))
                    retired_instruction_count += 1
                else:
                    if instruction_stream[vec_addsub_instr_id].vec_len > 0:
                        VectorFunctionalUnit.addsub.put_nowait(
                            vec_addsub_instr_id)
                        instruction_stream[
                            vec_addsub_instr_id].vec_len -= n_lanes
                    else:
                        VectorFunctionalUnit.addsub.put_nowait(None)

            vec_mul_result = VectorFunctionalUnit.multiply.get_nowait()
            if (vec_mul_instr_id := Scoreboard.functional_unit["vec_mul"]
               ) is not None:  # in use
                # Check if work is done
                pipe_tasks = set(VectorFunctionalUnit.multiply.queue)
                if vec_mul_result == vec_mul_instr_id and len(
                        pipe_tasks) == 1 and None in pipe_tasks:
                    Scoreboard.functional_unit["vec_mul"] = None
                    dprint(bgcolor("green")(f"retire {vec_mul_instr_id}"))
                    retired_instruction_count += 1
                else:
                    if instruction_stream[vec_mul_instr_id].vec_len > 0:
                        VectorFunctionalUnit.multiply.put_nowait(
                            vec_mul_instr_id)
                        instruction_stream[vec_mul_instr_id].vec_len -= n_lanes
                    else:
                        VectorFunctionalUnit.multiply.put_nowait(None)

            vec_div_result = VectorFunctionalUnit.divide.get_nowait()
            if (vec_div_instr_id := Scoreboard.functional_unit["vec_div"]
               ) is not None:  # in use
                # Check if work is done
                pipe_tasks = set(VectorFunctionalUnit.divide.queue)
                if vec_div_result == vec_div_instr_id and len(
                        pipe_tasks) == 1 and None in pipe_tasks:
                    Scoreboard.functional_unit["vec_div"] = None
                    dprint(bgcolor("green")(f"retire {vec_div_instr_id}"))
                    retired_instruction_count += 1
                else:
                    if instruction_stream[vec_div_instr_id].vec_len > 0:
                        VectorFunctionalUnit.divide.put_nowait(vec_div_instr_id)
                        instruction_stream[vec_div_instr_id].vec_len -= n_lanes
                    else:
                        VectorFunctionalUnit.divide.put_nowait(None)

            mem_results = [
                MemoryController.banks[bank_idx].get_nowait()
                for bank_idx in MemoryController.banks
            ]

            if (vector_mem_id := Scoreboard.functional_unit["vec_mem"]
               ) is not None:  # in use
                # Check if work is done
                pipe_tasks = set(
                    itertools.chain.from_iterable(
                        bank.queue for bank in MemoryController.banks.values()))
                if vector_mem_id in mem_results and len(
                        pipe_tasks) == 1 and None in pipe_tasks:
                    Scoreboard.functional_unit["vec_mem"] = None
                    dprint(bgcolor("green")(f"retire {vector_mem_id}"))
                    retired_instruction_count += 1
                else:
                    for bank_idx in MemoryController.banks:
                        if MemoryController.busy_counter[bank_idx] > 0:
                            MemoryController.banks[bank_idx].put_nowait(
                                vector_mem_id)
                            MemoryController.busy_counter[bank_idx] -= 1
                        else:
                            pass  # Take new requests from dispatch queue later

            # Each dispatch queue can issue one instruction per cycle
            # Dispatch if there's no structural hazard
            # NOTE: baseline uArch does not allow instruction overlapping
            if not DispatchQueue.scalar_q.empty():
                ScalarFunctionalUnit.put_nowait(
                    DispatchQueue.scalar_q.get_nowait())
            else:
                if not ScalarFunctionalUnit.full():
                    ScalarFunctionalUnit.put_nowait(None)

            if not DispatchQueue.vector_compute_q.empty():
                # Check for structural hazards
                vec_instr_id = DispatchQueue.vector_compute_q.queue[0]
                vec_instr = instruction_stream[vec_instr_id]
                if isinstance(vec_instr, VectorAddSubInstruction):
                    if not VectorFunctionalUnit.addsub.full():
                        if Scoreboard.functional_unit["vec_addsub"] is None:
                            DispatchQueue.vector_compute_q.get_nowait()
                            VectorFunctionalUnit.addsub.put_nowait(vec_instr_id)
                            Scoreboard.functional_unit[
                                "vec_addsub"] = vec_instr_id
                            vec_instr.vec_len -= n_lanes
                        else:
                            VectorFunctionalUnit.addsub.put_nowait(None)
                elif isinstance(vec_instr, VectorMultiplyInstruction):
                    if not VectorFunctionalUnit.multiply.full():
                        if Scoreboard.functional_unit["vec_mul"] is None:
                            DispatchQueue.vector_compute_q.get_nowait()
                            VectorFunctionalUnit.multiply.put_nowait(
                                vec_instr_id)
                            Scoreboard.functional_unit["vec_mul"] = vec_instr_id
                            vec_instr.vec_len -= n_lanes
                        else:
                            VectorFunctionalUnit.multiply.put_nowait(None)
                elif isinstance(vec_instr, VectorDivideInstruction):
                    if not VectorFunctionalUnit.divide.full():
                        if Scoreboard.functional_unit["vec_div"] is None:
                            DispatchQueue.vector_compute_q.get_nowait()
                            VectorFunctionalUnit.divide.put_nowait(vec_instr_id)
                            Scoreboard.functional_unit["vec_div"] = vec_instr_id
                            vec_instr.vec_len -= n_lanes
                        else:
                            VectorFunctionalUnit.divide.put_nowait(None)
            # Filling in the pipeline
            dprint("\t" + bgcolor("yellow")("fill pipeline with bubble"))
            for name, pipeline in {
                    "addsub FU pipe": VectorFunctionalUnit.addsub,
                    "mul    FU pipe": VectorFunctionalUnit.multiply,
                    "div    FU pipe": VectorFunctionalUnit.divide
            }.items():
                dprint("\t\t" + bgcolor("yellow")(f"{name}:{pipeline.qsize()}"),
                       end="")
                if not pipeline.full():
                    pipeline.put_nowait(None)
                    dprint(bgcolor("green")(f"->{pipeline.qsize()}"), end="")
                dprint()

            banks = MemoryController.banks
            busy_counter = MemoryController.busy_counter
            if not DispatchQueue.vector_mem_q.empty():
                # TODO: FIXME:
                mem_instr_id = DispatchQueue.vector_mem_q.queue[0]
                mem_instr = instruction_stream[mem_instr_id]

                # Special optimization to reduce bank conflicts:
                # Send only 1 memory request if stride=0
                if config.parameters["smartVectorMemoryStride"]:
                    if len(set(mem_instr.target_addrs)) == 1:
                        mem_instr.target_addrs = mem_instr.target_addrs[:1]

                if (Scoreboard.functional_unit["vec_mem"] is None):
                    # Start new instruction
                    DispatchQueue.vector_mem_q.get_nowait()
                    Scoreboard.functional_unit["vec_mem"] = mem_instr_id
                    # Try to send the top <numLane> requests
                    requests = mem_instr.target_addrs[:n_lanes]
                    for addr in requests:  # Try to place individual requests
                        bank_addr = addr % n_banks
                        if busy_counter[bank_addr]:
                            # Bank busy with its previous request
                            pass
                        else:
                            # Bank free, place request
                            if not banks[bank_addr].full():
                                banks[bank_addr].put_nowait(mem_instr_id)
                                busy_counter[bank_addr] = config.parameters[
                                    "bankbusytime"] - 1
                                mem_instr.target_addrs.remove(addr)
                else:
                    # Send requests from the current instruction
                    current_instr_idx = Scoreboard.functional_unit["vec_mem"]
                    current_instr = instruction_stream[current_instr_idx]
                    # Try to send the top <numLane> requests
                    requests = current_instr.target_addrs[:n_lanes]
                    for addr in requests:  # Try to place individual requests
                        bank_addr = addr % n_banks
                        if busy_counter[bank_addr]:
                            # Bank busy with its previous request
                            pass
                        else:
                            # Bank free, place request
                            if not banks[bank_addr].full():
                                banks[bank_addr].put_nowait(current_instr_idx)
                                busy_counter[bank_addr] = config.parameters[
                                    "bankbusytime"] - 1
                                current_instr.target_addrs.remove(addr)

            # Fill in bubbles
            for bank in MemoryController.banks.values():
                if not bank.full():
                    bank.put_nowait(None)

            dprint("@")
            dprint("ds <-", list(DispatchQueue.scalar_q.queue))
            dprint("dv <-", list(DispatchQueue.vector_compute_q.queue))
            dprint("dm <-", list(DispatchQueue.vector_mem_q.queue))
            dprint("@@")
            dprint("ps  ->", list(reversed(ScalarFunctionalUnit.queue)))
            dprint("pva ->", list(reversed(VectorFunctionalUnit.addsub.queue)))
            dprint("pvm ->",
                   list(reversed(VectorFunctionalUnit.multiply.queue)))
            dprint("pvd ->", list(reversed(VectorFunctionalUnit.divide.queue)))
            dprint("pM  -> ", end="")
            for bank_idx, bank in MemoryController.banks.items():
                dprint(
                    f"{'' if bank_idx == 0 else ' '*7}<{bank_idx:2d}> "
                    f"({'busy' if MemoryController.busy_counter[bank_idx] else 'free'})",
                    list(reversed(bank.queue)))
            dprint("@@@")

            # Update decode stage
            # Check for data hazards
            if isinstance(instruction, ScalarInstruction):
                dest_queue = DispatchQueue.scalar_q
            elif isinstance(instruction, VectorComputeInstruction):
                dest_queue = DispatchQueue.vector_compute_q
            elif isinstance(instruction, VectorMemoryInstruction):
                dest_queue = DispatchQueue.vector_mem_q
            else:
                raise RuntimeError("Unknown instruction category: " +
                                   type(instruction).__name__)
            if dest_queue.full() or any(Scoreboard.register[operand]
                                        for operand in instruction.operands):
                # Stall
                if dest_queue.full():
                    dprint(
                        color("red")(
                            f"{type(instruction).__name__} queue full"))
                    stall_reason[dest_queue] += 1
                if any(Scoreboard.register[operand]
                       for operand in instruction.operands):
                    for op in instruction.operands:
                        if Scoreboard.register[op] is True:
                            dprint(color("red")(f"{op} is in use"))
                            # TODO: track and show who is using this register
                    stall_reason["other"] += 0 if dest_queue.full() else 1
                dprint(bgcolor("red")(f"stall instr: {pseudo_PC}"))
            else:
                # Dispatch instruction
                dest_queue.put_nowait(pseudo_PC)
                dprint(bgcolor("blue")(f"dispatch instr: {pseudo_PC}"))
                pseudo_PC += 1

            # Check if all FU pipelines are filled
            assert ScalarFunctionalUnit.full()
            assert VectorFunctionalUnit.addsub.full()
            assert VectorFunctionalUnit.multiply.full()
            assert VectorFunctionalUnit.divide.full()
            assert all(map(lambda x: x.full(), MemoryController.banks.values()))

            dprint(f"time: {cycle_counter} cycles")
            dprint(
                "instr done: "
                f"{retired_instruction_count}/{len(instruction_stream)} "
                f"({100*retired_instruction_count/len(instruction_stream):.2f}%)"
            )
            cycle_counter += 1

        print(f"took {cycle_counter} cycles to run program")
        stall_count = sum(stall_reason.values())
        print(f"stalled {stall_count} times, "
              f"{100*stall_count/cycle_counter:.2f}% of execution time")
        print("stall reason")
        print("\tscalar queue full: "
              f"{stall_reason[DispatchQueue.scalar_q]}")
        print("\tvector compute queue full: "
              f"{stall_reason[DispatchQueue.vector_compute_q]}")
        print("\tvector memory queue full: "
              f"{stall_reason[DispatchQueue.vector_mem_q]}")
        print("\tother: "
              f"{stall_reason['other']}")

    def dump(self, path: pathlib.Path = None):
        self.scalar_register_file.dump(path)
        self.vector_register_file.dump(path)
        self.scalar_data_mem.dump(path)
        self.vector_data_mem.dump(path)

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


class Config:
    """Configuration holder class
    """

    def __init__(self, cfg_file: pathlib.Path):
        assert cfg_file.is_file()
        with cfg_file.open(mode="r", encoding="ascii") as cfg:
            self.parameters = {
                key: int(val) for key, val in re.findall(
                    r"^(\w+)\s*=\s*(\d+)", cfg.read(), re.MULTILINE)
            }

        print("Config parameters:")
        pprint.pprint(self.parameters, sort_dicts=False)
        print()


if __name__ == "__main__":
    # Parse arguments for input file location
    parser = argparse.ArgumentParser(
        description="Vector Core Performance Model")
    parser.add_argument(
        "--iodir",
        default="",
        type=str,
        help=
        "Path to the folder containing the input files - instructions and data."
    )
    parsed_args = parser.parse_args()

    io_dir = pathlib.Path(parsed_args.iodir).absolute()
    print("IO Directory:", io_dir)
    if not (config_file := io_dir / "Config.txt").is_file():
        print("\n" + "-" * 16)
        print(f"Couldn't find config file at {config_file!s}")
        config_file = pathlib.Path.cwd() / "Config.txt"
        print(f"Falling back to default config file at {config_file!s}")
        print("-" * 16 + "\n")
    config = Config(config_file)

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
