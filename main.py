import dataclasses
import itertools
import unittest
from dataclasses import dataclass


ir = dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
instr_counter = itertools.count()


def reset_instr_counter():
    global instr_counter
    instr_counter = itertools.count()


@ir
class Instr:
    id: int = dataclasses.field(default_factory=lambda: next(instr_counter), init=False)

    def var(self):
        return f"v{self.id}"

    def children(self):
        return tuple(
            getattr(self, name)
            for name, field in self.__dataclass_fields__.items()
            if issubclass(field.type, Instr)
        )

    def __repr__(self):
        op = self.__class__.__name__
        return (
            f"{self.var()} = {op} {', '.join(child.var() for child in self.children())}"
        )


@ir
class Const(Instr):
    value: int

    def __repr__(self):
        return f"{self.var()} = {self.value}"


@ir
class Array(Instr):
    value: tuple[Instr]

    def children(self):
        return self.value


@ir
class Add(Instr):
    left: Instr
    right: Instr


@ir
class Mul(Instr):
    left: Instr
    right: Instr


@ir
class Dot(Instr):
    left: Array
    right: Array


def topo(self):
    # topological order all of the children in the graph
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v.children():
                build_topo(child)
            topo.append(v)

    build_topo(self)
    return topo


x86 = dataclass(eq=True, frozen=True)


@x86
class Operand:
    pass


@x86
class Reg(Operand):
    index: int

    def __repr__(self):
        names = (
            "rax",
            "rcx",
            "rdx",
            "rbx",
            "rsp",
            "rbp",
            "rsi",
            "rdi",
            "r8",
            "r9",
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r15",
        )
        return names[self.index]


RAX = Reg(0)
RCX = Reg(1)
RDX = Reg(2)
RBX = Reg(3)
RSP = Reg(4)
RBP = Reg(5)
RSI = Reg(6)
RDI = Reg(7)
R8 = Reg(8)
R9 = Reg(9)
R10 = Reg(10)
R11 = Reg(11)
R12 = Reg(12)
R13 = Reg(13)
R14 = Reg(14)
R15 = Reg(15)


@x86
class Mem(Operand):
    base: Reg


@x86
class BaseDisp(Mem):
    disp: int

    def __repr__(self):
        base = self.base
        disp = self.disp
        if self.disp == 0:
            return f"[{base}]"
        elif self.disp > 0:
            return f"[{base}+{disp}]"
        else:
            return f"[{base}{disp}]"


@x86
class Imm(Operand):
    value: int

    def size(self):
        bl = self.value.bit_length()
        if bl <= 8:
            return 8
        if bl <= 16:
            return 16
        if bl <= 32:
            return 32
        if bl <= 64:
            return 64
        raise NotImplementedError(f"const {self.value} too big")

    def as_bytes(self):
        return self.value.to_bytes(self.size(), byteorder="little", signed=True)

    def __repr__(self):
        return f"Imm({self.value})"


class X86:
    @x86
    class X86Instr:
        pass

    @x86
    class Mov(X86Instr):
        dst: Operand
        src: Operand

    @x86
    class Add(X86Instr):
        dst: Operand
        src: Operand

    @x86
    class Mul(X86Instr):
        dst: Operand
        src: Operand

    @x86
    class Push(X86Instr):
        src: Operand

    @x86
    class Pop(X86Instr):
        dst: Operand


def naive(op):
    if isinstance(op, Const):
        return [X86.Mov(RAX, Imm(op.value))]
    elif isinstance(op, (Add, Mul)):
        right_code = naive(op.right)
        left_code = naive(op.left)
        opcode = {Add: X86.Add, Mul: X86.Mul}[type(op)]
        return [
            *right_code,
            X86.Push(RAX),
            *left_code,
            X86.Pop(RCX),
            opcode(RAX, RCX),
        ]


class Dest:
    STACK = 0
    ACCUM = 1
    NOWHERE = 2


class DDCG:
    def __init__(self):
        self.code = []

    def emit(self, op):
        self.code.append(op)

    def compile(self, exp):
        self._compile(exp, Dest.ACCUM)

    def _compile(self, exp, dst):
        tmp = RCX
        if isinstance(exp, Const):
            self._plug_imm(dst, exp.value)
        elif isinstance(exp, (Add, Mul)):
            self._compile(exp.left, Dest.STACK)
            self._compile(exp.right, Dest.ACCUM)
            self.emit(X86.Pop(tmp))
            opcode = {Add: X86.Add, Mul: X86.Mul}[type(exp)]
            self.emit(opcode(RAX, tmp))
            self._plug_reg(dst, RAX)
        else:
            raise NotImplementedError(exp)

    def _plug_imm(self, dst, value):
        if dst == Dest.STACK:
            self.emit(X86.Push(Imm(value)))
        elif dst == Dest.ACCUM:
            self.emit(X86.Mov(RAX, Imm(value)))
        else:
            raise NotImplementedError

    def _plug_reg(self, dst, reg):
        if dst == Dest.STACK:
            self.emit(X86.Push(reg))
        elif dst == Dest.ACCUM:
            if reg == RAX:
                pass
            else:
                raise NotImplementedError
                # self.emit(X86.Mov(RAX, reg))
        else:
            raise NotImplementedError


STACK_REGS = [R8, R9]


class DDCGStack:
    def __init__(self):
        self.code = []
        self.sp = 0

    def emit(self, op):
        self.code.append(op)

    def compile(self, exp):
        self._compile(exp, Dest.ACCUM)

    def push(self, val):
        assert isinstance(val, (Reg, Imm)), f"unexpected value {val}"
        sp = self.sp
        vreg_push = sp < len(STACK_REGS)
        self.sp += 1
        if not vreg_push:
            self.emit(X86.Push(val))
            return
        dst = STACK_REGS[sp]
        self.emit(X86.Mov(dst, val))

    def pop(self, dst):
        assert isinstance(dst, Reg), f"unexpected destination {dst}"
        assert self.sp > 0, f"stack underflow (sp is {self.sp})"
        vreg_pop = self.sp <= len(STACK_REGS)
        self.sp -= 1
        if not vreg_pop:
            self.emit(X86.Pop(dst))
            return
        src = STACK_REGS[self.sp]
        self.emit(X86.Mov(dst, src))

    def top_in_reg(self):
        return self.sp <= len(STACK_REGS)

    def top_reg(self):
        assert self.top_in_reg()
        return STACK_REGS[self.sp - 1]

    def _compile(self, exp, dst):
        if isinstance(exp, Const):
            self._plug_imm(dst, exp.value)
        elif isinstance(exp, (Add, Mul)):
            self._compile(exp.left, Dest.STACK)
            self._compile(exp.right, Dest.ACCUM)
            opcode = {Add: X86.Add, Mul: X86.Mul}[type(exp)]
            if self.top_in_reg():
                self.emit(opcode(RAX, self.top_reg()))
                self.sp -= 1
            else:
                tmp = RCX
                self.pop(tmp)
                self.emit(opcode(RAX, tmp))
            self._plug_reg(dst, RAX)
        else:
            raise NotImplementedError(exp)

    def _plug_imm(self, dst, value):
        if dst == Dest.STACK:
            self.push(Imm(value))
        elif dst == Dest.ACCUM:
            self.emit(X86.Mov(RAX, Imm(value)))
        else:
            raise NotImplementedError

    def _plug_reg(self, dst, reg):
        if dst == Dest.STACK:
            self.push(reg)
        elif dst == Dest.ACCUM:
            if reg == RAX:
                pass
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


class Simulator:
    def __init__(self):
        self.regs = [0] * 16
        self.memory = bytearray([0] * 256)
        self.code = []
        self.regs[RSP.index] = len(self.memory) // 2
        assert self.reg(RSP) % 8 == 0

    def load(self, code):
        self.code = code

    def run(self):
        for op in self.code:
            self.run_one(op)

    def reg(self, reg):
        return self.regs[reg.index]

    def memory_write(self, idx, value, nbytes):
        value_bytes = value.to_bytes(nbytes, byteorder="little", signed=True)
        self.memory[idx : idx + nbytes] = value_bytes

    def memory_write_imm(self, idx, imm):
        bs = imm.as_bytes()
        assert imm.size() == len(bs)
        self.memory[idx : idx + imm.size()] = bs

    def memory_read(self, idx, nbytes, signed=False):
        return int.from_bytes(
            self.memory[idx : idx + nbytes], byteorder="little", signed=signed
        )

    def stack_push(self, value):
        rsp = self.reg(RSP)
        nbytes = 8
        value_bytes = value.to_bytes(nbytes, byteorder="little", signed=True)
        self.memory[rsp : rsp + nbytes] = value_bytes
        self.regs[RSP.index] -= nbytes

    def stack_pop(self):
        nbytes = 8
        self.regs[RSP.index] += nbytes
        rsp = self.reg(RSP)
        return int.from_bytes(
            self.memory[rsp : rsp + nbytes], byteorder="little", signed=False
        )

    def run_one(self, op):
        if isinstance(op, X86.Mov):
            if isinstance(op.dst, Reg):
                if isinstance(op.src, Imm):
                    self.regs[op.dst.index] = op.src.value
                elif isinstance(op.src, Reg):
                    self.regs[op.dst.index] = self.reg(op.src)
                elif isinstance(op.src, Mem):
                    assert isinstance(
                        op.src, BaseDisp
                    ), "more complex memory not supported"
                    addr = self.reg(op.src.base) + op.src.disp
                    # TODO(max): Get read size from register size
                    self.regs[op.dst.index] = self.memory_read(addr, nbytes=8)
                else:
                    assert isinstance(op.src, Imm), "non-imm src unsupported"
            elif isinstance(op.dst, Mem):
                assert isinstance(op.dst, BaseDisp), "more complex memory not supported"
                assert op.dst.base == RSP, "non-stack memory unsupported"
                addr = self.reg(op.dst.base) + op.dst.disp
                if isinstance(op.src, Imm):
                    self.memory_write_imm(addr, op.src)
                elif isinstance(op.src, Reg):
                    value = self.reg(op.src)
                    # TODO(max): Get write size from register size
                    self.memory_write(addr, value, nbytes=8)
                else:
                    raise NotImplementedError("non-imm src")
            else:
                assert isinstance(op.dst, Reg), "non-reg dst unsupported"
        elif isinstance(op, X86.Add):
            if not isinstance(op.dst, Reg):
                raise NotImplementedError(f"only reg dst is supported: {op.dst}")
            if isinstance(op.src, Reg):
                self.regs[op.dst.index] = self.reg(op.dst) + self.reg(op.src)
            elif isinstance(op.src, Mem):
                assert isinstance(op.src, BaseDisp), "more complex memory not supported"
                assert op.src.base == RSP, "non-stack memory unsupported"
                self.regs[op.dst.index] = self.reg(op.dst) + self.memory_read(
                    op.src.disp, nbytes=8
                )
            else:
                raise NotImplementedError("only add reg, reg/mem is supported")
        elif isinstance(op, X86.Mul):
            if not isinstance(op.dst, Reg):
                raise NotImplementedError(f"only reg dst is supported: {op.dst}")
            if isinstance(op.src, Reg):
                self.regs[op.dst.index] = self.reg(op.dst) * self.reg(op.src)
            elif isinstance(op.src, Mem):
                assert isinstance(op.src, BaseDisp), "more complex memory not supported"
                assert op.src.base == RSP, "non-stack memory unsupported"
                self.regs[op.dst.index] = self.reg(op.dst) * self.memory_read(
                    op.src.disp, nbytes=8
                )
            else:
                raise NotImplementedError("only mul reg, reg is supported")
        elif isinstance(op, X86.Push):
            if isinstance(op.src, Imm):
                self.stack_push(op.src.value)
            elif isinstance(op.src, Reg):
                value = self.reg(op.src)
                self.stack_push(value)
            else:
                raise NotImplementedError("push with non-imm")
        elif isinstance(op, X86.Pop):
            if isinstance(op.dst, Reg):
                value = self.stack_pop()
                self.regs[op.dst.index] = value
            else:
                raise NotImplementedError("pop with non-reg")
        else:
            raise NotImplementedError(op)


class IrTests(unittest.TestCase):
    def setUp(self):
        reset_instr_counter()


class RenderTests(IrTests):
    def test_const(self):
        exp = Const(123)
        self.assertEqual(repr(exp), "v0 = 123")

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(repr(exp), "v2 = Add v0, v1")


class TopoTests(IrTests):
    def _topo(self, exp):
        ops = topo(exp)
        return [str(op) for op in ops]

    def test_const(self):
        exp = Const(2)
        self.assertEqual(self._topo(exp), ["v0 = 2"])

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(self._topo(exp), ["v0 = 2", "v1 = 3", "v2 = Add v0, v1"])

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(self._topo(exp), ["v0 = 2", "v1 = 3", "v2 = Mul v0, v1"])


class NaiveTests(unittest.TestCase):
    def _alloc(self, exp):
        x86 = naive(exp)
        return [str(op) for op in x86]

    def test_const(self):
        exp = Const(2)
        self.assertEqual(
            self._alloc(exp),
            ["X86.Mov(dst=rax, src=Imm(2))"],
        )

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(2))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
            ],
        )

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(2))",
                "X86.Pop(dst=rcx)",
                "X86.Mul(dst=rax, src=rcx)",
            ],
        )

    def test_mul_add(self):
        exp = Mul(
            Add(Const(1), Const(2)),
            Add(Const(3), Const(4)),
        )
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=rax, src=Imm(4))",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(2))",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(1))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Pop(dst=rcx)",
                "X86.Mul(dst=rax, src=rcx)",
            ],
        )


class DDCGTests(unittest.TestCase):
    def _alloc(self, exp):
        gen = DDCG()
        gen.compile(exp)
        return [str(op) for op in gen.code]

    def test_const(self):
        exp = Const(2)
        self.assertEqual(
            self._alloc(exp),
            ["X86.Mov(dst=rax, src=Imm(2))"],
        )

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Push(src=Imm(2))",
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
            ],
        )

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Push(src=Imm(2))",
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Pop(dst=rcx)",
                "X86.Mul(dst=rax, src=rcx)",
            ],
        )

    def test_mul_add(self):
        exp = Mul(
            Add(Const(1), Const(2)),
            Add(Const(3), Const(4)),
        )
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Push(src=Imm(1))",
                "X86.Mov(dst=rax, src=Imm(2))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Push(src=rax)",
                "X86.Push(src=Imm(3))",
                "X86.Mov(dst=rax, src=Imm(4))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Pop(dst=rcx)",
                "X86.Mul(dst=rax, src=rcx)",
            ],
        )


class DDCGStackTests(unittest.TestCase):
    def _alloc(self, exp):
        gen = DDCGStack()
        gen.compile(exp)
        return [str(op) for op in gen.code]

    def test_const(self):
        exp = Const(2)
        self.assertEqual(
            self._alloc(exp),
            ["X86.Mov(dst=rax, src=Imm(2))"],
        )

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Add(dst=rax, src=r8)",
            ],
        )

    def test_add_deep(self):
        # This tests pushing and popping beyond the limits of our virtual
        # stack.
        assert len(STACK_REGS) == 2
        exp = Add(Const(2), Add(Const(3), Add(Const(4), Const(5))))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=r9, src=Imm(3))",
                "X86.Push(src=Imm(4))",
                "X86.Mov(dst=rax, src=Imm(5))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Add(dst=rax, src=r9)",
                "X86.Add(dst=rax, src=r8)",
            ],
        )

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Mul(dst=rax, src=r8)",
            ],
        )

    def test_mul_add(self):
        exp = Mul(
            Add(Const(1), Const(2)),
            Add(Const(3), Const(4)),
        )
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(1))",
                "X86.Mov(dst=rax, src=Imm(2))",
                "X86.Add(dst=rax, src=r8)",
                "X86.Mov(dst=r8, src=rax)",
                "X86.Mov(dst=r9, src=Imm(3))",
                "X86.Mov(dst=rax, src=Imm(4))",
                "X86.Add(dst=rax, src=r9)",
                "X86.Mul(dst=rax, src=r8)",
            ],
        )


class SimTests(unittest.TestCase):
    def test_mov_reg_imm(self):
        sim = Simulator()
        sim.load([X86.Mov(RAX, Imm(123))])
        sim.run()
        self.assertEqual(sim.reg(RAX), 123)

    def test_mov_reg_imm_overwrite(self):
        sim = Simulator()
        sim.load(
            [
                X86.Mov(RCX, Imm(123)),
                X86.Mov(RCX, Imm(456)),
            ]
        )
        sim.run()
        self.assertEqual(sim.reg(RCX), 456)

    def test_mov_reg_reg(self):
        sim = Simulator()
        sim.load(
            [
                X86.Mov(RAX, Imm(123)),
                X86.Mov(RCX, RAX),
            ]
        )
        sim.run()
        self.assertEqual(sim.reg(RAX), 123)
        self.assertEqual(sim.reg(RCX), 123)

    def test_mov_stack_imm8(self):
        off = -8
        nbytes = 1
        val = 123
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes), val)

    def test_mov_stack_negative_imm8(self):
        off = -8
        nbytes = 1
        val = -123
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes, signed=True), val)

    def test_mov_stack_imm16(self):
        off = -8
        nbytes = 2
        val = 2**8 + 1
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes), val)

    def test_mov_stack_negative_imm16(self):
        off = -8
        nbytes = 2
        val = -(2**8 + 1)
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes, signed=True), val)

    def test_mov_stack_imm32(self):
        off = -8
        nbytes = 4
        val = 2**16 + 1
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes), val)

    def test_mov_stack_negative_imm32(self):
        off = -8
        nbytes = 4
        val = -(2**16 + 1)
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes, signed=True), val)

    def test_mov_stack_imm64(self):
        off = -8
        nbytes = 8
        val = 2**32 + 1
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes), val)

    def test_mov_stack_negative_imm64(self):
        off = -8
        nbytes = 8
        val = -(2**32 + 1)
        sim = Simulator()
        sim.load(
            [
                X86.Mov(BaseDisp(RSP, off), Imm(val)),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes, signed=True), val)

    # TODO(max): Test misaligned reads and writes
    # TODO(max): Test overlapping reads and writes

    def test_mov_stack_reg64(self):
        off = -8
        nbytes = 8
        val = 123
        sim = Simulator()
        sim.load(
            [
                X86.Mov(RAX, Imm(val)),
                X86.Mov(BaseDisp(RSP, off), RAX),
            ]
        )
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) + off, nbytes), val)

    def test_add_reg_reg(self):
        sim = Simulator()
        sim.load(
            [
                X86.Mov(RAX, Imm(3)),
                X86.Mov(RCX, Imm(4)),
                X86.Add(RAX, RCX),
            ]
        )
        sim.run()
        self.assertEqual(sim.reg(RAX), 7)
        self.assertEqual(sim.reg(RCX), 4)

    def test_mul_reg_reg(self):
        sim = Simulator()
        sim.load(
            [
                X86.Mov(RAX, Imm(3)),
                X86.Mov(RCX, Imm(4)),
                X86.Mul(RAX, RCX),
            ]
        )
        sim.run()
        self.assertEqual(sim.reg(RAX), 12)
        self.assertEqual(sim.reg(RCX), 4)

    def test_push_imm(self):
        off = -8
        nbytes = 8
        val = 3
        sim = Simulator()
        sim.load(
            [
                X86.Push(Imm(val)),
            ]
        )
        rsp_before = sim.reg(RSP)
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) - off, nbytes), val)
        self.assertEqual(sim.reg(RSP), rsp_before - 8)

    def test_push_reg(self):
        off = -8
        nbytes = 8
        val = 3
        sim = Simulator()
        sim.regs[RAX.index] = val
        sim.load(
            [
                X86.Push(RAX),
            ]
        )
        rsp_before = sim.reg(RSP)
        sim.run()
        self.assertEqual(sim.memory_read(sim.reg(RSP) - off, nbytes), val)
        self.assertEqual(sim.reg(RSP), rsp_before - 8)

    def test_pop_reg(self):
        val = 3
        sim = Simulator()
        rsp_before = sim.reg(RSP)
        sim.stack_push(val)
        sim.load(
            [
                X86.Pop(RAX),
            ]
        )
        sim.run()
        self.assertEqual(sim.reg(RSP), rsp_before)
        self.assertEqual(sim.reg(RAX), val)


class BaseEndToEndTests:
    def _run(self, exp):
        raise NotImplementedError("exercise your custom compiler!")

    def test_const(self):
        sim = self._run(Const(123))
        self.assertEqual(sim.reg(RAX), 123)

    def test_add(self):
        sim = self._run(Add(Const(3), Const(4)))
        self.assertEqual(sim.reg(RAX), 7)

    def test_mul(self):
        sim = self._run(Mul(Const(3), Const(4)))
        self.assertEqual(sim.reg(RAX), 12)

    def test_mul_add(self):
        sim = self._run(
            Mul(
                Add(Const(1), Const(2)),
                Add(Const(3), Const(4)),
            )
        )
        self.assertEqual(sim.reg(RAX), 21)


class NaiveEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _run(self, exp):
        x86 = naive(exp)
        sim = Simulator()
        sim.load(x86)
        sim.run()
        assert len(sim.memory) == 256, f"memory size changed: {len(sim.memory)}"
        return sim


class DDCGEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _run(self, exp):
        gen = DDCG()
        gen.compile(exp)
        sim = Simulator()
        sim.load(gen.code)
        sim.run()
        assert len(sim.memory) == 256, f"memory size changed: {len(sim.memory)}"
        return sim


class DDCGStackEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _run(self, exp):
        gen = DDCGStack()
        gen.compile(exp)
        sim = Simulator()
        sim.load(gen.code)
        sim.run()
        assert len(sim.memory) == 256, f"memory size changed: {len(sim.memory)}"
        return sim

    def test_add_deep(self):
        # This tests pushing and popping beyond the limits of our virtual
        # stack.
        assert len(STACK_REGS) == 2
        exp = Add(Const(2), Add(Const(3), Add(Const(4), Const(5))))
        gen = DDCGStack()
        gen.compile(exp)
        sim = Simulator()
        sim.load(gen.code)
        sim.run()
        assert len(sim.memory) == 256, f"memory size changed: {len(sim.memory)}"
        self.assertEqual(sim.reg(RAX), 14)


if __name__ == "__main__":
    unittest.main()
