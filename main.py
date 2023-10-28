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


def regalloc(ops):
    stack = []
    code = []

    def stack_at(idx):
        base = 8
        return BaseDisp(RSP, -(idx * 8 + base))

    for op in ops:
        if isinstance(op, Const):
            assert op not in stack
            idx = len(stack)
            stack.append(op)
            code.append(X86.Mov(stack_at(idx), Imm(op.value)))
        elif isinstance(op, (Add, Mul)):
            assert op not in stack
            left = stack.index(op.left)
            right = stack.index(op.right)
            idx = len(stack)
            stack.append(op)
            code.append(X86.Mov(RAX, stack_at(left)))
            code.append(X86.Mov(RCX, stack_at(right)))
            opcode = {Add: X86.Add, Mul: X86.Mul}[type(op)]
            code.append(opcode(RAX, RCX))
            code.append(X86.Mov(stack_at(idx), RAX))
        else:
            raise NotImplementedError(op)
    # Move the last result into RAX
    code.append(X86.Mov(RAX, stack_at(stack.index(ops[-1]))))
    return code


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
            raise NotImplementedError
            # self.emit(X86.Push(reg))
        elif dst == Dest.ACCUM:
            if reg == RAX:
                pass
            else:
                raise NotImplementedError
                # self.emit(X86.Mov(RAX, reg))
        else:
            raise NotImplementedError


class Simulator:
    def __init__(self):
        self.regs = [0] * 16
        # TODO(max): Represent memory as contiguous array with RSP pointing at
        # middle or something
        self.stack = bytearray([0] * 256)
        self.code = []
        self.regs[RSP.index] = -8

    def load(self, code):
        self.code = code

    def run(self):
        for op in self.code:
            self.run_one(op)

    def reg(self, reg):
        return self.regs[reg.index]

    def stack_write_imm(self, idx, imm):
        assert idx < 0, "Cannot read before stack frame"
        # The stack is backwards/upside-down...
        idx = -idx
        bs = imm.as_bytes()
        assert imm.size() == len(bs)
        self.stack[idx : idx + imm.size()] = bs

    def stack_write(self, idx, value, nbytes):
        assert idx < 0, "Cannot read before stack frame"
        # The stack is backwards/upside-down...
        idx = -idx
        bs = value.to_bytes(nbytes, byteorder="little", signed=True)
        self.stack[idx : idx + nbytes] = bs

    def stack_read(self, idx, nbytes, signed=False):
        assert idx < 0, "Cannot read before stack frame"
        # The stack is backwards/upside-down...
        idx = -idx
        return int.from_bytes(
            self.stack[idx : idx + nbytes], byteorder="little", signed=signed
        )

    def stack_push(self, value):
        rsp = self.reg(RSP)
        self.stack_write(rsp, value, nbytes=8)
        self.regs[RSP.index] -= 8

    def stack_pop(self, value):
        self.regs[RSP.index] += 8
        rsp = self.reg(RSP)
        return self.stack_read(rsp, nbytes=8)

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
                    # TODO(max): Get read size from register size
                    self.regs[op.dst.index] = self.stack_read(op.src.disp, nbytes=8)
                else:
                    assert isinstance(op.src, Imm), "non-imm src unsupported"
            elif isinstance(op.dst, Mem):
                assert isinstance(op.dst, BaseDisp), "more complex memory not supported"
                assert op.dst.base == RSP, "non-stack memory unsupported"
                if isinstance(op.src, Imm):
                    self.stack_write_imm(op.dst.disp, op.src)
                elif isinstance(op.src, Reg):
                    self.stack_write(op.dst.disp, self.reg(op.src), nbytes=8)
                else:
                    raise NotImplementedError("non-imm src")
            else:
                assert isinstance(op.dst, Reg), "non-reg dst unsupported"
        elif isinstance(op, X86.Add):
            if isinstance(op.dst, Reg) and isinstance(op.src, Reg):
                self.regs[op.dst.index] = (
                    self.reg(op.dst) + self.reg(op.src)
                )
            else:
                raise NotImplementedError("only add reg, reg is supported")
        elif isinstance(op, X86.Mul):
            if isinstance(op.dst, Reg) and isinstance(op.src, Reg):
                self.regs[op.dst.index] = (
                    self.reg(op.dst) * self.reg(op.src)
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
                value = self.stack_pop(op.dst)
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


class RegAllocTests(IrTests):
    def _alloc(self, exp):
        ops = topo(exp)
        x86 = regalloc(ops)
        return [str(op) for op in x86]

    def test_const(self):
        exp = Const(2)
        self.assertEqual(
            self._alloc(exp),
            ["X86.Mov(dst=[rsp-8], src=Imm(2))", "X86.Mov(dst=rax, src=[rsp-8])"],
        )

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=[rsp-8], src=Imm(2))",
                "X86.Mov(dst=[rsp-16], src=Imm(3))",
                "X86.Mov(dst=rax, src=[rsp-8])",
                "X86.Mov(dst=rcx, src=[rsp-16])",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Mov(dst=[rsp-24], src=rax)",
                "X86.Mov(dst=rax, src=[rsp-24])",
            ],
        )

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=[rsp-8], src=Imm(2))",
                "X86.Mov(dst=[rsp-16], src=Imm(3))",
                "X86.Mov(dst=rax, src=[rsp-8])",
                "X86.Mov(dst=rcx, src=[rsp-16])",
                "X86.Mul(dst=rax, src=rcx)",
                "X86.Mov(dst=[rsp-24], src=rax)",
                "X86.Mov(dst=rax, src=[rsp-24])",
            ],
        )


class DDCGTests(IrTests):
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
        self.assertEqual(sim.stack_read(off, nbytes), val)

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
        self.assertEqual(sim.stack_read(off, nbytes, signed=True), val)

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
        self.assertEqual(sim.stack_read(off, nbytes), val)

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
        self.assertEqual(sim.stack_read(off, nbytes, signed=True), val)

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
        self.assertEqual(sim.stack_read(off, nbytes), val)

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
        self.assertEqual(sim.stack_read(off, nbytes, signed=True), val)

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
        self.assertEqual(sim.stack_read(off, nbytes), val)

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
        self.assertEqual(sim.stack_read(off, nbytes, signed=True), val)

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
        self.assertEqual(sim.stack_read(off, nbytes), val)

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

    def test_rsp_points_to_beginning_of_frame(self):
        sim = Simulator()
        self.assertEqual(sim.reg(RSP), -8)

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
        sim.run()
        self.assertEqual(sim.stack_read(off, nbytes), val)
        self.assertEqual(sim.reg(RSP), -16)

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
        sim.run()
        self.assertEqual(sim.stack_read(off, nbytes), val)
        self.assertEqual(sim.reg(RSP), -16)

    def test_pop_reg(self):
        off = -8
        nbytes = 8
        val = 3
        sim = Simulator()
        sim.load(
            [
                X86.Push(Imm(val)),
                X86.Pop(RAX),
            ]
        )
        sim.run()
        self.assertEqual(sim.stack_read(off, nbytes), val)
        self.assertEqual(sim.reg(RSP), -8)
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


class BaselineEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _run(self, exp):
        ops = topo(exp)
        x86 = regalloc(ops)
        sim = Simulator()
        sim.load(x86)
        sim.run()
        assert len(sim.stack) == 256, f"stack size changed: {len(sim.stack)}"
        return sim


class DDCGEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _run(self, exp):
        gen = DDCG()
        gen.compile(exp)
        sim = Simulator()
        sim.load(gen.code)
        sim.run()
        assert len(sim.stack) == 256, f"stack size changed: {len(sim.stack)}"
        return sim


if __name__ == "__main__":
    unittest.main()
