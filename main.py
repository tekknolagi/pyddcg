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
    return code


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
        self.assertEqual(self._alloc(exp), ["X86.Mov(dst=[rsp-8], src=Imm(2))"])

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
            ],
        )


if __name__ == "__main__":
    unittest.main()
