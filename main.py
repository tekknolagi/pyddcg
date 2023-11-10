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

    def operands(self):
        return ()

    def __repr__(self):
        op = self.__class__.__name__
        return (
            f"{self.var()} = {op} {', '.join(child.var() for child in self.operands())}"
        )


@ir
class Const(Instr):
    value: int

    def __repr__(self):
        return f"{self.var()} = {self.value}"


@ir
class Array(Instr):
    value: tuple[Instr]

    def operands(self):
        return self.value


@ir
class Binary(Instr):
    left: Instr
    right: Instr

    def operands(self):
        return (self.left, self.right)


@ir
class Add(Binary):
    pass


@ir
class Mul(Binary):
    pass


@ir
class Dot(Instr):
    left: Array
    right: Array


def eval_exp(exp):
    if isinstance(exp, Const):
        return exp.value
    elif isinstance(exp, Add):
        left = eval_exp(exp.left)
        right = eval_exp(exp.right)
        return left + right
    elif isinstance(exp, Mul):
        left = eval_exp(exp.left)
        right = eval_exp(exp.right)
        return left * right
    else:
        raise NotImplementedError(f"unexpected exp {exp}")


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
        sizes = [8, 16, 32, 64]
        for size in sizes:
            if bl <= size:
                return size
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


def naive_compile(exp):
    tmp = RCX
    if isinstance(exp, Const):
        return [X86.Mov(RAX, Imm(exp.value))]
    elif isinstance(exp, Add):
        right_code = naive_compile(exp.right)
        left_code = naive_compile(exp.left)
        return [
            *right_code,
            X86.Push(RAX),
            *left_code,
            X86.Pop(tmp),
            X86.Add(RAX, tmp),
        ]
    elif isinstance(exp, Mul):
        right_code = naive_compile(exp.right)
        left_code = naive_compile(exp.left)
        return [
            *right_code,
            X86.Push(RAX),
            *left_code,
            X86.Pop(tmp),
            X86.Mul(RAX, tmp),
        ]
    else:
        raise NotImplementedError(f"unexpected exp {exp}")


class Dest:
    STACK = 0
    ACCUM = 1
    NOWHERE = 2


def _plug_imm(dst, value):
    if dst == Dest.STACK:
        return [X86.Push(Imm(value))]
    elif dst == Dest.ACCUM:
        return [X86.Mov(RAX, Imm(value))]
    else:
        raise NotImplementedError


def _plug_reg(dst, reg):
    if dst == Dest.STACK:
        return [X86.Push(reg)]
    elif dst == Dest.ACCUM:
        if reg == RAX:
            return []
        return [X86.Mov(RAX, reg)]
    else:
        raise NotImplementedError


def _ddcg_compile(exp, dst):
    tmp = RCX
    if isinstance(exp, Const):
        return _plug_imm(dst, exp.value)
    elif isinstance(exp, Add):
        return [
            *_ddcg_compile(exp.left, Dest.STACK),
            *_ddcg_compile(exp.right, Dest.ACCUM),
            X86.Pop(tmp),
            X86.Add(RAX, tmp),
            *_plug_reg(dst, RAX),
        ]
    elif isinstance(exp, Mul):
        return [
            *_ddcg_compile(exp.left, Dest.STACK),
            *_ddcg_compile(exp.right, Dest.ACCUM),
            X86.Pop(tmp),
            X86.Mul(RAX, tmp),
            *_plug_reg(dst, RAX),
        ]
    else:
        raise NotImplementedError(exp)


def ddcg_compile(code):
    return _ddcg_compile(code, Dest.ACCUM)


REGS = [R8, R9, R10, R11]


class DelayedDDCG:
    def __init__(self):
        self.code = []
        self.free_registers = {reg: True for reg in REGS}

    def _allocate_register(self):
        for reg in REGS:
            if self.free_registers[reg]:
                self.free_registers[reg] = False
                return reg
        raise Exception("could not allocate register")

    def _free_register(self, reg):
        self.free_registers[reg] = True

    def compile(self, exp):
        result = self._compile(exp)
        self.code.append(X86.Mov(RAX, result))

    def _compile(self, exp):
        tmp = RCX
        if isinstance(exp, Const):
            result = self._allocate_register()
            self.code.append(X86.Mov(result, Imm(exp.value)))
            return result
        elif isinstance(exp, Add):
            lhs = self._compile(exp.left)
            rhs = self._compile(exp.right)
            self.code.append(X86.Add(lhs, rhs))
            self._free_register(rhs)
            return lhs
        elif isinstance(exp, Mul):
            lhs = self._compile(exp.left)
            rhs = self._compile(exp.right)
            self.code.append(X86.Mul(lhs, rhs))
            self._free_register(rhs)
            return lhs
        else:
            raise NotImplementedError(exp)


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

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(repr(exp), "v2 = Mul v0, v1")


class EvalTests(unittest.TestCase):
    def test_const(self):
        exp = Const(2)
        self.assertEqual(eval_exp(exp), 2)

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(eval_exp(exp), 5)

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(eval_exp(exp), 6)

    def test_mul_add(self):
        exp = Mul(
            Add(Const(1), Const(2)),
            Add(Const(3), Const(4)),
        )
        self.assertEqual(eval_exp(exp), 21)


class NaiveCompilerTests(unittest.TestCase):
    def _alloc(self, exp):
        x86 = naive_compile(exp)
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

    def test_add_deep(self):
        exp = Add(Const(2), Add(Const(3), Add(Const(4), Const(5))))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=rax, src=Imm(5))",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(4))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(3))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Push(src=rax)",
                "X86.Mov(dst=rax, src=Imm(2))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
            ],
        )


class DDCGTests(unittest.TestCase):
    def _alloc(self, exp):
        x86 = ddcg_compile(exp)
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

    def test_add_deep(self):
        exp = Add(Const(2), Add(Const(3), Add(Const(4), Const(5))))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Push(src=Imm(2))",
                "X86.Push(src=Imm(3))",
                "X86.Push(src=Imm(4))",
                "X86.Mov(dst=rax, src=Imm(5))",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
                "X86.Pop(dst=rcx)",
                "X86.Add(dst=rax, src=rcx)",
            ],
        )


class DelayedDDCGTests(unittest.TestCase):
    def _alloc(self, exp):
        gen = DelayedDDCG()
        gen.compile(exp)
        x86 = gen.code
        return [str(op) for op in x86]

    def test_const(self):
        exp = Const(2)
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=rax, src=r8)",
            ],
        )

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=r9, src=Imm(3))",
                "X86.Add(dst=r8, src=r9)",
                "X86.Mov(dst=rax, src=r8)",
            ],
        )

    def test_mul(self):
        exp = Mul(Const(2), Const(3))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=r9, src=Imm(3))",
                "X86.Mul(dst=r8, src=r9)",
                "X86.Mov(dst=rax, src=r8)",
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
                "X86.Mov(dst=r9, src=Imm(2))",
                "X86.Add(dst=r8, src=r9)",
                "X86.Mov(dst=r9, src=Imm(3))",
                "X86.Mov(dst=r10, src=Imm(4))",
                "X86.Add(dst=r9, src=r10)",
                "X86.Mul(dst=r8, src=r9)",
                "X86.Mov(dst=rax, src=r8)",
            ],
        )

    def test_add_deep(self):
        exp = Add(Const(2), Add(Const(3), Add(Const(4), Const(5))))
        self.assertEqual(
            self._alloc(exp),
            [
                "X86.Mov(dst=r8, src=Imm(2))",
                "X86.Mov(dst=r9, src=Imm(3))",
                "X86.Mov(dst=r10, src=Imm(4))",
                "X86.Mov(dst=r11, src=Imm(5))",
                "X86.Add(dst=r10, src=r11)",
                "X86.Add(dst=r9, src=r10)",
                "X86.Add(dst=r8, src=r9)",
                "X86.Mov(dst=rax, src=r8)",
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
        x86 = self._compile(exp)
        sim = Simulator()
        sim.load(x86)
        sim.run()
        self.assertEqual(sim.reg(RAX), eval_exp(exp))
        assert len(sim.memory) == 256, f"memory size changed: {len(sim.memory)}"
        return sim

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


class NaiveCompilerEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _compile(self, exp):
        return naive_compile(exp)


class DDCGEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _compile(self, exp):
        return ddcg_compile(exp)


class DelayedDDCGEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _compile(self, exp):
        gen = DelayedDDCG()
        gen.compile(exp)
        return gen.code


class DDCGStackEndToEndTests(BaseEndToEndTests, unittest.TestCase):
    def _compile(self, exp):
        gen = DDCGStack()
        gen.compile(exp)
        return gen.code

    def test_add_deep(self):
        # This tests pushing and popping beyond the limits of our virtual
        # stack.
        assert len(STACK_REGS) == 2
        exp = Add(Const(2), Add(Const(3), Add(Const(4), Const(5))))
        x86 = self._compile(exp)
        sim = Simulator()
        sim.load(x86)
        sim.run()
        self.assertEqual(sim.reg(RAX), eval_exp(exp))
        assert len(sim.memory) == 256, f"memory size changed: {len(sim.memory)}"
        self.assertEqual(sim.reg(RAX), 14)


if __name__ == "__main__":
    unittest.main()
