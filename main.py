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


class IrTest(unittest.TestCase):
    def setUp(self):
        reset_instr_counter()


class RenderTest(IrTest):
    def test_const(self):
        exp = Const(123)
        self.assertEqual(repr(exp), "v0 = 123")

    def test_add(self):
        exp = Add(Const(2), Const(3))
        self.assertEqual(repr(exp), "v2 = Add v0, v1")


if __name__ == "__main__":
    unittest.main()
