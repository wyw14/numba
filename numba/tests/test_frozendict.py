"""
Tests for the read-only typed dictionary snapshot:
``numba.typed.FrozenDict`` and ``numba.typed.freeze``.
"""

import numpy as np

from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.typed import Dict, FrozenDict, freeze
from numba.tests.support import TestCase, MemoryLeakMixin, unittest


def _make_int_float():
    d = Dict.empty(types.int64, types.float64)
    d[3] = 3.0
    d[1] = 1.0
    d[2] = 2.0
    d[5] = 5.0
    return d


class FrozenDictTestMixin(MemoryLeakMixin):

    def _check_contents(self, fd):
        self.assertIs(type(fd), FrozenDict)
        self.assertEqual(len(fd), 4)
        self.assertEqual(list(fd), [3, 1, 2, 5])
        self.assertEqual(list(fd.keys()), [3, 1, 2, 5])
        self.assertEqual(list(fd.values()), [3.0, 1.0, 2.0, 5.0])
        self.assertEqual(list(fd.items()),
                         [(3, 3.0), (1, 1.0), (2, 2.0), (5, 5.0)])
        self.assertEqual(fd[1], 1.0)
        self.assertTrue(5 in fd)
        self.assertFalse(99 in fd)
        self.assertEqual(fd.get(2), 2.0)
        self.assertEqual(fd.get(99, -7.0), -7.0)


class TestPythonFrozenDict(FrozenDictTestMixin, TestCase):

    def test_empty(self):
        # exceptional exits from compiled code do not run NRT cleanups
        self.disable_leak_check()
        d = Dict.empty(types.int64, types.float64)
        fd = freeze(d)
        self._check_empty(fd)

    def test_empty_freeze_method(self):
        self.disable_leak_check()
        d = Dict.empty(types.int64, types.float64)
        fd = d.freeze()
        self._check_empty(fd)

    def _check_empty(self, fd):
        self.assertIs(type(fd), FrozenDict)
        self.assertEqual(len(fd), 0)
        self.assertEqual(list(fd), [])
        self.assertEqual(list(fd.items()), [])
        self.assertFalse(0 in fd)
        self.assertEqual(fd.get(0, -1.0), -1.0)
        with self.assertRaises(KeyError):
            fd[0]

    def test_basic_and_order(self):
        fd = freeze(_make_int_float())
        self._check_contents(fd)
        self.assertEqual(str(fd), "{3: 3.0, 1: 1.0, 2: 2.0, 5: 5.0}")
        self.assertIn("FrozenDictType", repr(fd))

    def test_string_keys(self):
        d = Dict.empty(types.unicode_type, types.int64)
        d["b"] = 2
        d["a"] = 1
        d["c"] = 3
        fd = freeze(d)
        self.assertEqual(list(fd), ["b", "a", "c"])
        self.assertEqual(fd["a"], 1)
        self.assertEqual(fd.get("missing", -1), -1)
        self.assertTrue("c" in fd)

    def test_tuples(self):
        kty = types.UniTuple(types.int64, 2)
        vty = types.Tuple([types.float64, types.unicode_type])
        d = Dict.empty(kty, vty)
        d[(1, 2)] = (1.5, "x")
        d[(3, 4)] = (2.5, "y")
        fd = freeze(d)
        self.assertEqual(list(fd), [(1, 2), (3, 4)])
        self.assertEqual(fd[(3, 4)], (2.5, "y"))
        self.assertTrue((1, 2) in fd)
        self.assertEqual(
            fd.get((9, 9), (0.0, "z")), (0.0, "z"))

    def test_isolation(self):
        d = _make_int_float()
        fd = freeze(d)

        d[1] = 99.0
        d[6] = 6.0
        del d[2]
        d.clear()

        self.assertEqual(len(fd), 4)
        self.assertEqual(fd[1], 1.0)
        self.assertEqual(fd[2], 2.0)
        self.assertFalse(6 in fd)
        self.assertEqual(list(fd), [3, 1, 2, 5])

    def test_isolation_tuple_values(self):
        kty = types.int64
        vty = types.UniTuple(types.int64, 2)
        d = Dict.empty(kty, vty)
        d[1] = (1, 10)
        fd = freeze(d)
        d[1] = (2, 20)
        self.assertEqual(fd[1], (1, 10))

    def test_freeze_of_frozen(self):
        fd = freeze(_make_int_float())
        self.assertIs(freeze(fd), fd)

    def test_frozen_copy_is_mutable(self):
        fd = freeze(_make_int_float())
        d = fd.copy()
        self.assertIs(type(d), Dict)
        d[9] = 9.0
        self.assertEqual(d[9], 9.0)
        self.assertFalse(9 in fd)

    def test_writes_raise_type_error(self):
        fd = freeze(_make_int_float())
        other = Dict.empty(types.int64, types.float64)
        other[7] = 7.0
        mutators = (
            lambda: fd.__setitem__(9, 9.0),
            lambda: fd.__delitem__(1),
            lambda: fd.pop(1),
            lambda: fd.pop(1, 0.0),
            lambda: fd.popitem(),
            lambda: fd.clear(),
            lambda: fd.update(other),
            lambda: fd.setdefault(9, 9.0),
        )
        for mut in mutators:
            with self.assertRaises(TypeError) as raises:
                mut()
            self.assertIn("read-only", str(raises.exception))
        self.assertEqual(len(fd), 4)

    def test_direct_construction_rejected(self):
        with self.assertRaises(TypeError):
            FrozenDict()

    def test_freeze_untyped_rejected(self):
        d = Dict()
        with self.assertRaises(TypeError) as raises:
            freeze(d)
        self.assertIn("untyped", str(raises.exception))

    def test_freeze_bad_argument(self):
        with self.assertRaises(TypeError):
            freeze({1: 2.0})

    def test_array_value_rejected(self):
        vty = types.Array(types.float64, 1, "C")
        d = Dict.empty(types.int64, vty)
        with self.assertRaises(TypeError) as raises:
            freeze(d)
        self.assertIn("freeze", str(raises.exception))

    def test_array_inside_tuple_value_rejected(self):
        vty = types.UniTuple(types.Array(types.float64, 1, "C"), 1)
        d = Dict.empty(types.int64, vty)
        with self.assertRaises(TypeError):
            freeze(d)

    def test_array_key_rejected(self):
        kty = types.Array(types.float64, 1, "C")
        d = Dict.empty(kty, types.int64)
        with self.assertRaises(TypeError):
            freeze(d)


@njit
def _read_all(d):
    keys = list(d.keys())
    values = list(d.values())
    pairs = [(k, v) for k, v in d.items()]
    return (d[1], d.get(99, -7.0), 99 in d, 3 in d, len(d),
            keys, values, pairs)


@njit
def _return_frozen(d):
    return d


@njit
def _freeze_inside(d):
    return freeze(d)


@njit
def _freeze_method_inside(d):
    return d.freeze()


@njit
def _collect_keys(d):
    out = []
    for k in d:
        out.append(k)
    return out


class TestCompiledFrozenDict(FrozenDictTestMixin, TestCase):

    def test_njit_reads(self):
        fd = freeze(_make_int_float())
        res = _read_all(fd)
        got = (res[0], res[1], res[2], res[3], res[4],
               list(res[5]), list(res[6]),
               [(p[0], p[1]) for p in res[7]])
        expected = (1.0, -7.0, False, True, 4,
                    [3, 1, 2, 5],
                    [3.0, 1.0, 2.0, 5.0],
                    [(3, 3.0), (1, 1.0), (2, 2.0), (5, 5.0)])
        self.assertEqual(got, expected)

    def test_njit_empty(self):
        fd = freeze(Dict.empty(types.int64, types.float64))

        @njit
        def check(d):
            return len(d), list(d.keys()), d.get(1, -1.0), 1 in d

        self.assertEqual(check(fd), (0, [], -1.0, False))

    def test_njit_missing_key(self):
        # exceptional exits from compiled code do not run NRT cleanups
        self.disable_leak_check()
        fd = freeze(_make_int_float())

        @njit
        def bad(d):
            return d[99]

        with self.assertRaises(KeyError):
            bad(fd)

    def test_iteration_order_njit(self):
        fd = freeze(_make_int_float())
        self.assertEqual(list(_collect_keys(fd)), [3, 1, 2, 5])

    def test_njit_takes_and_returns_frozen(self):
        fd = freeze(_make_int_float())
        ret = _return_frozen(fd)
        self.assertIs(type(ret), FrozenDict)
        self._check_contents(ret)

    def test_freeze_inside_njit(self):
        d = _make_int_float()
        fd = _freeze_inside(d)
        self.assertIs(type(fd), FrozenDict)
        self._check_contents(fd)

        d[9] = 99.0
        d.clear()
        self._check_contents(fd)

    def test_freeze_method_inside_njit(self):
        d = _make_int_float()
        fd = _freeze_method_inside(d)
        self.assertIs(type(fd), FrozenDict)
        self._check_contents(fd)

    def test_tuple_in_njit(self):
        kty = types.UniTuple(types.int64, 2)
        vty = types.Tuple([types.float64, types.unicode_type])
        d = Dict.empty(kty, vty)
        d[(1, 2)] = (1.5, "x")
        d[(3, 4)] = (2.5, "y")
        fd = freeze(d)

        @njit
        def check(f):
            return f[(1, 2)], (0, 0) in f, len(f)

        val, present, n = check(fd)
        self.assertEqual((val, present, n), ((1.5, "x"), False, 2))

    def test_type_distinction(self):
        fd = freeze(_make_int_float())
        ty = fd._numba_type_
        self.assertIs(type(ty), types.FrozenDictType)
        self.assertEqual(
            (ty.key_type, ty.value_type),
            (types.int64, types.float64))

        @njit
        def as_is(d):
            return d

        as_is(fd)
        sig = as_is.nopython_signatures[0]
        self.assertIs(type(sig.return_type), types.FrozenDictType)
        self.assertIs(type(sig.args[0]), types.FrozenDictType)

    def test_branch_unify_keeps_frozen_type(self):
        d = _make_int_float()

        @njit
        def branch(x, flag):
            a = x.freeze()
            b = x.freeze()
            return a if flag else b

        fd = branch(d, True)
        self.assertIs(type(fd), FrozenDict)
        sig = branch.nopython_signatures[0]
        self.assertIs(type(sig.return_type), types.FrozenDictType)

    def test_frozen_and_mutable_do_not_unify(self):
        self.disable_leak_check()
        d = _make_int_float()

        @njit
        def mixed(x, flag):
            if flag > 0:
                z = x.freeze()
            else:
                z = x.copy()
            return z

        with self.assertRaises(TypingError):
            mixed(d, 1)
        with self.assertRaises(TypingError):
            mixed(d, 0)

    def test_explicit_signatures_reject_wrong_kind(self):
        self.disable_leak_check()
        fdt = types.FrozenDictType(types.int64, types.float64)
        mt = types.DictType(types.int64, types.float64)

        @njit(types.int64(fdt))
        def take_frozen(d):
            return len(d)

        @njit(types.int64(mt))
        def take_mutable(d):
            d[9] = 9.0
            return len(d)

        fd = freeze(_make_int_float())
        d = _make_int_float()
        self.assertEqual(take_frozen(fd), 4)
        with self.assertRaises((TypeError, TypingError)):
            take_frozen(d)
        with self.assertRaises((TypeError, TypingError)):
            take_mutable(fd)

    def test_freeze_array_value_rejected_in_njit(self):
        self.disable_leak_check()
        vty = types.Array(types.float64, 1, "C")
        d = Dict.empty(types.int64, vty)

        @njit
        def do_freeze(x):
            return freeze(x)

        with self.assertRaises(TypingError) as raises:
            do_freeze(d)
        self.assertIn("freeze", str(raises.exception))


def _assert_frozen_typing_error(self, pyfunc):
    with self.assertRaises(TypingError) as raises:
        njit(pyfunc)()
    msg = str(raises.exception).lower()
    self.assertIn("frozen", msg)
    self.assertIn("read-only", msg)


class TestFrozenWritesRejectedAtCompile(MemoryLeakMixin, TestCase):

    def setUp(self):
        # typing failures do not run NRT cleanups
        self.disable_leak_check()
        self.fd = freeze(_make_int_float())
        self.other = _make_int_float()

    def test_setitem(self):
        fd = self.fd

        def impl():
            fd[9] = 9.0
        _assert_frozen_typing_error(self, impl)

    def test_delitem(self):
        fd = self.fd

        def impl():
            del fd[1]
        _assert_frozen_typing_error(self, impl)

    def test_pop(self):
        fd = self.fd

        def impl():
            return fd.pop(1)
        _assert_frozen_typing_error(self, impl)

    def test_pop_default(self):
        fd = self.fd

        def impl():
            return fd.pop(1, 0.0)
        _assert_frozen_typing_error(self, impl)

    def test_popitem(self):
        fd = self.fd

        def impl():
            return fd.popitem()
        _assert_frozen_typing_error(self, impl)

    def test_clear(self):
        fd = self.fd

        def impl():
            fd.clear()
        _assert_frozen_typing_error(self, impl)

    def test_setdefault(self):
        fd = self.fd

        def impl():
            return fd.setdefault(9, 9.0)
        _assert_frozen_typing_error(self, impl)

    def test_update(self):
        fd = self.fd
        other = self.other

        def impl():
            fd.update(other)
        _assert_frozen_typing_error(self, impl)

    def test_mutable_constructor_rejects_frozen(self):
        fd = self.fd

        def impl(f):
            return Dict(f)

        with self.assertRaises(TypingError):
            njit(impl)(fd)


if __name__ == '__main__':
    unittest.main()
