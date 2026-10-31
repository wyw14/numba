import bisect as pybisect

import numpy as np

from numba import jit, typed, types
from numba.tests.support import TestCase, MemoryLeakMixin


def bisect_left(a, x):
    return pybisect.bisect_left(a, x)


def bisect_right(a, x):
    return pybisect.bisect_right(a, x)


def bisect_left_lo(a, x, lo):
    return pybisect.bisect_left(a, x, lo)


def bisect_right_lo(a, x, lo):
    return pybisect.bisect_right(a, x, lo)


def bisect_left_bounds(a, x, lo, hi):
    return pybisect.bisect_left(a, x, lo, hi)


def bisect_right_bounds(a, x, lo, hi):
    return pybisect.bisect_right(a, x, lo, hi)


def bisect_left_kwargs(a, x, lo, hi):
    return pybisect.bisect_left(a, x, lo=lo, hi=hi)


def bisect_right_kwargs(a, x, lo, hi):
    return pybisect.bisect_right(a, x, lo=lo, hi=hi)


def bisect_left_explicit_none_hi(a, x, lo):
    return pybisect.bisect_left(a, x, lo, None)


def bisect_left_key(a, x, key):
    return pybisect.bisect_left(a, x, key=key)


class TestBisect(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(TestBisect, self).setUp()
        self.rnd = np.random.RandomState(42)
        self.left_cfunc = jit(nopython=True)(bisect_left)
        self.right_cfunc = jit(nopython=True)(bisect_right)
        self.left_lo_cfunc = jit(nopython=True)(bisect_left_lo)
        self.right_lo_cfunc = jit(nopython=True)(bisect_right_lo)
        self.bounds_left_cfunc = jit(nopython=True)(bisect_left_bounds)
        self.bounds_right_cfunc = jit(nopython=True)(bisect_right_bounds)
        self.kw_left_cfunc = jit(nopython=True)(bisect_left_kwargs)
        self.kw_right_cfunc = jit(nopython=True)(bisect_right_kwargs)
        self.none_hi_cfunc = jit(nopython=True)(bisect_left_explicit_none_hi)

    def _assert_matches_python(self, nb_container, py_container, values):
        n = len(py_container)
        bounds = [(0, n), (0, n - 1), (1, n), (0, 3), (2, 2),
                  (max(n - 1, 0), n)]
        for x in values:
            self.assertEqual(self.left_cfunc(nb_container, x),
                             pybisect.bisect_left(py_container, x))
            self.assertEqual(self.right_cfunc(nb_container, x),
                             pybisect.bisect_right(py_container, x))
            for lo, hi in bounds:
                if lo > n or hi > n or hi < 0:
                    continue
                self.assertEqual(
                    self.bounds_left_cfunc(nb_container, x, lo, hi),
                    pybisect.bisect_left(py_container, x, lo, hi))
                self.assertEqual(
                    self.bounds_right_cfunc(nb_container, x, lo, hi),
                    pybisect.bisect_right(py_container, x, lo, hi))
                self.assertEqual(
                    self.kw_left_cfunc(nb_container, x, lo, hi),
                    pybisect.bisect_left(py_container, x, lo, hi))
                self.assertEqual(
                    self.kw_right_cfunc(nb_container, x, lo, hi),
                    pybisect.bisect_right(py_container, x, lo, hi))

    def test_empty_container(self):
        empty_list = typed.List.empty_list(types.int64)
        for container in (empty_list,
                          np.empty(0, dtype=np.int64),
                          np.empty(0, dtype=np.float64)):
            self.assertEqual(self.left_cfunc(container, 0), 0)
            self.assertEqual(self.right_cfunc(container, 0), 0)
            self.assertEqual(
                self.bounds_left_cfunc(container, 0, 0, 0), 0)
            self.assertEqual(
                self.bounds_right_cfunc(container, 0, 0, 0), 0)

    def test_typed_list_int(self):
        data = [1, 1, 2, 3, 3, 3, 5, 8, 8, 10, 10, 10, 10]
        container = typed.List(data)
        values = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
        self._assert_matches_python(container, data, values)
        self.assertEqual(list(container), data)

    def test_typed_list_float(self):
        data = sorted(self.rnd.choice(np.arange(10, dtype=np.float64),
                                      size=200, replace=True).tolist())
        container = typed.List(data)
        values = [-1.0, 0.0, 0.5, 3.0, 5.5, 9.0, 10.0,
                  np.inf, -np.inf, 3, 7]
        self._assert_matches_python(container, data, values)
        self.assertEqual(list(container), data)

    def test_typed_list_unicode(self):
        data = ["apple", "apple", "banana", "fig", "fig", "fig",
                "lime", "lime", "pear"]
        container = typed.List(data)
        values = ["aardvark", "apple", "banana", "cherry", "fig",
                  "lime", "mango", "pear", "zebra"]
        self._assert_matches_python(container, data, values)
        self.assertEqual(list(container), data)

    def test_numpy_array_int(self):
        data = np.sort(self.rnd.randint(-20, 20, size=500))
        values = [-21, -20, -5, 0, 1, 19, 20]
        self._assert_matches_python(data, data, values)

    def test_numpy_array_float(self):
        data = np.sort(self.rnd.standard_normal(300))
        values = np.concatenate((data[::37], [-1e9, 0.0, 1e9])).tolist()
        self._assert_matches_python(data, data, values)

    def test_numpy_array_noncontiguous(self):
        base = np.arange(40, dtype=np.int64)
        view = base[::2]
        self.assertFalse(view.flags.c_contiguous)
        values = [-1, 0, 1, 10, 19, 38, 39, 40]
        self._assert_matches_python(view, view, values)

    def test_numpy_array_reversed_strided(self):
        base = -np.arange(40, dtype=np.int64)
        view = base[::-2]
        values = [-39, -38, -20, -1, 0, 1]
        self._assert_matches_python(view, view, values)

    def test_different_lengths_same_dispatcher(self):
        for n in [0, 1, 2, 7, 64, 1000]:
            data = np.sort(self.rnd.randint(0, 50, size=n))
            for x in (-1, 0, 25, 49, 50):
                self.assertEqual(
                    self.left_cfunc(data, x),
                    pybisect.bisect_left(data, x))
                self.assertEqual(
                    self.right_cfunc(data, x),
                    pybisect.bisect_right(data, x))
            py_list = data.tolist()
            lst = (typed.List(py_list) if n
                   else typed.List.empty_list(types.int64))
            self.assertEqual(
                self.left_cfunc(lst, 25),
                pybisect.bisect_left(py_list, 25))

    def test_duplicates_boundaries(self):
        for data, x, left, right in [
            ([1, 2, 2, 2, 3], 2, 1, 4),
            ([1, 1, 1], 1, 0, 3),
            ([1, 1, 1], 0, 0, 0),
            ([1, 1, 1], 2, 3, 3),
            ([1, 2, 3], 1, 0, 1),
            ([1, 2, 3], 3, 2, 3),
        ]:
            container = typed.List(data)
            self.assertEqual(self.left_cfunc(container, x), left)
            self.assertEqual(self.right_cfunc(container, x), right)

    def test_lo_only_and_empty_interval(self):
        data = [0, 1, 2, 3, 4, 5]
        container = typed.List(data)
        for x in (1, 4, 6):
            self.assertEqual(self.left_lo_cfunc(container, x, 2),
                             pybisect.bisect_left(data, x, 2))
            self.assertEqual(self.right_lo_cfunc(container, x, 2),
                             pybisect.bisect_right(data, x, 2))
            self.assertEqual(self.none_hi_cfunc(container, x, 2),
                             pybisect.bisect_left(data, x, 2, None))
        self.assertEqual(
            self.bounds_left_cfunc(container, 3, 4, 4), 4)
        self.assertEqual(
            self.bounds_right_cfunc(container, 3, 4, 4), 4)

    def test_container_not_modified(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        view = arr[::2]
        self.left_cfunc(view, 3)
        self.right_cfunc(view, 3)
        self.bounds_left_cfunc(view, 3, 1, 2)
        self.assertPreciseEqual(
            arr, np.array([1, 2, 3, 4, 5], dtype=np.float64))
        lst = typed.List(["a", "b", "c"])
        self.left_cfunc(lst, "b")
        self.right_cfunc(lst, "b")
        self.assertEqual(list(lst), ["a", "b", "c"])

    def test_invalid_bounds(self):
        # Exceptions leak references
        self.disable_leak_check()
        container = typed.List([1, 2, 3, 4])
        arr = np.arange(4, dtype=np.int64)

        with self.assertRaises(ValueError) as cm:
            self.left_lo_cfunc(container, 2, -1)
        self.assertIn("lo must be non-negative", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.bounds_left_cfunc(container, 2, 5, 4)
        self.assertIn("len(a)", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.bounds_right_cfunc(container, 2, 0, 5)
        self.assertIn("len(a)", str(cm.exception))

        with self.assertRaises(ValueError):
            self.bounds_left_cfunc(container, 2, 0, -1)

        with self.assertRaises(ValueError) as cm:
            self.left_lo_cfunc(arr, 2, -1)
        self.assertIn("lo must be non-negative", str(cm.exception))

    def test_unsupported_container(self):
        self.disable_leak_check()

        @jit(nopython=True)
        def use_tuple(a, x):
            return pybisect.bisect_left(a, x)

        @jit(nopython=True)
        def use_2d(a, x):
            return pybisect.bisect_left(a, x)

        with self.assertTypingError() as e:
            use_tuple((1, 2, 3), 2)
        self.assertIn("numba.typed.List", str(e.exception))

        with self.assertTypingError() as e:
            use_2d(np.zeros((2, 2), dtype=np.int64), 0)
        self.assertIn("one-dimensional", str(e.exception))

        with self.assertTypingError() as e:
            use_tuple({1, 2, 3}, 2)
        self.assertIn("numba.typed.List", str(e.exception))

    def test_incomparable_elements(self):
        self.disable_leak_check()

        @jit(nopython=True)
        def f(a, x):
            return pybisect.bisect_left(a, x)

        with self.assertTypingError() as e:
            f(typed.List([1 + 1j, 2 + 2j]), 1 + 1j)
        self.assertIn("'<' not supported", str(e.exception))

        with self.assertTypingError() as e:
            f(np.array([1.0, 2.0]), 1 + 0j)
        self.assertIn("'<' not supported", str(e.exception))

        with self.assertTypingError():
            f(typed.List(["a", "b"]), 1)

        with self.assertTypingError():
            f(typed.List([1, 2]), "a")

        with self.assertTypingError() as e:
            f(np.array(["a", "b"]), "a")
        self.assertIn("numeric", str(e.exception))

    def test_invalid_bound_type(self):
        self.disable_leak_check()

        @jit(nopython=True)
        def f(a, x, lo):
            return pybisect.bisect_left(a, x, lo)

        with self.assertTypingError() as e:
            f(typed.List([1, 2, 3]), 2, 1.5)
        self.assertIn("'lo'", str(e.exception))

    def test_key_unsupported(self):
        self.disable_leak_check()
        cfunc = jit(nopython=True)(bisect_left_key)
        identity = jit(nopython=True)(lambda v: v)
        with self.assertTypingError() as e:
            cfunc(typed.List([1, 2, 3]), 2, identity)
        self.assertIn("'key'", str(e.exception))

    def test_explicit_key_none(self):
        @jit(nopython=True)
        def f(a, x):
            return pybisect.bisect_left(a, x, 0, None, key=None)

        self.assertEqual(f(typed.List([1, 2, 3]), 2), 1)
