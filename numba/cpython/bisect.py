import bisect as _bisect

from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.core.extending import overload


def _is_ordered_number(t):
    return isinstance(t, (types.Integer, types.Float, types.Boolean))


def _container_dtype(a):
    if isinstance(a, (types.List, types.ListType)):
        return a.dtype, True
    if isinstance(a, types.Array):
        if a.ndim != 1:
            raise TypingError(
                "bisect only supports one-dimensional NumPy arrays")
        return a.dtype, False
    raise TypingError(
        "the first argument to bisect must be a numba.typed.List or a "
        "one-dimensional NumPy array")


def _check_value(dtype, x, unicode_allowed):
    if isinstance(dtype, types.Complex) or isinstance(x, types.Complex):
        raise TypingError(
            "'<' not supported between instances of 'complex' and "
            "'complex'")
    if isinstance(dtype, types.UnicodeType):
        if not unicode_allowed:
            raise TypingError(
                "bisect only supports one-dimensional NumPy arrays with a "
                "numeric dtype")
        if not isinstance(x, types.UnicodeType):
            raise TypingError(
                "bisect requires the value to be a unicode string when the "
                "container holds unicode strings")
        return
    if isinstance(x, types.UnicodeType) or not _is_ordered_number(dtype) \
            or not _is_ordered_number(x):
        raise TypingError(
            "bisect only supports ordered numeric elements and values")


def _check_bound(ty, name):
    if isinstance(ty, (types.Integer, types.Boolean)) or isinstance(ty, int):
        return
    raise TypingError(
        "the '%s' argument to bisect must be an integer" % (name,))


def _check_args(a, x, lo, hi, key):
    dtype, unicode_allowed = _container_dtype(a)
    _check_value(dtype, x, unicode_allowed)
    _check_bound(lo, 'lo')
    if not cgutils.is_nonelike(hi):
        _check_bound(hi, 'hi')
    if not cgutils.is_nonelike(key):
        raise TypingError("bisect does not support the 'key' parameter")


@overload(_bisect.bisect_left)
def bisect_left(a, x, lo=0, hi=None, key=None):
    _check_args(a, x, lo, hi, key)

    def bisect_left_impl(a, x, lo=0, hi=None, key=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        n = len(a)
        if hi is None:
            hi = n
        if lo > n:
            raise ValueError('lo must be less than or equal to len(a)')
        if hi < 0:
            raise ValueError('hi must be non-negative')
        if hi > n:
            raise ValueError('hi must be less than or equal to len(a)')
        while lo < hi:
            mid = (lo + hi) >> 1
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return bisect_left_impl


@overload(_bisect.bisect_right)
def bisect_right(a, x, lo=0, hi=None, key=None):
    _check_args(a, x, lo, hi, key)

    def bisect_right_impl(a, x, lo=0, hi=None, key=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        n = len(a)
        if hi is None:
            hi = n
        if lo > n:
            raise ValueError('lo must be less than or equal to len(a)')
        if hi < 0:
            raise ValueError('hi must be non-negative')
        if hi > n:
            raise ValueError('hi must be less than or equal to len(a)')
        while lo < hi:
            mid = (lo + hi) >> 1
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    return bisect_right_impl
