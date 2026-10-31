"""
Python wrapper that connects CPython interpreter to the numba dictobject.
"""
from types import MappingProxyType
from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType, FrozenDictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
    overload,
    overload_method,
    box,
    unbox,
    NativeValue,
    type_callable,
    overload_classmethod,
)
from numba.typed import dictobject
from numba.core.typing import signature


_FROZEN_READONLY_MSG = (
    "'FrozenDict' object is read-only; use d.copy() to obtain a mutable "
    "numba.typed.Dict"
)


def _sentry_freeze_eligible_type(ty, role, exc):
    if isinstance(ty, (types.Number, types.Boolean, types.UnicodeType)):
        return
    if isinstance(ty, types.BaseTuple):
        for i, elem in enumerate(ty.types):
            _sentry_freeze_eligible_type(elem, f"{role} tuple element #{i}",
                                         exc)
        return
    raise exc(
        f"Cannot freeze a dictionary with {role} of type {ty}: only "
        "immutable scalars (bool, int, float, complex, str) and tuples "
        "thereof are supported; mutable values (e.g. arrays) are not "
        "deep-frozen"
    )


@njit
def _make_dict(keyty, valty, n_keys=0):
    return dictobject._as_meminfo(dictobject.new_dict(keyty, valty,
                                                      n_keys=n_keys))


@njit
def _length(d):
    return len(d)


@njit
def _setitem(d, key, value):
    d[key] = value


@njit
def _getitem(d, key):
    return d[key]


@njit
def _delitem(d, key):
    del d[key]


@njit
def _contains(d, key):
    return key in d


@njit
def _get(d, key, default):
    return d.get(key, default)


@njit
def _setdefault(d, key, default):
    return d.setdefault(key, default)


@njit
def _iter(d):
    return list(d.keys())


@njit
def _popitem(d):
    return d.popitem()


@njit
def _copy(d):
    return d.copy()


def _from_meminfo_ptr(ptr, dicttype):
    d = Dict(meminfo=ptr, dcttype=dicttype)
    return d


def _frozen_from_meminfo_ptr(ptr, dicttype):
    d = FrozenDict(meminfo=ptr, dcttype=dicttype)
    return d


class Dict(MutableMapping):
    """A typed-dictionary usable in Numba compiled functions.

    Implements the MutableMapping interface.
    """

    def __new__(cls, dcttype=None, meminfo=None, n_keys=0):
        if config.DISABLE_JIT:
            return dict.__new__(dict)
        else:
            return object.__new__(cls)

    @classmethod
    def empty(cls, key_type, value_type, n_keys=0):
        """Create a new empty Dict with *key_type* and *value_type*
        as the types for the keys and values of the dictionary respectively.

        Optionally, allocate enough memory to hold *n_keys* without requiring
        resizes. The default value of 0 returns a dict with minimum size.
        """
        if config.DISABLE_JIT:
            return dict()
        else:
            return cls(dcttype=DictType(key_type, value_type), n_keys=n_keys)

    def __init__(self, *args, **kwargs):
        """
        For users, the constructor does not take any parameters.
        The keyword arguments are for internal use only.

        Parameters
        ----------
        dcttype : numba.core.types.DictType; keyword-only
            Used internally for the dictionary type.
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        """
        if kwargs:
            self._dict_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._dict_type = None

        if args:
            # CPython checks for at most 1 argument
            # https://github.com/python/cpython/blob/f215d7cac9a6f9b51ba864e4252686dee4e45d64/Objects/dictobject.c#L2693-L2695
            _len = len(args)
            if _len > 1:
                raise errors.TypingError("Dict expect at most 1 argument, "
                                         f"got {_len}")

            # check if argument is iterable
            arg = args[0]
            if not isinstance(arg, Iterable):
                msg = (f"'{type(arg)}' object is not iterable. Supported type "
                       "constructor are Dict() and Dict(iterable)")
                raise errors.TypingError(msg)
            elif isinstance(arg, Mapping):
                raise errors.TypingError("dict(mapping) is not supported")

            for idx, item in enumerate(arg):
                if len(item) != 2:
                    msg = (f"dictionary update sequence element #{idx} has "
                           f"length {len(item)}; 2 is required")
                    raise ValueError(msg)
                k, v = item
                self.__setitem__(k, v)

    def _parse_arg(self, dcttype, meminfo=None, n_keys=0):
        if not isinstance(dcttype, DictType):
            raise TypeError('*dcttype* must be a DictType')

        if meminfo is not None:
            opaque = meminfo
        else:
            opaque = _make_dict(dcttype.key_type, dcttype.value_type,
                                n_keys=n_keys)
        return dcttype, opaque

    @property
    def _numba_type_(self):
        if self._dict_type is None:
            raise TypeError("invalid operation on untyped dictionary")
        return self._dict_type

    @property
    def _typed(self):
        """Returns True if the dictionary is typed.
        """
        return self._dict_type is not None

    def _initialise_dict(self, key, value):
        dcttype = types.DictType(typeof(key), typeof(value))
        self._dict_type, self._opaque = self._parse_arg(dcttype)

    def __getitem__(self, key):
        if not self._typed:
            raise KeyError(key)
        else:
            return _getitem(self, key)

    def __setitem__(self, key, value):
        if not self._typed:
            self._initialise_dict(key, value)
        return _setitem(self, key, value)

    def __delitem__(self, key):
        if not self._typed:
            raise KeyError(key)
        _delitem(self, key)

    def __iter__(self):
        if not self._typed:
            return iter(())
        else:
            return iter(_iter(self))

    def __len__(self):
        if not self._typed:
            return 0
        else:
            return _length(self)

    def __contains__(self, key):
        if len(self) == 0:
            return False
        else:
            return _contains(self, key)

    def __str__(self):
        buf = []
        for k, v in self.items():
            buf.append("{}: {}".format(k, v))
        return '{{{0}}}'.format(', '.join(buf))

    def __repr__(self):
        body = str(self)
        prefix = str(self._dict_type)
        return "{prefix}({body})".format(prefix=prefix, body=body)

    def get(self, key, default=None):
        if not self._typed:
            return default
        return _get(self, key, default)

    def setdefault(self, key, default=None):
        if not self._typed:
            if default is not None:
                self._initialise_dict(key, default)
        return _setdefault(self, key, default)

    def popitem(self):
        if len(self) == 0:
            raise KeyError('dictionary is empty')
        return _popitem(self)

    def copy(self):
        return _copy(self)

    def freeze(self):
        """Return an immutable, isolated snapshot of this dictionary.

        See :func:`numba.typed.freeze`.
        """
        return freeze(self)


class FrozenDict(Mapping):
    """A read-only snapshot of a homogeneous :class:`Dict`.

    Instances are created with :func:`freeze` (or ``Dict.freeze``).  The
    snapshot is an isolated copy: later mutations of the source dictionary
    do not affect it.  Lookup, ``get``, ``in``, ``len`` and insertion-ordered
    iteration are supported from both Python and compiled functions; every
    mutating operation raises ``TypeError``.
    """

    def __new__(cls, dcttype=None, meminfo=None):
        if config.DISABLE_JIT:
            return MappingProxyType({})
        return object.__new__(cls)

    def __init__(self, *, dcttype=None, meminfo=None):
        if not isinstance(dcttype, FrozenDictType):
            raise TypeError('*dcttype* must be a FrozenDictType')
        if meminfo is None:
            raise TypeError(
                'FrozenDict cannot be constructed directly; use freeze()')
        self._dict_type = dcttype
        self._opaque = meminfo

    @property
    def _numba_type_(self):
        return self._dict_type

    @property
    def _typed(self):
        return True

    def __getitem__(self, key):
        return _getitem(self, key)

    def __iter__(self):
        return iter(_iter(self))

    def __len__(self):
        return _length(self)

    def __contains__(self, key):
        return _contains(self, key)

    def __str__(self):
        buf = []
        for k, v in self.items():
            buf.append("{}: {}".format(k, v))
        return '{{{0}}}'.format(', '.join(buf))

    def __repr__(self):
        body = str(self)
        prefix = str(self._dict_type)
        return "{prefix}({body})".format(prefix=prefix, body=body)

    def get(self, key, default=None):
        return _get(self, key, default)

    def copy(self):
        return _copy(self)

    def __setitem__(self, key, value):
        raise TypeError(_FROZEN_READONLY_MSG)

    def __delitem__(self, key):
        raise TypeError(_FROZEN_READONLY_MSG)

    def pop(self, *args):
        raise TypeError(_FROZEN_READONLY_MSG)

    def popitem(self):
        raise TypeError(_FROZEN_READONLY_MSG)

    def clear(self):
        raise TypeError(_FROZEN_READONLY_MSG)

    def update(self, *args, **kwargs):
        raise TypeError(_FROZEN_READONLY_MSG)

    def setdefault(self, *args):
        raise TypeError(_FROZEN_READONLY_MSG)


def freeze(d):
    """Return a read-only snapshot of the typed dictionary *d*.

    The snapshot copies *d* and can be read from both Python and compiled
    functions.  Later mutations of *d* are not visible through the snapshot.
    Only immutable scalar keys/values (bool, int, float, complex, str) and
    tuples thereof are accepted; mutable values such as arrays are rejected
    as they cannot be deep-frozen.

    Returns
    -------
    FrozenDict
    """
    if config.DISABLE_JIT:
        return MappingProxyType(dict(d))
    if isinstance(d, FrozenDict):
        return d
    if not isinstance(d, Dict):
        raise TypeError(
            "freeze() expects a numba.typed.Dict, got {}".format(type(d)))
    if not d._typed:
        raise TypeError(
            "cannot freeze an untyped dictionary; create it with "
            "Dict.empty(key_type, value_type) first")
    _sentry_freeze_eligible_type(d._dict_type.key_type, 'keys', TypeError)
    _sentry_freeze_eligible_type(d._dict_type.value_type, 'values',
                                 TypeError)
    return _freeze(d)


@overload(freeze)
def impl_freeze(d):
    if isinstance(d, types.FrozenDictType):
        return lambda d: d
    if not isinstance(d, types.DictType):
        return
    _sentry_freeze_eligible_type(d.key_type, 'keys', errors.TypingError)
    _sentry_freeze_eligible_type(d.value_type, 'values', errors.TypingError)

    def impl(d):
        return dictobject._dict_as_frozen(d.copy())

    return impl


@overload_method(types.DictType, 'freeze')
def impl_dict_freeze(d):
    if isinstance(d, types.FrozenDictType):
        return lambda d: d
    _sentry_freeze_eligible_type(d.key_type, 'keys', errors.TypingError)
    _sentry_freeze_eligible_type(d.value_type, 'values', errors.TypingError)

    def impl(d):
        return dictobject._dict_as_frozen(d.copy())

    return impl


@njit
def _freeze(d):
    return freeze(d)


@overload_classmethod(types.DictType, 'empty')
def typeddict_empty(cls, key_type, value_type, n_keys=0):
    if cls.instance_type is not DictType:
        return

    def impl(cls, key_type, value_type, n_keys=0):
        return dictobject.new_dict(key_type, value_type, n_keys=n_keys)

    return impl


def _box_dicttype(typ, val, c, from_meminfo_name):
    context = c.context
    builder = c.builder

    # XXX deduplicate
    ctor = cgutils.create_struct_proxy(typ)
    dstruct = ctor(context, builder, value=val)
    # Returns the plain MemInfo
    boxed_meminfo = c.box(
        types.MemInfoPointer(types.voidptr),
        dstruct.meminfo,
    )

    modname = c.context.insert_const_string(
        c.builder.module, 'numba.typed.typeddict',
    )
    typeddict_mod = c.pyapi.import_module(modname)
    fmp_fn = c.pyapi.object_getattr_string(typeddict_mod, from_meminfo_name)

    dicttype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

    result_var = builder.alloca(c.pyapi.pyobj)
    builder.store(cgutils.get_null_value(c.pyapi.pyobj), result_var)
    with builder.if_then(cgutils.is_not_null(builder, dicttype_obj)):
        res = c.pyapi.call_function_objargs(
            fmp_fn, (boxed_meminfo, dicttype_obj),
        )
        c.pyapi.decref(fmp_fn)
        c.pyapi.decref(typeddict_mod)
        c.pyapi.decref(boxed_meminfo)
        builder.store(res, result_var)
    return builder.load(result_var)


@box(types.DictType)
def box_dicttype(typ, val, c):
    return _box_dicttype(typ, val, c, '_from_meminfo_ptr')


@box(types.FrozenDictType)
def box_frozen_dicttype(typ, val, c):
    return _box_dicttype(typ, val, c, '_frozen_from_meminfo_ptr')


def _unbox_dicttype(typ, val, c, py_dict_cls):
    context = c.context

    # Check that `type(val) is py_dict_cls`
    dict_type = c.pyapi.unserialize(c.pyapi.serialize_object(py_dict_cls))
    valtype = c.pyapi.object_type(val)
    same_type = c.builder.icmp_unsigned("==", valtype, dict_type)

    with c.builder.if_else(same_type) as (then, orelse):
        with then:
            miptr = c.pyapi.object_getattr_string(val, '_opaque')

            mip_type = types.MemInfoPointer(types.voidptr)
            native = c.unbox(mip_type, miptr)

            mi = native.value

            argtypes = mip_type, typeof(typ)

            def convert(mi, typ):
                return dictobject._from_meminfo(mi, typ)

            sig = signature(typ, *argtypes)
            nil_typeref = context.get_constant_null(argtypes[1])
            args = (mi, nil_typeref)
            is_error, dctobj = c.pyapi.call_jit_code(convert, sig, args)
            # decref here because we are stealing a reference.
            c.context.nrt.decref(c.builder, typ, dctobj)

            c.pyapi.decref(miptr)
            bb_unboxed = c.builder.basic_block

        with orelse:
            # Raise error on incorrect type
            c.pyapi.err_format(
                "PyExc_TypeError",
                "can't unbox a %S as a %S",
                valtype, dict_type,
            )
            bb_else = c.builder.basic_block

    # Phi nodes to gather the output
    dctobj_res = c.builder.phi(dctobj.type)
    is_error_res = c.builder.phi(is_error.type)

    dctobj_res.add_incoming(dctobj, bb_unboxed)
    dctobj_res.add_incoming(dctobj.type(None), bb_else)

    is_error_res.add_incoming(is_error, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)

    # cleanup
    c.pyapi.decref(dict_type)
    c.pyapi.decref(valtype)

    return NativeValue(dctobj_res, is_error=is_error_res)


@unbox(types.DictType)
def unbox_dicttype(typ, val, c):
    return _unbox_dicttype(typ, val, c, Dict)


@unbox(types.FrozenDictType)
def unbox_frozen_dicttype(typ, val, c):
    return _unbox_dicttype(typ, val, c, FrozenDict)


@type_callable(DictType)
def typeddict_call(context):
    """
    Defines typing logic for ``Dict()`` and ``Dict(iterable)``.
    Produces Dict[undefined, undefined] or Dict[key, value]
    """
    def typer(arg=None):
        if arg is None:
            return types.DictType(types.undefined, types.undefined)
        elif type(arg) is types.DictType:
            return arg
        elif isinstance(arg, types.FrozenDictType):
            raise errors.TypingError(
                "cannot construct a mutable Dict from a FrozenDictType; "
                "use .copy() in compiled code")
        elif isinstance(arg, types.Tuple) and len(arg) == 0:  # Dict(())
            msg = "non-precise type 'dict(())'"
            raise errors.TypingError(msg)
        elif isinstance(arg, types.IterableType):
            dtype = arg.iterator_type.yield_type
            if isinstance(dtype, types.UniTuple):
                key = value = dtype.key[0]
                return types.DictType(key, value)
            elif isinstance(dtype, types.Tuple):
                key, value = dtype.key
                return types.DictType(key, value)
    return typer


@overload(numba_typeref_ctor)
def impl_numba_typeref_ctor(cls, *args):
    """
    Defines lowering for ``Dict()`` and ``Dict(iterable)``.

    The type-inferred version of the dictionary ctor.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise DictType.
    args: tuple
        A tuple that contains a single iterable (optional)

    Returns
    -------
    impl : function
        An implementation suitable for lowering the constructor call.

    See also: `redirect_type_ctor` in numba/cpython/builtins.py
    """
    dict_ty = cls.instance_type
    if not isinstance(dict_ty, types.DictType):
        return  # reject
    # Ensure the dictionary is precisely typed.
    if not dict_ty.is_precise():
        msg = "expecting a precise DictType but got {}".format(dict_ty)
        raise errors.LoweringError(msg)

    key_type = types.TypeRef(dict_ty.key_type)
    value_type = types.TypeRef(dict_ty.value_type)

    if args:
        if isinstance(args[0], types.IterableType):
            def impl(cls, *args):
                # Instantiate an empty dict and populate it with values from
                # the iterable.
                d = Dict.empty(key_type, value_type)
                for k, v in args[0]:
                    d[k] = v
                return d
    else:
        def impl(cls, *args):
            # Simply call .empty() with the key/value types from *cls*
            return Dict.empty(key_type, value_type)

    return impl
