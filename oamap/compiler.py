#!/usr/bin/env python

# Copyright (c) 2017, DIANA-HEP
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pickle
import sys

import numpy

import oamap.schema
import oamap.generator
import oamap.proxy

if sys.version_info[0] > 2:
    basestring = str

try:
    import numba
    import llvmlite.llvmpy.core
except ImportError:
    pass
else:
    ################################################################ Schema objects in compiled code

    class SchemaType(numba.types.Type):
        def __init__(self, schema, generator=None, matchable=True):
            self.schema = schema
            self.matchable = matchable
            if generator is None:
                self.generator = self.schema.generator()
            else:
                self.generator = generator
            super(SchemaType, self).__init__(name="OAMap-Schema{0} {1}".format("" if self.matchable else " (unmatchable)", self.schema.tojsonstring()))

        def unmatchable(self):
            return SchemaType(self.schema, generator=self.generator, matchable=False)

        def content(self):
            return SchemaType(self.schema.content, generator=self.generator.content)

        def possibilities(self, i):
            return SchemaType(self.schema.possibilities[i], generator=self.generator.possibilities[i])

        def fields(self, n):
            return SchemaType(self.schema.fields[n], generator=self.generator.fields[n])

        def types(self, i):
            return SchemaType(self.schema.types[i], generator=self.generator.types[i])

        def target(self):
            return SchemaType(self.schema.target, generator=self.generator.target)

    @numba.extending.typeof_impl.register(oamap.schema.Schema)
    def typeof_proxy(val, c):
        return SchemaType(val)

    @numba.extending.register_model(SchemaType)
    class SchemaModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            # don't carry any information about the Schema at runtime: it's a purely compile-time constant
            super(SchemaModel, self).__init__(dmm, fe_type, [])
    
    primtypes = (numba.types.Boolean, numba.types.Integer, numba.types.Float, numba.types.Complex, numba.types.npytypes.CharSeq)

    @numba.extending.infer_getattr
    class SchemaAttribute(numba.typing.templates.AttributeTemplate):
        key = SchemaType
        def generic_resolve(self, typ, attr):
            if typ.matchable:
                if attr == "nullable":
                    return numba.types.boolean

                elif isinstance(typ.schema, oamap.schema.Primitive) and attr == "dtype":
                    return numba.types.DType(numba.from_dtype(typ.schema.dtype))

                elif isinstance(typ.schema, oamap.schema.List) and attr == "content":
                    return typ.content()

                elif isinstance(typ.schema, oamap.schema.Union) and attr == "possibilities":
                    return typ.unmatchable()

                elif isinstance(typ.schema, oamap.schema.Record) and attr == "fields":
                    return typ.unmatchable()

                elif isinstance(typ.schema, oamap.schema.Tuple) and attr == "types":
                    return typ.unmatchable()

                elif isinstance(typ.schema, oamap.schema.Pointer) and attr == "target":
                    return typ.target()

                else:
                    raise AssertionError("unrecognized schema type: {0} ({1})".format(typ.schema.__class__, repr(typ.schema)))

        @numba.typing.templates.bound_function("schema.case")
        def resolve_case(self, schematype, args, kwds):
            if len(args) == 1:
                arg, = args
                if isinstance(arg, primtypes + (ProxyNumbaType,)) or (isinstance(arg, numba.types.Optional) and isinstance(arg.type, primtypes + (ProxyNumbaType,))):
                    return numba.types.boolean(arg)

        @numba.typing.templates.bound_function("schema.cast")
        def resolve_cast(self, schematype, args, kwds):
            if len(args) == 1:
                arg, = args
                if isinstance(schematype.schema, oamap.schema.Primitive):
                    if schematype.schema.nullable:
                        return numba.types.optional(numba.from_dtype(schematype.schema.dtype))(arg)
                    else:
                        return numba.from_dtype(schematype.schema.dtype)(arg)

                elif isinstance(schematype.schema, (oamap.schema.List, oamap.schema.Union, oamap.schema.Record, oamap.schema.Tuple)):
                    if isinstance(arg, UnionProxyNumbaType):
                        for datatag, datatype in enumerate(arg.generator.schema.possibilities):
                            if schematype.schema == datatype:
                                return typeof_generator(arg.generator.possibilities[datatag])(arg)

                    if isinstance(arg, numba.types.Optional) and isinstance(arg.type, UnionProxyNumbaType):
                        for datatag, datatype in enumerate(arg.type.generator.schema.possibilities):
                            if schematype.schema == datatype:
                                return typeof_generator(arg.type.generator.possibilities[datatag])(arg)

                    if arg.generator.schema == schematype.schema:
                        return typeof_generator(arg.generator)(arg)
                    else:
                        return typeof_generator(schematype.generator)(arg)

                elif isinstance(schematype.schema, oamap.schema.Pointer):
                    raise TypeError("cannot cast data as a Pointer")

                else:
                    raise AssertionError("unrecognized schema type: {0} ({1})".format(schematype.schema.__class__, repr(schematype.schema)))

    @numba.extending.lower_getattr_generic(SchemaType)
    def schema_getattr(context, builder, typ, val, attr):
        if attr == "nullable":
            return literal_boolean(1 if typ.schema.nullable else 0)

        elif attr == "dtype":
            return numba.targets.imputils.impl_ret_untracked(context, builder, numba.types.DType(numba.from_dtype(typ.schema.dtype)), context.get_dummy_value())

        else:
            return numba.cgutils.create_struct_proxy(typ)(context, builder)._getvalue()

    @numba.extending.lower_builtin("schema.case", SchemaType, numba.types.Type)
    def schema_case(context, builder, sig, args):
        schematype, argtype = sig.args
        dummy, argval = args

        if isinstance(argtype, numba.types.Optional):
            # unwrap the optval and apply the check to the contents
            optval = context.make_helper(builder, argtype, value=argval)
            out = schema_case(context, builder, numba.types.boolean(schematype, argtype.type), (dummy, optval.data))
            if schematype.schema.nullable:
                return out
            else:
                return builder.and_(optval.valid, out)

        elif isinstance(argtype, UnionProxyNumbaType):
            # do a runtime check
            for datatag, datatype in enumerate(argtype.generator.schema.possibilities):
                if schematype.schema == datatype:
                    unionproxy = numba.cgutils.create_struct_proxy(argtype)(context, builder, value=argval)
                    out_ptr = numba.cgutils.alloca_once(builder, llvmlite.llvmpy.core.Type.int(1))
                    with builder.if_else(builder.icmp_unsigned("==", unionproxy.tag, literal_int(datatag, argtype.generator.tagdtype.itemsize))) as (success, failure):
                        with success:
                            builder.store(literal_boolean(True), out_ptr)
                        with failure:
                            builder.store(literal_boolean(False), out_ptr)
                    out = builder.load(out_ptr)
                    return out

            # none of the data possibilities will ever match
            return literal_boolean(False)

        elif isinstance(argtype, primtypes):
            # do a compile-time check
            if isinstance(schematype.schema, oamap.schema.Primitive):
                return literal_boolean(numba.from_dtype(schematype.schema.dtype) == argtype)
            else:
                return literal_boolean(False)

        elif isinstance(argtype, ProxyNumbaType):
            # do a compile-time check
            return literal_boolean(schematype.schema == argtype.generator.schema)

        else:
            raise AssertionError

    @numba.extending.lower_builtin("schema.cast", SchemaType, numba.types.Type)
    def schema_cast(context, builder, sig, args):
        outtype, (schematype, argtype) = sig.return_type, sig.args
        dummy, argval = args

        def error(case):
            raise_exception(context, builder, case, TypeError("cannot cast {0} to {1}".format(argtype, outtype)))

        def error2(case):
            raise_exception(context, builder, case, TypeError("cannot cast all members of {0} to {1}".format(argtype, outtype)))

        if argtype == outtype:
            return argval

        if isinstance(argtype, numba.types.Optional):
            # unwrap the optval and apply the check to the contents
            optval = context.make_helper(builder, argtype, value=argval)
            out = schema_cast(context, builder, outtype(schematype, argtype.type), (dummy, optval.data))
            if schematype.schema.nullable:
                return out
            else:
                error2(builder.not_(optval.valid))
                return out

        elif isinstance(argtype, UnionProxyNumbaType):
            # do a runtime check
            for datatag, datatype in enumerate(argtype.generator.schema.possibilities):
                if schematype.schema == datatype:
                    unionproxy = numba.cgutils.create_struct_proxy(argtype)(context, builder, value=argval)
                    error2(builder.icmp_unsigned("!=", unionproxy.tag, literal_int(datatag, argtype.generator.tagdtype.itemsize)))
                    return generate(context, builder, argtype.generator.possibilities[datatag], unionproxy.baggage, unionproxy.ptrs, unionproxy.lens, unionproxy.offset)

            # none of the data possibilities will ever match
            error(None)
            argproxy = numba.cgutils.create_struct_proxy(argtype)(context, builder)
            return generate_empty(context, builder, outtype.generator, argproxy.baggage)

        elif isinstance(argtype, primtypes):
            # do a conversion
            return context.cast(builder, argval, argtype, outtype)

        elif isinstance(argtype, ProxyNumbaType):
            # always fail
            error(None)
            argproxy = numba.cgutils.create_struct_proxy(argtype)(context, builder)
            return generate_empty(context, builder, outtype.generator, argproxy.baggage)

        else:
            raise AssertionError

    @numba.typing.templates.infer
    class SchemaGetItem(numba.typing.templates.AbstractTemplate):
        key = "static_getitem"
        def generic(self, args, kwds):
            if len(args) == 2:
                tpe, idx = args
                if isinstance(tpe, SchemaType) and isinstance(tpe.schema, oamap.schema.Union) and isinstance(idx, int):
                    if idx < 0:
                        normindex = idx + len(tpe.schema._possibilities)
                    else:
                        normindex = idx
                    if 0 <= normindex < len(tpe.schema._possibilities):
                        return tpe.possibilities(normindex)
                    else:
                        raise IndexError("possibility {0} out of range for type {1}".format(idx, tpe.schema))
                    
                elif isinstance(tpe, SchemaType) and isinstance(tpe.schema, (oamap.schema.Record)) and isinstance(idx, basestring):
                    if idx in tpe.schema._fields:
                        return tpe.fields(idx)
                    else:
                        raise KeyError("no field named {0} in type {1}".format(repr(idx), tpe.schema))

                elif isinstance(tpe, SchemaType) and isinstance(tpe.schema, oamap.schema.Tuple) and isinstance(idx, int):
                    if idx < 0:
                        normindex = idx + len(tpe.schema._types)
                    else:
                        normindex = idx
                    if 0 <= normindex < len(tpe.schema._types):
                        return tpe.types(normindex)
                    else:
                        raise IndexError("item {0} out of range for type {1}".format(idx, tpe.schema))

    @numba.extending.lower_builtin("static_getitem", SchemaType, numba.types.Const)
    def schema_static_getitem(context, builder, sig, args):
        typ, _ = sig.args
        return numba.cgutils.create_struct_proxy(typ)(context, builder)._getvalue()

    @numba.targets.imputils.lower_constant(SchemaType)
    def schema_constant(context, builder, ty, pyval):
        return numba.cgutils.create_struct_proxy(ty)(context, builder)._getvalue()

    @numba.extending.unbox(SchemaType)
    def unbox_schema(typ, obj, c):
        # no information is carried over at runtime
        schemaproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
        is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return numba.extending.NativeValue(schemaproxy._getvalue(), is_error=is_error)

    @numba.extending.box(SchemaType)
    def box_schema(typ, val, c):
        # generate schema from the compile-time constant
        if typ.matchable:
            return c.pyapi.unserialize(c.pyapi.serialize_object(typ.schema))

        elif isinstance(typ.schema, oamap.schema.Union):
            return c.pyapi.unserialize(c.pyapi.serialize_object(typ.schema.possibilities))

        elif isinstance(typ.schema, oamap.schema.Record):
            out = c.pyapi.dict_new(len(typ.schema._fields))
            for n, x in typ.schema._fields.items():
                c.pyapi.dict_setitem_string(out, n, c.pyapi.unserialize(c.pyapi.serialize_object(x)))
            return out

        elif isinstance(typ.schema, oamap.schema.Tuple):
            return c.pyapi.unserialize(c.pyapi.serialize_object(typ.schema.types))

        else:
            raise AssertionError

    ################################################################ Baggage (tracing reference counts to reconstitute Python objects)

    class BaggageType(numba.types.Type):
        def __init__(self):
            super(BaggageType, self).__init__(name="OAMap-Baggage")

    baggagetype = BaggageType()

    @numba.extending.register_model(BaggageType)
    class BaggageModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("arrays", numba.types.pyobject),
                       ("cache", numba.types.pyobject),
                       ("ptrs", numba.types.pyobject),
                       ("lens", numba.types.pyobject)]
            super(BaggageModel, self).__init__(dmm, fe_type, members)

    def unbox_baggage(context, builder, pyapi, generator_obj, arrays_obj, cache_obj):
        entercompiled_fcn = pyapi.object_getattr_string(generator_obj, "_entercompiled")
        results_obj = pyapi.call_function_objargs(entercompiled_fcn, (arrays_obj, cache_obj))
        with builder.if_then(numba.cgutils.is_not_null(builder, pyapi.err_occurred()), likely=False):
            builder.ret(llvmlite.llvmpy.core.Constant.null(pyapi.pyobj))

        baggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder)
        baggage.arrays = arrays_obj
        baggage.cache = cache_obj
        baggage.ptrs = pyapi.tuple_getitem(results_obj, 0)
        baggage.lens = pyapi.tuple_getitem(results_obj, 1)

        ptrs_obj = pyapi.tuple_getitem(results_obj, 2)
        lens_obj = pyapi.tuple_getitem(results_obj, 3)
        ptrs = pyapi.long_as_voidptr(ptrs_obj)
        lens = pyapi.long_as_voidptr(lens_obj)

        pyapi.incref(baggage.ptrs)
        pyapi.incref(baggage.lens)

        pyapi.decref(generator_obj)
        pyapi.decref(results_obj)

        return baggage._getvalue(), ptrs, lens

    def box_baggage(context, builder, pyapi, generator, baggage_val):
        generator_obj = pyapi.unserialize(pyapi.serialize_object(generator))
        new_fcn = pyapi.object_getattr_string(generator_obj, "_new")
        results_obj = pyapi.call_function_objargs(new_fcn, ())
        with builder.if_then(numba.cgutils.is_not_null(builder, pyapi.err_occurred()), likely=False):
            builder.ret(llvmlite.llvmpy.core.Constant.null(pyapi.pyobj))
        pyapi.decref(results_obj)

        baggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=baggage_val)

        return generator_obj, baggage.arrays, baggage.cache

    ################################################################ general routines for all proxies

    class ProxyNumbaType(numba.types.Type):
        def __repr__(self):
            return "\n    " + self.generator.schema.__repr__(indent="    ") + "\n"

        def unify(self, context, other):
            if isinstance(other, ProxyNumbaType) and self.generator is other.generator:
                return self

    @numba.extending.typeof_impl.register(oamap.proxy.Proxy)
    def typeof_proxy(val, c):
        return typeof_generator(val._generator)

    def typeof_generator(generator, checkmasked=True):
        if checkmasked and isinstance(generator, oamap.generator.Masked):
            tpe = typeof_generator(generator, checkmasked=False)
            if isinstance(tpe, numba.types.Optional):
                return tpe
            else:
                return numba.types.optional(tpe)

        if isinstance(generator, oamap.generator.PrimitiveGenerator):
            return numba.from_dtype(generator.dtype)

        elif isinstance(generator, oamap.generator.ListGenerator):
            return ListProxyNumbaType(generator)

        elif isinstance(generator, oamap.generator.UnionGenerator):
            return UnionProxyNumbaType(generator)

        elif isinstance(generator, oamap.generator.RecordGenerator):
            return RecordProxyNumbaType(generator)

        elif isinstance(generator, oamap.generator.TupleGenerator):
            return TupleProxyNumbaType(generator)

        elif isinstance(generator, oamap.generator.PointerGenerator):
            return typeof_generator(generator.target)

        elif isinstance(generator, oamap.generator.ExtendedGenerator):
            return typeof_generator(generator.generic)

        else:
            raise AssertionError("unrecognized generator type: {0} ({1})".format(generator.__class__, repr(generator)))

    def literal_int(value, itemsize):
        return llvmlite.llvmpy.core.Constant.int(llvmlite.llvmpy.core.Type.int(itemsize * 8), value)

    def literal_int64(value):
        return literal_int(value, 8)

    def literal_intp(value):
        return literal_int(value, numba.types.intp.bitwidth // 8)

    def literal_boolean(value):
        return llvmlite.llvmpy.core.Constant.int(llvmlite.llvmpy.core.Type.int(1), value)

    def cast_int(builder, value, itemsize):
        bitwidth = itemsize * 8
        if value.type.width < bitwidth:
            return builder.zext(value, llvmlite.llvmpy.core.Type.int(bitwidth))
        elif value.type.width > bitwidth:
            return builder.trunc(value, llvmlite.llvmpy.core.Type.int(bitwidth))
        else:
            return builder.bitcast(value, llvmlite.llvmpy.core.Type.int(bitwidth))

    def cast_int64(builder, value):
        return cast_int(builder, value, 8)

    def cast_intp(builder, value):
        return cast_int(builder, value, numba.types.intp.bitwidth // 8)

    def all_(builder, predicates, *args, **kwds):
        if len(predicates) == 0:
            return literal_boolean(True)
        elif len(predicates) == 1:
            first, = predicates
            return first
        else:
            first, rest = predicates[0], predicates[1:]
            return builder.and_(first, all_(builder, rest, *args, **kwds), *args, **kwds)

    def any_(builder, predicates, *args, **kwds):
        if len(predicates) == 0:
            return literal_boolean(False)
        elif len(predicates) == 1:
            first, = predicates
            return first
        else:
            first, rest = predicates[0], predicates[1:]
            return builder.or_(first, any_(builder, rest, *args, **kwds), *args, **kwds)

    def arrayitem(context, builder, idx, ptrs, lens, at, dtype):
        offset = builder.mul(idx, literal_int64(numba.types.intp.bitwidth // 8))

        ptrposition = builder.inttoptr(
            builder.add(builder.ptrtoint(ptrs, llvmlite.llvmpy.core.Type.int(numba.types.intp.bitwidth)), offset),
            llvmlite.llvmpy.core.Type.pointer(context.get_value_type(numba.types.intp)))

        lenposition = builder.inttoptr(
            builder.add(builder.ptrtoint(lens, llvmlite.llvmpy.core.Type.int(numba.types.intp.bitwidth)), offset),
            llvmlite.llvmpy.core.Type.pointer(context.get_value_type(numba.types.intp)))

        ptr = numba.targets.arrayobj.load_item(context, builder, numba.types.intp[:], ptrposition)
        len = numba.targets.arrayobj.load_item(context, builder, numba.types.intp[:], lenposition)

        raise_exception(context, builder, builder.icmp_unsigned(">=", at, len), RuntimeError("array index out of range"))

        finalptr = builder.inttoptr(
            builder.add(ptr, builder.mul(at, literal_int64(dtype.itemsize))),
            llvmlite.llvmpy.core.Type.pointer(context.get_value_type(numba.from_dtype(dtype))))

        return numba.targets.arrayobj.load_item(context, builder, numba.from_dtype(dtype)[:], finalptr)

    def raise_exception(context, builder, case, exception):
        if case is None:
            case = builder.icmp_unsigned("==", literal_int64(0), literal_int64(0))

        with builder.if_then(case, likely=False):
            pyapi = context.get_python_api(builder)
            excptr = context.call_conv._get_excinfo_argument(builder.function)

            if excptr.name == "excinfo" and excptr.type == llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.struct([llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.int(8)), llvmlite.llvmpy.core.Type.int(32)]))):
                exc = pyapi.serialize_object(exception)
                builder.store(exc, excptr)
                builder.ret(numba.targets.callconv.RETCODE_USEREXC)

            elif excptr.name == "py_args" and excptr.type == llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.int(8)):
                exc = pyapi.unserialize(pyapi.serialize_object(exception))
                pyapi.raise_object(exc)
                builder.ret(llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.pyobject)))

            else:
                raise AssertionError("unrecognized exception calling convention: {0}".format(excptr))

    def generate_empty(context, builder, generator, baggage):
        typ = typeof_generator(generator, checkmasked=False)

        if isinstance(generator, oamap.generator.PrimitiveGenerator):
            return llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.from_dtype(generator.dtype)))

        elif isinstance(generator, oamap.generator.ListGenerator):
            listproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            listproxy.baggage = baggage
            listproxy.ptrs = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            listproxy.lens = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            listproxy.whence = literal_int64(-1)
            listproxy.stride = literal_int64(-1)
            listproxy.length = literal_int64(-1)
            return listproxy._getvalue()

        elif isinstance(generator, oamap.generator.UnionGenerator):
            unionproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            unionproxy.baggage = baggage
            unionproxy.ptrs = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            unionproxy.lens = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            unionproxy.tag = literal_int64(-1)
            unionproxy.offset = literal_int64(-1)
            return unionproxy._getvalue()

        elif isinstance(generator, oamap.generator.RecordGenerator):
            recordproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            recordproxy.baggage = baggage
            recordproxy.ptrs = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            recordproxy.lens = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            recordproxy.index = literal_int64(-1)
            return recordproxy._getvalue()

        elif isinstance(generator, oamap.generator.TupleGenerator):
            tupleproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            tupleproxy.baggage = baggage
            tupleproxy.ptrs = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            tupleproxy.lens = llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))
            tupleproxy.index = literal_int64(-1)
            return tupleproxy._getvalue()

        elif isinstance(generator, oamap.generator.PointerGenerator):
            return generate_empty(context, builder, generator.target, baggage)

        elif isinstance(generator, oamap.generator.ExtendedGenerator):
            return generate(context, builder, generator.generic, baggage, ptrs, lens, at)

        else:
            raise AssertionError("unrecognized generator type: {0} ({1})".format(generator.__class__, repr(generator)))

    def generate(context, builder, generator, baggage, ptrs, lens, at, checkmasked=True):
        generator._required = True

        if checkmasked and isinstance(generator, oamap.generator.Masked):
            maskidx = literal_int64(generator.maskidx)
            maskvalue = arrayitem(context, builder, maskidx, ptrs, lens, at, generator.maskdtype)

            comparison = builder.icmp_unsigned("==", maskvalue, literal_int(generator.maskedvalue, generator.maskdtype.itemsize))

            outoptval = context.make_helper(builder, typeof_generator(generator))

            if isinstance(generator, oamap.generator.PointerGenerator) and isinstance(generator.target, oamap.generator.Masked):
                with builder.if_else(comparison) as (is_not_valid, is_valid):
                    with is_valid:
                        nested = generate(context, builder, generator, baggage, ptrs, lens, cast_int64(builder, maskvalue), checkmasked=False)
                        wrapped = context.make_helper(builder, typeof_generator(generator), value=nested)
                        outoptval.valid = wrapped.valid
                        outoptval.data  = wrapped.data

                    with is_not_valid:
                        outoptval.valid = numba.cgutils.false_bit
                        outoptval.data = generate_empty(context, builder, generator, baggage)

            else:
                with builder.if_else(comparison) as (is_not_valid, is_valid):
                    with is_valid:
                        outoptval.valid = numba.cgutils.true_bit
                        outoptval.data = generate(context, builder, generator, baggage, ptrs, lens, cast_int64(builder, maskvalue), checkmasked=False)

                    with is_not_valid:
                        outoptval.valid = numba.cgutils.false_bit
                        outoptval.data = generate_empty(context, builder, generator, baggage)

            return outoptval._getvalue()

        typ = typeof_generator(generator, checkmasked=False)

        if isinstance(generator, oamap.generator.PrimitiveGenerator):
            dataidx = literal_int64(generator.dataidx)
            return arrayitem(context, builder, dataidx, ptrs, lens, at, generator.dtype)

        elif isinstance(generator, oamap.generator.ListGenerator):
            startsidx = literal_int64(generator.startsidx)
            stopsidx  = literal_int64(generator.stopsidx)
            start = cast_int64(builder, arrayitem(context, builder, startsidx, ptrs, lens, at, generator.posdtype))
            stop  = cast_int64(builder, arrayitem(context, builder, stopsidx,  ptrs, lens, at, generator.posdtype))
            listproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            listproxy.baggage = baggage
            listproxy.ptrs = ptrs
            listproxy.lens = lens
            listproxy.whence = start
            listproxy.stride = literal_int64(1)
            listproxy.length = builder.sub(stop, start)
            return listproxy._getvalue()

        elif isinstance(generator, oamap.generator.UnionGenerator):
            tagsidx    = literal_int64(generator.tagsidx)
            offsetsidx = literal_int64(generator.offsetsidx)
            tag    = cast_int64(builder, arrayitem(context, builder, tagsidx,    ptrs, lens, at, generator.tagdtype))
            offset = cast_int64(builder, arrayitem(context, builder, offsetsidx, ptrs, lens, at, generator.offsetdtype))
            raise_exception(context,
                            builder,
                            builder.or_(builder.icmp_signed("<", tag, literal_int64(0)),
                                        builder.icmp_signed(">=", tag, literal_int64(len(generator.possibilities)))),
                            RuntimeError("tag out of bounds for union"))
            unionproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            unionproxy.baggage = baggage
            unionproxy.ptrs = ptrs
            unionproxy.lens = lens
            unionproxy.tag = tag
            unionproxy.offset = offset
            return unionproxy._getvalue()

        elif isinstance(generator, oamap.generator.RecordGenerator):
            recordproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            recordproxy.baggage = baggage
            recordproxy.ptrs = ptrs
            recordproxy.lens = lens
            recordproxy.index = at
            return recordproxy._getvalue()

        elif isinstance(generator, oamap.generator.TupleGenerator):
            tupleproxy = numba.cgutils.create_struct_proxy(typ)(context, builder)
            tupleproxy.baggage = baggage
            tupleproxy.ptrs = ptrs
            tupleproxy.lens = lens
            tupleproxy.index = at
            return tupleproxy._getvalue()

        elif isinstance(generator, oamap.generator.PointerGenerator):
            positionsidx = literal_int64(generator.positionsidx)
            index = cast_int64(builder, arrayitem(context, builder, positionsidx, ptrs, lens, at, generator.posdtype))
            return generate(context, builder, generator.target, baggage, ptrs, lens, index)

        elif isinstance(generator, oamap.generator.ExtendedGenerator):
            return generate(context, builder, generator.generic, baggage, ptrs, lens, at)

        else:
            raise AssertionError("unrecognized generator type: {0} ({1})".format(generator.__class__, repr(generator)))

    class ProxyCompare(numba.typing.templates.AbstractTemplate):
        def generic(self, args, kwds):
            lhs, rhs = args
            if isinstance(lhs, numba.types.Optional):
                lhs = lhs.type
            if isinstance(rhs, numba.types.Optional):
                rhs = rhs.type

            if isinstance(lhs, ListProxyNumbaType) and isinstance(rhs, ListProxyNumbaType):
                if lhs.generator.schema.content == rhs.generator.schema.content:
                    return numba.types.boolean(*args)

            elif isinstance(lhs, RecordProxyNumbaType) and isinstance(rhs, RecordProxyNumbaType):
                if lhs.generator.schema.fields == rhs.generator.schema.fields:
                    return numba.types.boolean(*args)

            elif isinstance(lhs, TupleProxyNumbaType) and isinstance(rhs, TupleProxyNumbaType):
                if lhs.generator.schema.types == rhs.generator.schema.types:
                    return numba.types.boolean(*args)

            elif isinstance(lhs, UnionProxyNumbaType) and isinstance(rhs, UnionProxyNumbaType):
                for x in lhs.generator.schema.possibilities:
                    for y in rhs.generator.schema.possibilities:
                        if x.copy(nullable=False) == y.copy(nullable=False):
                            return numba.types.boolean(*args)

            elif isinstance(lhs, UnionProxyNumbaType) and isinstance(rhs, ProxyNumbaType):
                for x in lhs.generator.schema.possibilities:
                    if x.copy(nullable=False) == rhs.generator.schema.copy(nullable=False):
                        return numba.types.boolean(*args)

            elif isinstance(rhs, UnionProxyNumbaType) and isinstance(lhs, ProxyNumbaType):
                for x in rhs.generator.schema.possibilities:
                    if x.copy(nullable=False) == lhs.generator.schema.copy(nullable=False):
                        return numba.types.boolean(*args)

            elif isinstance(lhs, UnionProxyNumbaType) and isinstance(rhs, primtypes):
                for x in lhs.generator.schema.possibilities:
                    if isinstance(x, oamap.schema.Primitive) and numba.from_dtype(x.dtype) == rhs:
                        return numba.types.boolean(*args)

            elif isinstance(rhs, UnionProxyNumbaType) and isinstance(lhs, primtypes):
                for x in rhs.generator.schema.possibilities:
                    if isinstance(x, oamap.schema.Primitive) and numba.from_dtype(x.dtype) == lhs:
                        return numba.types.boolean(*args)

    ################################################################ ListProxy

    class ListProxyNumbaType(ProxyNumbaType):
        def __init__(self, generator):
            self.generator = generator
            super(ListProxyNumbaType, self).__init__(name="OAMap-ListProxy-" + self.generator.id)

    @numba.extending.register_model(ListProxyNumbaType)
    class ListProxyModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("baggage", baggagetype),
                       ("ptrs", numba.types.voidptr),
                       ("lens", numba.types.voidptr),
                       ("whence", numba.types.int64),
                       ("stride", numba.types.int64),
                       ("length", numba.types.int64)]
            super(ListProxyModel, self).__init__(dmm, fe_type, members)

    numba.extending.make_attribute_wrapper(ListProxyNumbaType, "whence", "_whence")
    numba.extending.make_attribute_wrapper(ListProxyNumbaType, "stride", "_stride")
    numba.extending.make_attribute_wrapper(ListProxyNumbaType, "length", "_length")

    @numba.extending.type_callable(len)
    def listproxy_len_type(context):
        def typer(listproxy):
            if isinstance(listproxy, ListProxyNumbaType):
                return numba.types.int64   # verified len type
        return typer

    @numba.extending.lower_builtin(len, ListProxyNumbaType)
    def listproxy_len(context, builder, sig, args):
        listtpe, = sig.args
        listval, = args
        listproxy = numba.cgutils.create_struct_proxy(listtpe)(context, builder, value=listval)
        return listproxy.length

    @numba.typing.templates.infer
    class ListProxyGetItem(numba.typing.templates.AbstractTemplate):
        key = "getitem"
        def generic(self, args, kwds):
            if len(args) == 2:
                tpe, idx = args
                if isinstance(tpe, ListProxyNumbaType):
                    if isinstance(idx, numba.types.Integer):
                        return typeof_generator(tpe.generator.content)(tpe, idx)
                    elif isinstance(idx, numba.types.SliceType):
                        return typeof_generator(tpe.generator)(tpe, idx)

    @numba.extending.lower_builtin("getitem", ListProxyNumbaType, numba.types.Integer)
    def listproxy_getitem(context, builder, sig, args):
        listtpe, indextpe = sig.args
        listval, indexval = args

        listproxy = numba.cgutils.create_struct_proxy(listtpe)(context, builder, value=listval)
        if listtpe.generator.schema.nullable:
            raise_exception(context,
                            builder,
                            builder.icmp_signed("==", listproxy.ptrs, llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))),
                            TypeError("'NoneType' object has no attribute '__getitem__'"))

        indexval = cast_int64(builder, indexval)

        normindex_ptr = numba.cgutils.alloca_once_value(builder, indexval)
        with builder.if_then(builder.icmp_signed("<", indexval, literal_int64(0))):
            builder.store(builder.add(indexval, listproxy.length), normindex_ptr)
        normindex = builder.load(normindex_ptr)

        raise_exception(context,
                        builder,
                        builder.or_(builder.icmp_signed("<", normindex, literal_int64(0)),
                                    builder.icmp_signed(">=", normindex, listproxy.length)),
                        IndexError("index out of bounds"))

        at = builder.add(listproxy.whence, builder.mul(listproxy.stride, normindex))
        return generate(context, builder, listtpe.generator.content, listproxy.baggage, listproxy.ptrs, listproxy.lens, at)

    @numba.extending.lower_builtin("getitem", ListProxyNumbaType, numba.types.SliceType)
    def listproxy_getitem_slice(context, builder, sig, args):
        listtpe, indextpe = sig.args
        listval, indexval = args

        sliceproxy = context.make_helper(builder, indextpe, indexval)
        listproxy = numba.cgutils.create_struct_proxy(listtpe)(context, builder, value=listval)
        slicedlistproxy = numba.cgutils.create_struct_proxy(listtpe)(context, builder)
        if listtpe.generator.schema.nullable:
            raise_exception(context,
                            builder,
                            builder.icmp_signed("==", listproxy.ptrs, llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))),
                            TypeError("'NoneType' object has no attribute '__getitem__'"))

        numba.targets.slicing.guard_invalid_slice(context, builder, indextpe, sliceproxy)
        numba.targets.slicing.fix_slice(builder, sliceproxy, listproxy.length)

        slicedlistproxy.baggage = listproxy.baggage
        slicedlistproxy.ptrs = listproxy.ptrs
        slicedlistproxy.lens = listproxy.lens
        slicedlistproxy.whence = sliceproxy.start
        slicedlistproxy.stride = sliceproxy.step
        slicedlistproxy.length = numba.targets.slicing.get_slice_length(builder, sliceproxy)

        return slicedlistproxy._getvalue()

    @numba.extending.lower_builtin("is", ListProxyNumbaType, ListProxyNumbaType)
    def listproxytype_is(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.id == rtype.generator.id:
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            lbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=lproxy.baggage)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            rbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=rproxy.baggage)
            return all_(builder, [
                builder.icmp_signed("==", lbaggage.arrays, rbaggage.arrays),
                builder.icmp_signed("==", lproxy.whence, rproxy.whence),
                builder.icmp_signed("==", lproxy.stride, rproxy.stride),
                builder.icmp_signed("==", lproxy.length, rproxy.length)
                ])
        else:
            return literal_boolean(False)

    @numba.typing.templates.infer
    class ListProxyEq(ProxyCompare):
        key = "=="

    @numba.typing.templates.infer
    class ListProxyEq(ProxyCompare):
        key = "!="

    @numba.extending.lower_builtin("==", ListProxyNumbaType, ListProxyNumbaType)
    def listproxy_eq(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.schema.copy(nullable=False) == rtype.generator.schema.copy(nullable=False):
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            same_size = builder.icmp_signed("==", lproxy.length, rproxy.length)
            out_ptr = numba.cgutils.alloca_once_value(builder, same_size)
            with builder.if_then(same_size):
                with numba.cgutils.for_range(builder, lproxy.length) as loop:
                    litem = generate(context, builder, ltype.generator.content, lproxy.baggage, lproxy.ptrs, lproxy.lens, loop.index)
                    ritem = generate(context, builder, rtype.generator.content, rproxy.baggage, rproxy.ptrs, rproxy.lens, loop.index)
                    predicate = context.get_function("==", numba.types.boolean(typeof_generator(ltype.generator.content), typeof_generator(rtype.generator.content)))(builder, (litem, ritem))
                    with builder.if_then(builder.not_(predicate)):
                        builder.store(literal_boolean(False), out_ptr)
                        loop.do_break()
            return builder.load(out_ptr)
        else:
            return literal_boolean(False)

    @numba.extending.lower_builtin("!=", ListProxyNumbaType, ListProxyNumbaType)
    def listproxy_ne(context, builder, sig, args):
        return builder.not_(listproxy_eq(context, builder, sig, args))

    @numba.typing.templates.infer
    class ListProxyIn(numba.typing.templates.AbstractTemplate):
        key = "in"
        def generic(self, args, kwds):
            item, container = args
            if isinstance(container, ListProxyNumbaType):
                if isinstance(item, ProxyNumbaType) and item.generator.schema.copy(nullable=False) == container.generator.schema.content.copy(nullable=False):
                    return numba.types.boolean(item, container)
                elif isinstance(item, primtypes) and isinstance(container.generator.schema.content, oamap.schema.Primitive):
                    return numba.types.boolean(typeof_generator(container.generator.content), container)

    @numba.extending.lower_builtin("in", numba.types.Boolean, ListProxyNumbaType)
    @numba.extending.lower_builtin("in", numba.types.Integer, ListProxyNumbaType)
    @numba.extending.lower_builtin("in", numba.types.Float, ListProxyNumbaType)
    @numba.extending.lower_builtin("in", numba.types.Complex, ListProxyNumbaType)
    @numba.extending.lower_builtin("in", numba.types.npytypes.CharSeq, ListProxyNumbaType)
    @numba.extending.lower_builtin("in", ProxyNumbaType, ListProxyNumbaType)
    def listproxy_in(context, builder, sig, args):
        itemtpe, listtpe = sig.args
        itemval, listval = args
        listproxy = numba.cgutils.create_struct_proxy(listtpe)(context, builder, value=listval)
        out_ptr = numba.cgutils.alloca_once_value(builder, literal_boolean(False))
        with numba.cgutils.for_range(builder, listproxy.length) as loop:
            listitem = generate(context, builder, listtpe.generator.content, listproxy.baggage, listproxy.ptrs, listproxy.lens, loop.index)
            predicate = context.get_function("==", numba.types.boolean(itemtpe, typeof_generator(listtpe.generator.content)))(builder, (itemval, listitem))
            with builder.if_then(predicate):
                builder.store(literal_boolean(True), out_ptr)
                loop.do_break()
        return builder.load(out_ptr)

    @numba.extending.unbox(ListProxyNumbaType)
    def unbox_listproxy(typ, obj, c):
        generator_obj = c.pyapi.object_getattr_string(obj, "_generator")
        arrays_obj = c.pyapi.object_getattr_string(obj, "_arrays")
        cache_obj = c.pyapi.object_getattr_string(obj, "_cache")
        whence_obj = c.pyapi.object_getattr_string(obj, "_whence")
        stride_obj = c.pyapi.object_getattr_string(obj, "_stride")
        length_obj = c.pyapi.object_getattr_string(obj, "_length")

        listproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
        listproxy.baggage, listproxy.ptrs, listproxy.lens = unbox_baggage(c.context, c.builder, c.pyapi, generator_obj, arrays_obj, cache_obj)
        listproxy.whence = c.pyapi.long_as_longlong(whence_obj)
        listproxy.stride = c.pyapi.long_as_longlong(stride_obj)
        listproxy.length = c.pyapi.long_as_longlong(length_obj)

        c.pyapi.decref(generator_obj)
        c.pyapi.decref(arrays_obj)
        c.pyapi.decref(cache_obj)
        c.pyapi.decref(whence_obj)
        c.pyapi.decref(stride_obj)
        c.pyapi.decref(length_obj)

        is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return numba.extending.NativeValue(listproxy._getvalue(), is_error=is_error)

    @numba.extending.box(ListProxyNumbaType)
    def box_listproxy(typ, val, c):
        listproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        whence_obj = c.pyapi.long_from_longlong(listproxy.whence)
        stride_obj = c.pyapi.long_from_longlong(listproxy.stride)
        length_obj = c.pyapi.long_from_longlong(listproxy.length)

        listproxy_cls = c.pyapi.unserialize(c.pyapi.serialize_object(oamap.proxy.ListProxy))
        generator_obj, arrays_obj, cache_obj = box_baggage(c.context, c.builder, c.pyapi, typ.generator, listproxy.baggage)
        out = c.pyapi.call_function_objargs(listproxy_cls, (generator_obj, arrays_obj, cache_obj, whence_obj, stride_obj, length_obj))

        c.pyapi.decref(listproxy_cls)

        return out

    ################################################################ ListProxyIterator

    class ListProxyIteratorType(numba.types.common.SimpleIteratorType):
        def __init__(self, listproxytype):
            self.listproxy = listproxytype
            super(ListProxyIteratorType, self).__init__("iter({0})".format(listproxytype.name), typeof_generator(listproxytype.generator.content))

    @numba.datamodel.registry.register_default(ListProxyIteratorType)
    class ListProxyIteratorModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("index", numba.types.EphemeralPointer(numba.types.int64)),
                       ("listproxy", fe_type.listproxy)]
            super(ListProxyIteratorModel, self).__init__(dmm, fe_type, members)

    @numba.typing.templates.infer
    class ListProxy_getiter(numba.typing.templates.AbstractTemplate):
        key = "getiter"
        def generic(self, args, kwds):
            if len(args) == 1:
                objtyp, = args
                if isinstance(objtyp, ListProxyNumbaType):
                    return numba.typing.templates.signature(ListProxyIteratorType(objtyp), objtyp)

    @numba.extending.lower_builtin("getiter", ListProxyNumbaType)
    def listproxy_getiter(context, builder, sig, args):
        listtpe, = sig.args
        listval, = args

        iterobj = context.make_helper(builder, sig.return_type)
        iterobj.index = numba.cgutils.alloca_once_value(builder, literal_int64(0))
        iterobj.listproxy = listval

        if context.enable_nrt:
            context.nrt.incref(builder, listtpe, listval)

        return numba.targets.imputils.impl_ret_new_ref(context, builder, sig.return_type, iterobj._getvalue())

    @numba.extending.lower_builtin("iternext", ListProxyIteratorType)
    @numba.targets.imputils.iternext_impl
    def listproxy_iternext(context, builder, sig, args, result):
        itertpe, = sig.args
        iterval, = args
        iterproxy = context.make_helper(builder, itertpe, value=iterval)
        listproxy = numba.cgutils.create_struct_proxy(itertpe.listproxy)(context, builder, value=iterproxy.listproxy)

        index = builder.load(iterproxy.index)
        is_valid = builder.icmp_signed("<", index, listproxy.length)
        result.set_valid(is_valid)

        with builder.if_then(is_valid, likely=True):
            at = builder.add(listproxy.whence, builder.mul(listproxy.stride, index))
            result.yield_(generate(context, builder, itertpe.listproxy.generator.content, listproxy.baggage, listproxy.ptrs, listproxy.lens, at))
            nextindex = numba.cgutils.increment_index(builder, index)
            builder.store(nextindex, iterproxy.index)

    ################################################################ UnionProxy

    class UnionProxyNumbaType(ProxyNumbaType):
        def __init__(self, generator):
            self.generator = generator
            super(UnionProxyNumbaType, self).__init__(name="OAMap-UnionProxy-" + self.generator.id)

    class SyntheticUnion(UnionProxyNumbaType):
        class SyntheticGenerator(object): pass
        def __init__(self, generators):
            generator = SyntheticUnion.SyntheticGenerator()
            generator.id = " ".join(x.id for x in generators)
            generator.possibilities = generators
            generator.schema = oamap.schema.Union([x.schema for x in generators])
            super(SyntheticUnion, self).__init__(generator)

    @numba.extending.register_model(UnionProxyNumbaType)
    class UnionProxyModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("baggage", baggagetype),
                       ("ptrs", numba.types.voidptr),
                       ("lens", numba.types.voidptr),
                       ("tag", numba.types.int64),
                       ("offset", numba.types.int64)]
            super(UnionProxyModel, self).__init__(dmm, fe_type, members)

    @numba.extending.infer_getattr
    class UnionProxyAttribute(numba.typing.templates.AttributeTemplate):
        key = UnionProxyNumbaType
        def generic_resolve(self, typ, attr):
            if all(isinstance(x, oamap.generator.RecordGenerator) for x in typ.generator.possibilities):
                allout = None
                recordproxyattribute = RecordProxyAttribute(self.context)
                for x in typ.generator.possibilities:
                    try:
                        out = recordproxyattribute.generic_resolve(typeof_generator(x), attr)
                    except AttributeError:
                        raise AttributeError("not all Records of {0} have a {1} attribute".format(typ.generator.shema, repr(attr)))
                    else:
                        if allout is None:
                            allout = out
                        else:
                            allout = allout.unify(self.context, out)
                            if allout is None:
                                raise TypeError("not all Records of {0} yield equivalent types for the {1} attribute".format(typ.generator.schema, repr(attr)))
                return allout

    @numba.extending.lower_getattr_generic(UnionProxyNumbaType)
    def unionproxy_getattr(context, builder, typ, val, attr):
        unifiedtype = UnionProxyAttribute(context).generic_resolve(typ, attr)
        unionproxy = numba.cgutils.create_struct_proxy(typ)(context, builder, value=val)
        if typ.generator.schema.nullable:
            raise_exception(context,
                            builder,
                            builder.icmp_signed("==", unionproxy.ptrs, llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))),
                            TypeError("'NoneType' object has no attribute {0}".format(repr(attr))))

        if all(isinstance(x, oamap.generator.RecordGenerator) for x in typ.generator.possibilities):        
            out_ptr = numba.cgutils.alloca_once(builder, context.get_value_type(unifiedtype))

            bbelse = builder.append_basic_block("switch.else")
            bbend = builder.append_basic_block("switch.end")
            switch = builder.switch(unionproxy.tag, bbelse)

            with builder.goto_block(bbelse):
                context.call_conv.return_user_exc(builder, RuntimeError, ("tag out of bounds for union",))

            with builder.goto_block(bbend):
                pass

            for datatag, datagen in enumerate(typ.generator.possibilities):
                ki = literal_int64(datatag)
                bbi = builder.append_basic_block("switch.{0}".format(datatag))
                switch.add_case(ki, bbi)
                with builder.goto_block(bbi):
                    recordval = generate(context, builder, datagen, unionproxy.baggage, unionproxy.ptrs, unionproxy.lens, unionproxy.offset)
                    attrval = recordproxy_getattr(context, builder, typeof_generator(datagen), recordval, attr)
                    convertedval = context.cast(builder, attrval, typeof_generator(datagen.fields[attr]), unifiedtype)
                    builder.store(convertedval, out_ptr)
                    builder.branch(bbend)

            builder.position_at_end(bbend)
            return numba.targets.imputils.impl_ret_borrowed(context, builder, unifiedtype, builder.load(out_ptr))

        else:
            raise AssertionError

    @numba.extending.lower_builtin("is", UnionProxyNumbaType, UnionProxyNumbaType)
    def unionproxytype_is(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.id == rtype.generator.id:
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            lbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=lproxy.baggage)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            rbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=rproxy.baggage)
            return all_(builder, [
                builder.icmp_signed("==", lbaggage.arrays, rbaggage.arrays),
                builder.icmp_signed("==", lproxy.tag, rproxy.tag),
                builder.icmp_signed("==", lproxy.offset, rproxy.offset)
                ])
        else:
            return literal_boolean(False)

    # NOTE: untested until we have SyntheticUnions
    @numba.extending.lower_builtin("is", UnionProxyNumbaType, ProxyNumbaType)
    def unionproxytype_is_left(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        for datatag, datatype in ltype.generator.possibilities:
            if datatype.generator.id == rtype.generator.id:
                lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
                lbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=lproxy.baggage)
                rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
                rbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=rproxy.baggage)
                if isinstance(rtype, ListProxyNumbaType):
                    rindex = rproxy.whence
                elif isinstance(rtype, RecordProxyNumbaType):
                    rindex = rproxy.index
                elif isinstance(rtype, TupleProxyNumbaType):
                    rindex = rproxy.index
                return all_(builder, [
                    builder.icmp_signed("==", lbaggage.arrays, rbaggage.arrays),
                    builder.icmp_signed("==", lproxy.tag, literal_int64(datatag)),
                    builder.icmp_signed("==", lproxy.offset, rindex)
                    ])
        return literal_boolean(False)

    @numba.extending.lower_builtin("is", ProxyNumbaType, UnionProxyNumbaType)
    def unionproxytype_is_right(context, builder, sig, args):
        rettype, (ltype, rtype) = sig.return_type, sig.args
        lval, rval = args
        # reverse the order and use the above
        return unionproxytype_is_left(context, builder, rettype(rtype, ltype), (rval, lval))

    @numba.typing.templates.infer
    class UnionProxyEq(ProxyCompare):
        key = "=="

    @numba.typing.templates.infer
    class UnionProxyEq(ProxyCompare):
        key = "!="

    @numba.extending.lower_builtin("==", UnionProxyNumbaType, UnionProxyNumbaType)
    def unionproxy_eq(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
        rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
        out_ptr = numba.cgutils.alloca_once_value(builder, literal_boolean(False))
        for li, lgen in enumerate(ltype.generator.possibilities):
            for ri, rgen in enumerate(rtype.generator.possibilities):
                if lgen.schema.copy(nullable=False) == rgen.schema.copy(nullable=False):
                    with builder.if_then(builder.and_(
                        builder.icmp_signed("==", lproxy.tag, literal_int(li, ltype.generator.tagdtype.itemsize)),
                        builder.icmp_signed("==", rproxy.tag, literal_int(ri, rtype.generator.tagdtype.itemsize)))):
                        ldata = generate(context, builder, lgen, lproxy.baggage, lproxy.ptrs, lproxy.lens, lproxy.offset)
                        rdata = generate(context, builder, rgen, rproxy.baggage, rproxy.ptrs, rproxy.lens, rproxy.offset)
                        with builder.if_then(context.get_function("==", numba.types.boolean(typeof_generator(lgen), typeof_generator(rgen)))(builder, (ldata, rdata))):
                            builder.store(literal_boolean(True), out_ptr)
                            # TODO: also break to the end
        return builder.load(out_ptr)

    @numba.extending.lower_builtin("!=", UnionProxyNumbaType, UnionProxyNumbaType)
    def unionproxy_ne(context, builder, sig, args):
        return builder.not_(unionproxy_eq(context, builder, sig, args))

    @numba.extending.lower_builtin("==", UnionProxyNumbaType, ProxyNumbaType)
    def unionproxytype_eq_left(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
        rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
        out_ptr = numba.cgutils.alloca_once_value(builder, literal_boolean(False))
        for li, lgen in enumerate(ltype.generator.possibilities):
            if lgen.schema.copy(nullable=False) == rtype.generator.schema.copy(nullable=False):
                with builder.if_then(builder.icmp_signed("==", lproxy.tag, literal_int(li, ltype.generator.tagdtype.itemsize))):
                    ldata = generate(context, builder, lgen, lproxy.baggage, lproxy.ptrs, lproxy.lens, lproxy.offset)
                    rdata = generate(context, builder, rtype.generator, rproxy.baggage, rproxy.ptrs, rproxy.lens, rproxy.offset)
                    with builder.if_then(context.get_function("==", numba.types.boolean(typeof_generator(lgen), rtype))(builder, (ldata, rdata))):
                        builder.store(literal_boolean(True), out_ptr)
                        # TODO: also break to the end
        return builder.load(out_ptr)

    @numba.extending.lower_builtin("==", ProxyNumbaType, UnionProxyNumbaType)
    def unionproxytype_eq_right(context, builder, sig, args):
        rettype, (ltype, rtype) = sig.return_type, sig.args
        lval, rval = args
        # reverse the order and use the above
        return unionproxytype_eq_left(context, builder, rettype(rtype, ltype), (rval, lval))

    @numba.extending.lower_builtin("==", UnionProxyNumbaType, numba.types.Boolean)
    @numba.extending.lower_builtin("==", UnionProxyNumbaType, numba.types.Integer)
    @numba.extending.lower_builtin("==", UnionProxyNumbaType, numba.types.Float)
    @numba.extending.lower_builtin("==", UnionProxyNumbaType, numba.types.Complex)
    @numba.extending.lower_builtin("==", UnionProxyNumbaType, numba.types.npytypes.CharSeq)
    def unionproxytype_eq_left_primitive(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
        out_ptr = numba.cgutils.alloca_once_value(builder, literal_boolean(False))
        for li, lgen in enumerate(ltype.generator.possibilities):
            if isinstance(lgen.schema, oamap.schema.Primitive) and numba.from_dtype(lgen.schema.dtype) == rtype:
                with builder.if_then(builder.icmp_signed("==", lproxy.tag, literal_int(li, ltype.generator.tagdtype.itemsize))):
                    ldata = generate(context, builder, lgen, lproxy.baggage, lproxy.ptrs, lproxy.lens, lproxy.offset)
                    with builder.if_then(context.get_function("==", numba.types.boolean(typeof_generator(lgen), rtype))(builder, (ldata, rval))):
                        builder.store(literal_boolean(True), out_ptr)
        return builder.load(out_ptr)

    @numba.extending.lower_builtin("==", numba.types.Boolean, UnionProxyNumbaType)
    @numba.extending.lower_builtin("==", numba.types.Integer, UnionProxyNumbaType)
    @numba.extending.lower_builtin("==", numba.types.Float, UnionProxyNumbaType)
    @numba.extending.lower_builtin("==", numba.types.Complex, UnionProxyNumbaType)
    @numba.extending.lower_builtin("==", numba.types.npytypes.CharSeq, UnionProxyNumbaType)
    def unionproxytype_eq_right_primitive(context, builder, sig, args):
        rettype, (ltype, rtype) = sig.return_type, sig.args
        lval, rval = args
        # reverse the order and use the above
        return unionproxytype_eq_left_primitive(context, builder, rettype(rtype, ltype), (rval, lval))

    # FIXME: untested
    @numba.extending.lower_cast(UnionProxyNumbaType, numba.types.Boolean)
    @numba.extending.lower_cast(UnionProxyNumbaType, numba.types.Integer)
    @numba.extending.lower_cast(UnionProxyNumbaType, numba.types.Float)
    @numba.extending.lower_cast(UnionProxyNumbaType, numba.types.Complex)
    @numba.extending.lower_cast(UnionProxyNumbaType, numba.types.npytypes.CharSeq)
    def unionproxy_to_primitive(context, builder, fromty, toty, val):
        unionproxy = numba.cgutils.create_struct_proxy(fromty)(context, builder, value=val)
        out_ptr = numba.cgutils.alloca_once(builder, toty)
        filled_ptr = numba.cgutils.alloca_once_value(builder, literal_boolean(False))

        for tag, gem in enumerate(fromty.generator.possibilities):
            if isinstance(gen.schema, oamap.schema.Primitive):
                with builder.if_then(builder.icmp_signed("==", unionproxy.tag, literal_int(tag, ltype.generator.tagdtype.itemsize))):
                    builder.store(context.cast(builder, generate(content, builder, gen, unionproxy.baggage, unionproxy.ptrs, unionproxy.lens, unionproxy.offset), typeof_gen(gen), toty), out_ptr)
                    builder.store(literal_boolean(True), filled_ptr)

        raise_exception(context,
                        builder,
                        builder.not_(builder.load(filled_ptr)),
                        TypeError("cannot cast all members of {0} to {1}".format(fromty, toty)))

        return builder.load(out_ptr)

    @numba.extending.box(UnionProxyNumbaType)
    def box_unionproxy(typ, val, c):
        unionproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        offset_obj = c.pyapi.long_from_longlong(unionproxy.offset)

        generator_obj, arrays_obj, cache_obj = box_baggage(c.context, c.builder, c.pyapi, typ.generator, unionproxy.baggage)
        possibilities_obj = c.pyapi.object_getattr_string(generator_obj, "possibilities")
        possibility_obj = c.pyapi.list_getitem(possibilities_obj, c.context.cast(c.builder, unionproxy.tag, numba.types.int64, numba.types.intp))
        generate_fcn = c.pyapi.object_getattr_string(possibility_obj, "_generate")

        return c.pyapi.call_function_objargs(generate_fcn, (arrays_obj, offset_obj, cache_obj))

    ################################################################ RecordProxy

    class RecordProxyNumbaType(ProxyNumbaType):
        def __init__(self, generator):
            self.generator = generator
            super(RecordProxyNumbaType, self).__init__(name="OAMap-RecordProxy-" + self.generator.id)

    @numba.extending.register_model(RecordProxyNumbaType)
    class RecordProxyModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("baggage", baggagetype),
                       ("ptrs", numba.types.voidptr),
                       ("lens", numba.types.voidptr),
                       ("index", numba.types.int64)]
            super(RecordProxyModel, self).__init__(dmm, fe_type, members)

    numba.extending.make_attribute_wrapper(RecordProxyNumbaType, "index", "_index")

    @numba.extending.infer_getattr
    class RecordProxyAttribute(numba.typing.templates.AttributeTemplate):
        key = RecordProxyNumbaType
        def generic_resolve(self, typ, attr):
            fieldgenerator = typ.generator.fields.get(attr, None)
            if fieldgenerator is not None:
                return typeof_generator(fieldgenerator)
            else:
                raise AttributeError("{0} object has no attribute {1}".format(repr("Record" if typ.generator.name is None else typ.generator.name), repr(attr)))

    @numba.extending.lower_getattr_generic(RecordProxyNumbaType)
    def recordproxy_getattr(context, builder, typ, val, attr):
        recordproxy = numba.cgutils.create_struct_proxy(typ)(context, builder, value=val)
        if typ.generator.schema.nullable:
            raise_exception(context,
                            builder,
                            builder.icmp_signed("==", recordproxy.ptrs, llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))),
                            TypeError("'NoneType' object has no attribute {0}".format(repr(attr))))
        return generate(context, builder, typ.generator.fields[attr], recordproxy.baggage, recordproxy.ptrs, recordproxy.lens, recordproxy.index)

    @numba.extending.lower_builtin("is", RecordProxyNumbaType, RecordProxyNumbaType)
    def recordproxytype_is(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.id == rtype.generator.id:
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            lbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=lproxy.baggage)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            rbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=rproxy.baggage)
            return all_(builder, [
                builder.icmp_signed("==", lbaggage.arrays, rbaggage.arrays),
                builder.icmp_signed("==", lproxy.index, rproxy.index)
                ])
        else:
            return literal_boolean(False)

    @numba.typing.templates.infer
    class RecordProxyEq(ProxyCompare):
        key = "=="

    @numba.typing.templates.infer
    class RecordProxyEq(ProxyCompare):
        key = "!="

    @numba.extending.lower_builtin("==", RecordProxyNumbaType, RecordProxyNumbaType)
    def recordproxy_eq(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.schema.copy(nullable=False) == rtype.generator.schema.copy(nullable=False):
            predicates = []
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            lbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=lproxy.baggage)
            rbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=rproxy.baggage)
            for attr in ltype.generator.schema.fields:
                lfieldgen = ltype.generator.fields[attr]
                rfieldgen = rtype.generator.fields[attr]
                lfield = generate(context, builder, lfieldgen, lproxy.baggage, lproxy.ptrs, lproxy.lens, lproxy.index)
                rfield = generate(context, builder, rfieldgen, rproxy.baggage, rproxy.ptrs, rproxy.lens, rproxy.index)
                predicates.append(context.get_function("==", numba.types.boolean(typeof_generator(lfieldgen), typeof_generator(rfieldgen)))(builder, (lfield, rfield)))
            return all_(builder, predicates)
        else:
            return literal_boolean(False)

    @numba.extending.lower_builtin("!=", RecordProxyNumbaType, RecordProxyNumbaType)
    def recordproxy_ne(context, builder, sig, args):
        return builder.not_(recordproxy_eq(context, builder, sig, args))

    @numba.extending.unbox(RecordProxyNumbaType)
    def unbox_recordproxy(typ, obj, c):
        generator_obj = c.pyapi.object_getattr_string(obj, "_generator")
        arrays_obj = c.pyapi.object_getattr_string(obj, "_arrays")
        cache_obj = c.pyapi.object_getattr_string(obj, "_cache")
        index_obj = c.pyapi.object_getattr_string(obj, "_index")

        recordproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
        recordproxy.baggage, recordproxy.ptrs, recordproxy.lens = unbox_baggage(c.context, c.builder, c.pyapi, generator_obj, arrays_obj, cache_obj)
        recordproxy.index = c.pyapi.long_as_longlong(index_obj)

        c.pyapi.decref(generator_obj)
        c.pyapi.decref(arrays_obj)
        c.pyapi.decref(cache_obj)
        c.pyapi.decref(index_obj)

        is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return numba.extending.NativeValue(recordproxy._getvalue(), is_error=is_error)

    @numba.extending.box(RecordProxyNumbaType)
    def box_recordproxy(typ, val, c):
        recordproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        index_obj = c.pyapi.long_from_longlong(recordproxy.index)

        recordproxy_cls = c.pyapi.unserialize(c.pyapi.serialize_object(oamap.proxy.RecordProxy))
        generator_obj, arrays_obj, cache_obj = box_baggage(c.context, c.builder, c.pyapi, typ.generator, recordproxy.baggage)
        out = c.pyapi.call_function_objargs(recordproxy_cls, (generator_obj, arrays_obj, cache_obj, index_obj))

        c.pyapi.decref(recordproxy_cls)

        return out

    ################################################################ TupleProxy

    class TupleProxyNumbaType(ProxyNumbaType):
        def __init__(self, generator):
            self.generator = generator
            super(TupleProxyNumbaType, self).__init__(name="OAMap-TupleProxy-" + self.generator.id)

    @numba.extending.register_model(TupleProxyNumbaType)
    class TupleProxyModel(numba.datamodel.models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("baggage", baggagetype),
                       ("ptrs", numba.types.voidptr),
                       ("lens", numba.types.voidptr),
                       ("index", numba.types.int64)]
            super(TupleProxyModel, self).__init__(dmm, fe_type, members)

    numba.extending.make_attribute_wrapper(TupleProxyNumbaType, "index", "_index")

    @numba.extending.type_callable(len)
    def tupleproxy_len_type(context):
        def typer(tupleproxy):
            if isinstance(tupleproxy, TupleProxyNumbaType):
                return numba.types.int64   # verified len type
        return typer

    @numba.extending.lower_builtin(len, TupleProxyNumbaType)
    def tupleproxy_len(context, builder, sig, args):
        listtpe, = sig.args
        return literal_int64(len(listtpe.generator.types))

    @numba.typing.templates.infer
    class TupleProxyGetItem(numba.typing.templates.AbstractTemplate):
        key = "static_getitem"
        def generic(self, args, kwds):
            if len(args) == 2:
                tpe, idx = args
                if isinstance(tpe, TupleProxyNumbaType) and isinstance(idx, int):
                    if idx < 0:
                        normindex = idx + len(tpe.generator.types)
                    else:
                        normindex = idx
                    if 0 <= normindex < len(tpe.generator.types):
                        return typeof_generator(tpe.generator.types[normindex])
                    else:
                        raise IndexError("item {0} out of range for type {1}".format(idx, tpe.generator.schema))

    @numba.extending.lower_builtin("static_getitem", TupleProxyNumbaType, numba.types.Const)
    def tupleproxy_static_getitem(context, builder, sig, args):
        tupletpe, _ = sig.args
        tupleval, idx = args
        if isinstance(idx, int):
            if idx < 0:
                normindex = idx + len(tupletpe.generator.types)
            else:
                normindex = idx
            tupleproxy = numba.cgutils.create_struct_proxy(tupletpe)(context, builder, value=tupleval)
            if tupletpe.generator.schema.nullable:
                raise_exception(context,
                                builder,
                                builder.icmp_signed("==", tupleproxy.ptrs, llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.voidptr))),
                                TypeError("'NoneType' object has no attribute '__getitem__'"))
            return generate(context, builder, tupletpe.generator.types[normindex], tupleproxy.baggage, tupleproxy.ptrs, tupleproxy.lens, tupleproxy.index)

    @numba.extending.lower_builtin("is", TupleProxyNumbaType, TupleProxyNumbaType)
    def tupleproxytype_is(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.id == rtype.generator.id:
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            lbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=lproxy.baggage)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            rbaggage = numba.cgutils.create_struct_proxy(baggagetype)(context, builder, value=rproxy.baggage)
            return all_(builder, [
                builder.icmp_signed("==", lbaggage.arrays, rbaggage.arrays),
                builder.icmp_signed("==", lproxy.index, rproxy.index)
                ])
        else:
            return literal_boolean(False)
            
    @numba.typing.templates.infer
    class TupleProxyEq(ProxyCompare):
        key = "=="

    @numba.typing.templates.infer
    class TupleProxyEq(ProxyCompare):
        key = "!="

    @numba.extending.lower_builtin("==", TupleProxyNumbaType, TupleProxyNumbaType)
    def tupleproxy_eq(context, builder, sig, args):
        ltype, rtype = sig.args
        lval, rval = args
        if ltype.generator.schema.copy(nullable=False) == rtype.generator.schema.copy(nullable=False):
            predicates = []
            lproxy = numba.cgutils.create_struct_proxy(ltype)(context, builder, value=lval)
            rproxy = numba.cgutils.create_struct_proxy(rtype)(context, builder, value=rval)
            for i in range(len(ltype.generator.schema.types)):
                lfieldgen = ltype.generator.types[i]
                rfieldgen = rtype.generator.types[i]
                lfield = generate(context, builder, lfieldgen, lproxy.baggage, lproxy.ptrs, lproxy.lens, lproxy.index)
                rfield = generate(context, builder, rfieldgen, rproxy.baggage, rproxy.ptrs, rproxy.lens, rproxy.index)
                predicates.append(context.get_function("==", numba.types.boolean(typeof_generator(lfieldgen), typeof_generator(rfieldgen)))(builder, (lfield, rfield)))
            return all_(builder, predicates)
        else:
            return literal_boolean(False)

    @numba.extending.lower_builtin("!=", TupleProxyNumbaType, TupleProxyNumbaType)
    def tupleproxy_ne(context, builder, sig, args):
        return builder.not_(tupleproxy_eq(context, builder, sig, args))

    @numba.extending.unbox(TupleProxyNumbaType)
    def unbox_tupleproxy(typ, obj, c):
        generator_obj = c.pyapi.object_getattr_string(obj, "_generator")
        arrays_obj = c.pyapi.object_getattr_string(obj, "_arrays")
        cache_obj = c.pyapi.object_getattr_string(obj, "_cache")
        index_obj = c.pyapi.object_getattr_string(obj, "_index")

        tupleproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
        tupleproxy.baggage, tupleproxy.ptrs, tupleproxy.lens = unbox_baggage(c.context, c.builder, c.pyapi, generator_obj, arrays_obj, cache_obj)
        tupleproxy.index = c.pyapi.long_as_longlong(index_obj)

        c.pyapi.decref(generator_obj)
        c.pyapi.decref(arrays_obj)
        c.pyapi.decref(cache_obj)
        c.pyapi.decref(index_obj)

        is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return numba.extending.NativeValue(tupleproxy._getvalue(), is_error=is_error)

    @numba.extending.box(TupleProxyNumbaType)
    def box_tupleproxy(typ, val, c):
        tupleproxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        index_obj = c.pyapi.long_from_longlong(tupleproxy.index)

        tupleproxy_cls = c.pyapi.unserialize(c.pyapi.serialize_object(oamap.proxy.TupleProxy))
        generator_obj, arrays_obj, cache_obj = box_baggage(c.context, c.builder, c.pyapi, typ.generator, tupleproxy.baggage)
        out = c.pyapi.call_function_objargs(tupleproxy_cls, (generator_obj, arrays_obj, cache_obj, index_obj))

        c.pyapi.decref(tupleproxy_cls)

        return out
