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

import sys
import datetime
import os

import numpy

import oamap.proxy
from oamap.util import OrderedDict

if sys.version_info[0] > 2:
    basestring = str

# for sources that perform specialized actions on particular kinds of arrays
class Role(object):
    def __init__(self, name, namespace):
        self.name = name
        self.namespace = namespace
    @property
    def args(self):
        return (self.name, self.namespace)
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, ", ".join(repr(str(x)) for x in self.args))
    def __str__(self):
        return self.name
    def __hash__(self):
        return hash((Role, self.name, self.namespace))
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name and self.namespace == other.namespace
    def __ne__(self, other):
        return not self.__eq__(other)

class NoRole(Role): pass

class MaskRole(Role):
    def __init__(self, name, namespace, others):
        super(MaskRole, self).__init__(name, namespace)
        self.others = others
    @property
    def args(self):
        return (self.name, self.namespace, self.others)

class DataRole(Role): pass

class StartsRole(Role):
    def __init__(self, name, namespace, stops):
        super(StartsRole, self).__init__(name, namespace)
        self.stops = stops
    @property
    def args(self):
        return (self.name, self.namespace, self.stops)

class StopsRole(Role):
    def __init__(self, name, namespace, starts):
        super(StopsRole, self).__init__(name, namespace)
        self.starts = starts
    @property
    def args(self):
        return (self.name, self.namespace, self.starts)

class TagsRole(Role):
    def __init__(self, name, namespace, offsets):
        super(TagsRole, self).__init__(name, namespace)
        self.offsets = offsets
    @property
    def args(self):
        return (self.name, self.namespace, self.offsets)

class OffsetsRole(Role):
    def __init__(self, name, namespace, tags):
        super(OffsetsRole, self).__init__(name, namespace)
        self.tags = tags
    @property
    def args(self):
        return (self.name, self.namespace, self.tags)

class PositionsRole(Role): pass

# base class of all runtime-object generators (one for each type)
class Generator(object):
    _starttime = datetime.datetime.now().isoformat()
    _startpid = os.getpid()
    _nextid = 0

    @staticmethod
    def nextid():
        out = "{0}-pid{1}-{2}".format(Generator._starttime, Generator._startpid, Generator._nextid)
        Generator._nextid += 1
        return out

    def __init__(self, namespace, packing, name, derivedname, schema):
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.derivedname = derivedname
        self.schema = schema

        self.id = self.nextid()
        self._required = False

    def _new(self, memo=None):
        self.id = self.nextid()
        self._required = False

    def fromdata(self, value, pointer_fromequal=False):
        import oamap.fill
        return self(oamap.fill.fromdata(value, generator=self, pointer_fromequal=pointer_fromequal))

    def fromiterdata(self, values, limit=lambda entries, arrayitems, arraybytes: False, pointer_fromequal=False):
        import oamap.fill
        return self(oamap.fill.fromiterdata(values, generator=self, limit=limit, pointer_fromequal=pointer_fromequal))

    def __call__(self, arrays):
        return self._generate(arrays, 0, self._newcache())

    def _getarrays(self, arrays, cache, roles, require_arrays=False):
        if self.packing is not None:
            arrays = self.packing.anchor(arrays)

        if hasattr(arrays, "getall"):
            out = arrays.getall(list(roles))                           # pass on the roles to a source that knows about getall
        else:
            out = dict((name, arrays[str(name)]) for name in roles)    # drop the roles; it's a plain-dict interface

        for name, array in out.items():
            idx, dtype = roles[name]

            if isinstance(array, bytes):
                array = numpy.frombuffer(array, dtype)

            if (require_arrays and not isinstance(array, numpy.ndarray)) or getattr(array, "dtype", dtype) != dtype:
                array = numpy.array(array, dtype=dtype)

            cache[idx] = array

    def _newcache(self):
        return [None] * self._cachelen

    def _clearcache(self, cache, listofarrays, index):
        if 0 <= index < len(listofarrays):
            arrays = listofarrays[index]
            if hasattr(arrays, "close"):
                arrays.close()
        for i in range(len(cache)):
            cache[i] = None

    def _entercompiled(self, arrays, cache, bottomup=True):
        roles = self._togetall(arrays, cache, bottomup, set())
        self._getarrays(arrays, cache, roles, require_arrays=True)

        ptrs = numpy.zeros(self._cachelen, dtype=numpy.intp)
        lens = numpy.zeros(self._cachelen, dtype=numpy.intp)
        for i, x in enumerate(cache):
            if x is not None:
                ptrs[i] = x.ctypes.data
                lens[i] = x.shape[0]

        return ptrs, lens, ptrs.ctypes.data, lens.ctypes.data

    def names(self, namespace=False, idx=False):
        return list(self.iternames(namespace=namespace, idx=idx))

    def namespaces(self):
        return set(ns for n, ns in self.iternames(namespace=True))

    def namedschema(self):
        return self._namedschema({})

    def findbynames(self, schematype, namespace, **names):
        return self._findbynames(schematype, namespace, names, set())

    def case(self, obj):
        return self.schema.case(obj)

    def cast(self, obj):
        return self.schema.cast(obj)

# mix-in for all generators of nullable types
class Masked(object):
    maskdtype = numpy.dtype(numpy.int32)
    maskedvalue = -1

    def __init__(self, mask, maskidx):
        self.mask = mask
        self.maskidx = maskidx

    def _toget(self, arrays, cache):
        others = self.__class__.__bases__[1]._toget(self, arrays, cache)
        out = OrderedDict([(MaskRole(self.mask, self.namespace, others), (self.maskidx, self.maskdtype))])
        out.update(others)
        return out

    def _togetall(self, arrays, cache, bottomup, memo):
        key = (id(self),)
        if key not in memo:
            memo.add(key)
            out = self.__class__.__bases__[1]._togetall(self, arrays, cache, bottomup, memo)
            if self._required and cache[self.maskidx] is None:
                if bottomup:
                    out.update(self._toget(arrays, cache))
                else:
                    out2 = self._toget(arrays, cache)
                    out2.update(out)
                    out = out2
            return out
        else:
            return OrderedDict()

    def _getmask(self, arrays, cache):
        mask = cache[self.maskidx]
        if mask is None:
            self._getarrays(arrays, cache, self._toget(arrays, cache))
            mask = cache[self.maskidx]
        return mask

    def _generate(self, arrays, index, cache):
        value = self._getmask(arrays, cache)[index]
        if value == self.maskedvalue:
            return None
        else:
            # otherwise, the value is the index for compactified data
            return self.__class__.__bases__[1]._generate(self, arrays, value, cache)

    def iternames(self, namespace=False, idx=False):
        out = (self.mask,)
        if namespace:
            out = out + (self.namespace,)
        if idx:
            out = out + (self.maskidx,)
        if len(out) == 1:
            out = out[0]
        yield out

        for x in self.__class__.__bases__[1].iternames(self, namespace=namespace, idx=idx):
            yield x

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        key = (id(self),)
        if key not in memo:
            memo.add(key)
            if cache[self.maskidx] is not None:
                yield self.mask
            for x in self.__class__.__bases__[1].loaded(self, cache, memo):
                yield x

    def required(self, memo=None):
        if memo is None:
            memo = set()
        key = (id(self),)
        if key not in memo:
            memo.add(key)
            if self._required:
                yield self.mask
            for x in self.__class__.__bases__[1].required(self, memo):
                yield x

################################################################ Primitives

class PrimitiveGenerator(Generator):
    def __init__(self, data, dataidx, dtype, namespace, packing, name, derivedname, schema):
        self.data = data
        self.dataidx = dataidx
        self.dtype = dtype
        Generator.__init__(self, namespace, packing, name, derivedname, schema)

    def _toget(self, arrays, cache):
        return OrderedDict([(DataRole(self.data, self.namespace), (self.dataidx, self.dtype))])

    def _togetall(self, arrays, cache, bottomup, memo):
        if id(self) not in memo:
            memo.add(id(self))
            if self._required and cache[self.dataidx] is None:
                return self._toget(arrays, cache)
        return OrderedDict()

    def _getdata(self, arrays, cache):
        data = cache[self.dataidx]
        if data is None:
            self._getarrays(arrays, cache, self._toget(arrays, cache))
            data = cache[self.dataidx]
        return data

    def _generate(self, arrays, index, cache):
        return self._getdata(arrays, cache)[index]

    def _requireall(self, memo=None):
        self._required = True

    def iternames(self, namespace=False, idx=False):
        out = (self.data,)
        if namespace:
            out = out + (self.namespace,)
        if idx:
            out = out + (self.dataidx,)
        if len(out) == 1:
            out = out[0]
        yield out

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if cache[self.dataidx] is not None:
                yield self.data

    def required(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if self._required:
                yield self.data

    def _namedschema(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        if isinstance(self, Masked):
            memo[id(self)] = self.schema.copy(data=self.data, mask=self.mask)
        else:
            memo[id(self)] = self.schema.copy(data=self.data)
        return memo[id(self)]

    def _findbynames(self, schematype, namespace, names, memo):
        if schematype == "Primitive" and namespace == self.namespace and names.get("data", None) == self.data and (not isinstance(self, Masked) or names.get("mask", None) == self.mask):
            return self
        else:
            return None

class MaskedPrimitiveGenerator(Masked, PrimitiveGenerator):
    def __init__(self, mask, maskidx, data, dataidx, dtype, namespace, packing, name, derivedname, schema):
        Masked.__init__(self, mask, maskidx)
        PrimitiveGenerator.__init__(self, data, dataidx, dtype, namespace, packing, name, derivedname, schema)

################################################################ Lists

class ListGenerator(Generator):
    posdtype = numpy.dtype(numpy.int32)

    def __init__(self, starts, startsidx, stops, stopsidx, content, namespace, packing, name, derivedname, schema):
        self.starts = starts
        self.startsidx = startsidx
        self.stops = stops
        self.stopsidx = stopsidx
        self.content = content
        Generator.__init__(self, namespace, packing, name, derivedname, schema)

    def _new(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            super(ListGenerator, self)._new(memo)
            self.content._new(memo)

    def _toget(self, arrays, cache):
        starts = StartsRole(self.starts, self.namespace, None)
        stops = StopsRole(self.stops, self.namespace, None)
        starts.stops = stops
        stops.starts = starts
        return OrderedDict([(starts, (self.startsidx, self.posdtype)), (stops, (self.stopsidx, self.posdtype))])

    def _togetall(self, arrays, cache, bottomup, memo):
        if id(self) not in memo:
            memo.add(id(self))
            out = self.content._togetall(arrays, cache, bottomup, memo)
            if self._required and (cache[self.startsidx] is None or cache[self.stopsidx] is None):
                if bottomup:
                    out.update(self._toget(arrays, cache))
                else:
                    out2 = self._toget(arrays, cache)
                    out2.update(out)
                    out = out2
            return out
        else:
            return OrderedDict()

    def _getstartsstops(self, arrays, cache):
        starts = cache[self.startsidx]
        stops = cache[self.stopsidx]
        if starts is None or stops is None:
            self._getarrays(arrays, cache, self._toget(arrays, cache))
            starts = cache[self.startsidx]
            stops = cache[self.stopsidx]
        return starts, stops

    def _generate(self, arrays, index, cache):
        starts, stops = self._getstartsstops(arrays, cache)
        return oamap.proxy.ListProxy(self, arrays, cache, starts[index], 1, stops[index] - starts[index])

    def _requireall(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            self._required = True
            self.content._requireall(memo)

    def iternames(self, namespace=False, idx=False):
        out1 = (self.starts,)
        out2 = (self.stops,)
        if namespace:
            out1 = out1 + (self.namespace,)
            out2 = out2 + (self.namespace,)
        if idx:
            out1 = out1 + (self.startsidx,)
            out2 = out2 + (self.stopsidx,)
        if len(out1) == 1:
            out1 = out1[0]
            out2 = out2[0]
        yield out1
        yield out2
            
        for x in self.content.iternames(namespace=namespace, idx=idx):
            yield x

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if cache[self.startsidx] is not None:
                yield self.starts
            if cache[self.stopsidx] is not None:
                yield self.stops
            for x in self.content.loaded(cache, memo):
                yield x

    def required(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if self._required:
                yield self.starts
                yield self.stops
            for x in self.content.required(memo):
                yield x

    def _namedschema(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        if isinstance(self, Masked):
            memo[id(self)] = self.schema.copy(starts=self.starts, stops=self.stops, mask=self.mask)
        else:
            memo[id(self)] = self.schema.copy(starts=self.starts, stops=self.stops)
        memo[id(self)].content = self.content._namedschema(memo)
        return memo[id(self)]

    def _findbynames(self, schematype, namespace, names, memo):
        if schematype == "List" and namespace == self.namespace and names.get("starts", None) == self.starts and names.get("stops", None) == self.stops and (not isinstance(self, Masked) or names.get("mask", None) == self.mask):
            return self
        else:
            return self.content._findbynames(schematype, namespace, names, memo)

class MaskedListGenerator(Masked, ListGenerator):
    def __init__(self, mask, maskidx, starts, startsidx, stops, stopsidx, content, namespace, packing, name, derivedname, schema):
        Masked.__init__(self, mask, maskidx)
        ListGenerator.__init__(self, starts, startsidx, stops, stopsidx, content, namespace, packing, name, derivedname, schema)

################################################################ Unions

class UnionGenerator(Generator):
    tagdtype = numpy.dtype(numpy.int8)
    offsetdtype = numpy.dtype(numpy.int32)

    def __init__(self, tags, tagsidx, offsets, offsetsidx, possibilities, namespace, packing, name, derivedname, schema):
        self.tags = tags
        self.tagsidx = tagsidx
        self.offsets = offsets
        self.offsetsidx = offsetsidx
        self.possibilities = possibilities
        Generator.__init__(self, namespace, packing, name, derivedname, schema)

    def _new(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            super(UnionGenerator, self)._new(memo)
            for x in self.possibilities:
                x._new(memo)

    def _toget(self, arrays, cache):
        tags = TagsRole(self.tags, self.namespace, None)
        offsets = OffsetsRole(self.offsets, self.namespace, None)
        tags.offsets = offsets
        offsets.tags = tags
        return OrderedDict([(tags, (self.tagsidx, self.tagdtype)), (offsets, (self.offsetsidx, self.offsetdtype))])

    def _togetall(self, arrays, cache, bottomup, memo):
        if id(self) not in memo:
            memo.add(id(self))
            out = OrderedDict()
            for x in self.possibilities:
                out.update(x._togetall(arrays, cache, bottomup, memo))
            if self._required and (cache[self.tagsidx] is None or cache[self.offsetsidx] is None):
                if bottomup:
                    out.update(self._toget(arrays, cache))
                else:
                    out2 = self._toget(arrays, cache)
                    out2.update(out)
                    out = out2
            return out
        else:
            return OrderedDict()

    def _gettagsoffsets(self, arrays, cache):
        tags = cache[self.tagsidx]
        offsets = cache[self.offsetsidx]
        if tags is None or offsets is None:
            self._getarrays(arrays, cache, self._toget(arrays, cache))
            tags = cache[self.tagsidx]
            offsets = cache[self.offsetsidx]
        return tags, offsets

    def _generate(self, arrays, index, cache):
        tags, offsets = self._gettagsoffsets(arrays, cache)
        return self.possibilities[tags[index]]._generate(arrays, offsets[index], cache)

    def _requireall(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            self._required = True
            for x in self.possibilities:
                x._requireall(memo)

    def iternames(self, namespace=False, idx=False):
        out1 = (self.tags,)
        out2 = (self.offsets,)
        if namespace:
            out1 = out1 + (self.namespace,)
            out2 = out2 + (self.namespace,)
        if idx:
            out1 = out1 + (self.tagsidx,)
            out2 = out2 + (self.offsetsidx,)
        if len(out1) == 1:
            out1 = out1[0]
            out2 = out2[0]
        yield out1
        yield out2

        for x in self.possibilities:
            for y in x.iternames(namespace=namespace, idx=idx):
                yield y

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if cache[self.tagsidx] is not None:
                yield self.tags
            if cache[self.offsetsidx] is not None:
                yield self.offsets
            for possibility in self.possibilities:
                for x in possibility.loaded(cache, memo):
                    yield x

    def required(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if self._required:
                yield self.tags
                yield self.offsets
            for possibility in self.possibilities:
                for x in possibility.required(memo):
                    yield x

    def _namedschema(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        if isinstance(self, Masked):
            memo[id(self)] = self.schema.copy(tags=self.tags, offsets=self.offsets, mask=self.mask)
        else:
            memo[id(self)] = self.schema.copy(tags=self.tags, offsets=self.offsets)
        memo[id(self)].possibilities = [x._namedschema(memo) for x in self.possibilities]
        return memo[id(self)]

    def _findbynames(self, schematype, namespace, names, memo):
        if schematype == "Union" and namespace == self.namespace and names.get("tags", None) == self.tags and names.get("offsets", None) == self.offsets and (not isinstance(self, Masked) or names.get("mask", None) == self.mask):
            return self
        else:
            for possibility in self.possibilities:
                out = possibility._findbynames(schematype, namespace, names, memo)
                if out is not None:
                    return out
            return None

class MaskedUnionGenerator(Masked, UnionGenerator):
    def __init__(self, mask, maskidx, tags, tagsidx, offsets, offsetsidx, possibilities, namespace, packing, name, derivedname, schema):
        Masked.__init__(self, mask, maskidx)
        UnionGenerator.__init__(self, tags, tagsidx, offsets, offsetsidx, possibilities, namespace, packing, name, derivedname, schema)

################################################################ Records

class RecordGenerator(Generator):
    def __init__(self, fields, namespace, packing, name, derivedname, schema):
        self.fields = fields
        Generator.__init__(self, namespace, packing, name, derivedname, schema)

    def _new(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            super(RecordGenerator, self)._new(memo)
            for x in self.fields.values():
                x._new(memo)

    def _toget(self, arrays, cache):
        return OrderedDict()

    def _togetall(self, arrays, cache, bottomup, memo):
        if id(self) not in memo:
            memo.add(id(self))
            out = OrderedDict()
            for x in self.fields.values():
                out.update(x._togetall(arrays, cache, bottomup, memo))
            return out
        else:
            return OrderedDict()

    def _generate(self, arrays, index, cache):
        return oamap.proxy.RecordProxy(self, arrays, cache, index)

    def _requireall(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            self._required = True
            for x in self.fields.values():
                x._requireall(memo)

    def iternames(self, namespace=False, idx=False):
        for x in self.fields.values():
            for y in x.iternames(namespace=namespace, idx=idx):
                yield y

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            for field in self.fields.values():
                for x in field.loaded(cache, memo):
                    yield x

    def required(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            for field in self.fields.values():
                for x in field.required(memo):
                    yield x

    def _namedschema(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        if isinstance(self, Masked):
            memo[id(self)] = self.schema.copy(mask=self.mask)
        else:
            memo[id(self)] = self.schema.copy()
        for n in memo[id(self)].fields:
            memo[id(self)][n] = self.fields[n]._namedschema(memo)
        return memo[id(self)]

    def _findbynames(self, schematype, namespace, names, memo):
        if schematype == "Record" and namespace == self.namespace and (not isinstance(self, Masked) or names.get("mask", None) == self.mask):
            return self
        else:
            for x in self.fields.values():
                out = x._findbynames(schematype, namespace, names, memo)
                if out is not None:
                    return out
            return None

class MaskedRecordGenerator(Masked, RecordGenerator):
    def __init__(self, mask, maskidx, fields, namespace, packing, name, derivedname, schema):
        Masked.__init__(self, mask, maskidx)
        RecordGenerator.__init__(self, fields, namespace, packing, name, derivedname, schema)

################################################################ Tuples

class TupleGenerator(Generator):
    def __init__(self, types, namespace, packing, name, derivedname, schema):
        self.types = types
        Generator.__init__(self, namespace, packing, name, derivedname, schema)

    def _new(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            super(TupleGenerator, self)._new(memo)
            for x in self.types:
                x._new(memo)

    def _toget(self, arrays, cache):
        return OrderedDict()

    def _togetall(self, arrays, cache, bottomup, memo):
        if id(self) not in memo:
            memo.add(id(self))
            out = OrderedDict()
            for x in self.types:
                out.update(x._togetall(arrays, cache, bottomup, memo))
            return out
        else:
            return OrderedDict()

    def _generate(self, arrays, index, cache):
        return oamap.proxy.TupleProxy(self, arrays, cache, index)

    def _requireall(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            self._required = True
            for x in self.types:
                x._requireall(memo)

    def iternames(self, namespace=False, idx=False):
        for x in self.types:
            for y in x.iternames(namespace=namespace, idx=idx):
                yield y

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            for field in self.types:
                for x in field.loaded(cache, memo):
                    yield x

    def required(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            for field in self.types:
                for x in field.required(memo):
                    yield x

    def _namedschema(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        if isinstance(self, Masked):
            memo[id(self)] = self.schema.copy(mask=self.mask)
        else:
            memo[id(self)] = self.schema.copy()
        memo[id(self)].types = tuple(x._namedschema(memo) for x in self.types)
        return memo[id(self)]

    def _findbynames(self, schematype, namespace, names, memo):
        if schematype == "Tuple" and namespace == self.namespace and (not isinstance(self, Masked) or names.get("mask", None) == self.mask):
            return self
        else:
            for x in self.types:
                out = x._findbynames(schematype, namespace, names, memo)
                if out is not None:
                    return out
            return None

class MaskedTupleGenerator(Masked, TupleGenerator):
    def __init__(self, mask, maskidx, types, namespace, packing, name, derivedname, schema):
        Masked.__init__(self, mask, maskidx)
        TupleGenerator.__init__(self, types, namespace, packing, name, derivedname, schema)

################################################################ Pointers

class PointerGenerator(Generator):
    posdtype = numpy.dtype(numpy.int32)

    def __init__(self, positions, positionsidx, target, namespace, packing, name, derivedname, schema):
        self.positions = positions
        self.positionsidx = positionsidx
        self.target = target
        Generator.__init__(self, namespace, packing, name, derivedname, schema)

    def _new(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            super(PointerGenerator, self)._new(memo)
            self.target._new(memo)

    def _toget(self, arrays, cache):
        return OrderedDict([(PositionsRole(self.positions, self.namespace), (self.positionsidx, self.posdtype))])

    def _togetall(self, arrays, cache, bottomup, memo):
        if id(self) not in memo:
            memo.add(id(self))
            out = self.target._togetall(arrays, cache, bottomup, memo)
            if self._required and cache[self.positionsidx] is None:
                if bottomup:
                    out.update(self._toget(arrays, cache))
                else:
                    out2 = self._toget(arrays, cache)
                    out2.update(out)
                    out = out2
            return out
        else:
            return OrderedDict()

    def _getpositions(self, arrays, cache):
        positions = cache[self.positionsidx]
        if positions is None:
            self._getarrays(arrays, cache, self._toget(arrays, cache))
            positions = cache[self.positionsidx]
        return positions

    def _generate(self, arrays, index, cache):
        return self.target._generate(arrays, self._getpositions(arrays, cache)[index], cache)

    def _requireall(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            self._required = True
            self.target._requireall(memo)

    def iternames(self, namespace=False, idx=False):
        out = (self.positions,)
        if namespace:
            out = out + (self.namespace,)
        if idx:
            out = out + (self.positionsidx,)
        if len(out) == 1:
            out = out[0]
        yield out

        if not self._internal:
            for x in self.target.iternames(namespace=namespace, idx=idx):
                yield x

    def loaded(self, cache, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if cache[self.positionsidx] is not None:
                yield self.positions
            for x in self.target.loaded(cache, memo):
                yield x

    def required(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            if self._required:
                yield self.positions
            for x in self.target.required(memo):
                yield x

    def _namedschema(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        if isinstance(self, Masked):
            memo[id(self)] = self.schema.copy(positions=self.positions, mask=self.mask)
        else:
            memo[id(self)] = self.schema.copy(positions=self.positions)
        memo[id(self)].target = self.target._namedschema(memo)
        return memo[id(self)]

    def _findbynames(self, schematype, namespace, names, memo):
        if schematype == "Pointer" and namespace == self.namespace and names.get("positions", None) == self.positions and (not isinstance(self, Masked) or names.get("mask", None) == self.mask):
            return self
        else:
            if id(self) in memo:
                return None
            memo.add(id(self))
            return self.target._findbynames(schematype, namespace, names, memo)

class MaskedPointerGenerator(Masked, PointerGenerator):
    def __init__(self, mask, maskidx, positions, positionsidx, target, namespace, packing, name, derivedname, schema):
        Masked.__init__(self, mask, maskidx)
        PointerGenerator.__init__(self, positions, positionsidx, target, namespace, packing, name, derivedname, schema)

################################################################ for extensions: domain-specific and user

class ExtendedGenerator(Generator):
    # extensions *must* override pattern, _generate, and degenerate, *may* override matches
    pattern = None

    # default implementation: generate a given type of proxy
    proxyclass = None
    def _generate(self, arrays, index, cache):
        return self.proxyclass(self, arrays, cache, index)

    # default implementation: do nothing; generic type is the same as extended type
    def degenerate(self, obj):
        return obj

    def __init__(self, genericclass, *args):
        self.generic = genericclass(*args)

    def _new(self, memo=None):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            super(ExtendedGenerator, self)._new(memo)
            self.generic._new(memo)

    def _toget(self, arrays, cache):
        return self.generic._toget(arrays, cache)

    def _togetall(self, arrays, cache, bottomup, memo):
        return self.generic._togetall(arrays, cache, bottomup, memo)

    @property
    def namespace(self):
        return self.generic.namespace

    @property
    def packing(self):
        return self.generic.packing

    @property
    def name(self):
        return self.generic.name

    @property
    def derivedname(self):
        return self.generic.derivedname

    @property
    def schema(self):
        return self.generic.schema

    def iternames(self, namespace=False, idx=False):
        for x in self.generic.iternames(namespace=namespace, idx=idx):
            yield x

    def loaded(self, cache, memo=None):
        for x in self.generic.loaded(cache, memo):
            yield x

    def required(self, memo=None):
        for x in self.generic.required(memo):
            yield x

    def _namedschema(self, memo):
        return self.generic._namedschema(memo)

    def _findbynames(self, schematype, namespace, names, memo):
        return self.generic._findbynames(schematype, namespace, names, memo)

    @classmethod
    def matches(cls, schema):
        import oamap.schema
        def recurse(pattern, schema):
            if isinstance(pattern, basestring):
                return schema["type"] == "primitive" and schema["dtype"] == pattern
            else:
                assert isinstance(pattern, dict) and "type" in pattern

                if "nullable" in pattern and pattern["nullable"] != schema["nullable"]:
                    return False
                if "name" in pattern and pattern["name"] != schema["name"]:
                    return False
                if "label" in pattern and pattern["label"] != schema["label"]:
                    return False

                if pattern["type"] == "primitive":
                    if "dtype" in pattern and oamap.schema.Primitive._dtype2str(numpy.dtype(pattern["dtype"]), "-") != schema["dtype"]:
                        return False
                    return True

                elif pattern["type"] == "list":
                    if "content" in pattern:
                        return recurse(pattern["content"], schema["content"])
                    return True

                elif pattern["type"] == "union":
                    if "possibilities" in pattern:
                        return len(pattern["possibilities"]) == len(schema["possibilities"]) and all(recurse(x, y) for x, y in zip(pattern["possibilities"], schema["possibilities"]))
                    return True

                elif pattern["type"] == "record":
                    if "fields" in pattern:
                        if isinstance(pattern["fields"], dict):
                            return set(pattern["fields"].keys()).issubset(set(n for n, x in schema["fields"])) and all(recurse(px, [sx for sn, sx in schema["fields"] if pn == sn][0]) for pn, px in pattern["fields"].items())
                        else:
                            return len(pattern["fields"]) == len(schema["fields"]) and all(pn == sn and recurse(px, sx) for (pn, px), (sn, sx) in zip(pattern["fields"], schema["fields"]))
                    return True

                elif pattern["type"] == "tuple":
                    if "types" in pattern:
                        return len(pattern["types"]) == len(schema["types"]) and all(recurse(x, y) for x, y in zip(pattern["types"], schema["types"]))
                    return True

                elif pattern["type"] == "pointer":
                    if "target" in pattern:
                        return recurse(pattern["target"], schema["target"])
                    return True

                else:
                    assert pattern["type"] in ("primitive", "list", "union", "record", "tuple", "pointer")

        return recurse(cls.pattern, schema.tojson(explicit=True))
