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

import math
import numbers
import sys

import numpy

import oamap.schema
import oamap.generator
import oamap.proxy

if sys.version_info[0] < 3:
    range = xrange

################################################################ utilities

def newvar(avoid, trial=None):
    while trial is None or trial in avoid:
        trial = "v" + str(len(avoid))
    avoid.add(trial)
    return trial

def trycompile(numba):
    if numba is None or numba is False:
        return lambda fcn: fcn
    else:
        try:
            import numba as nb
        except ImportError:
            return lambda fcn: fcn
        else:
            if numba is True:
                numbaopts = {}
            else:
                numbaopts = numba
            return lambda fcn: fcn if isinstance(fcn, nb.dispatcher.Dispatcher) else nb.jit(**numbaopts)(fcn)

class DualSource(object):
    def __init__(self, old, oldns):
        self.old = old
        self.new = {}

        i = 0
        self.namespace = None
        while self.namespace is None or self.namespace in oldns:
            self.namespace = "namespace-" + str(i)
            i += 1

        self._arraynum = 0

    def arrayname(self):
        trial = None
        while trial is None or trial in self.new:
            trial = "array-" + str(self._arraynum)
            self._arraynum += 1
        return trial

    def getall(self, roles):
        out = {}

        if hasattr(self.old, "getall"):
            out.update(self.old.getall([x for x in roles if x.namespace != self.namespace]))
        else:
            for x in roles:
                if x.namespace != self.namespace:
                    out[x] = self.old[str(x)]

        if hasattr(self.new, "getall"):
            out.update(self.new.getall([x for x in roles if x.namespace == self.namespace]))
        else:
            for x in roles:
                if x.namespace == self.namespace:
                    out[x] = self.new[str(x)]

        return out

    def put(self, schemanode, *arrays):
        if isinstance(schemanode, oamap.schema.Primitive):
            datarole = oamap.generator.DataRole(self.arrayname(), self.namespace)
            roles2arrays = {datarole: arrays[0]}
            schemanode.data = str(datarole)

        elif isinstance(schemanode, oamap.schema.List):
            startsrole = oamap.generator.StartsRole(self.arrayname(), self.namespace, None)
            stopsrole = oamap.generator.StopsRole(self.arrayname(), self.namespace, None)
            startsrole.stops = stopsrole
            stopsrole.starts = startsrole
            roles2arrays = {startsrole: arrays[0], stopsrole: arrays[1]}
            schemanode.starts = str(startsrole)
            schemanode.stops = str(stopsrole)

        elif isinstance(schemanode, oamap.schema.Union):
            tagsrole = oamap.generator.TagsRole(self.arrayname(), self.namespace, None)
            offsetsrole = oamap.generator.OffsetsRole(self.arrayname(), self.namespace, None)
            tagsrole.offsets = offsetsrole
            offsetsrole.tags = tagsrole
            roles2arrays = {tagsrole: arrays[0], offsetsrole: arrays[1]}
            schemanode.tags = str(tagsrole)
            schemanode.offsets = str(offsetsrole)

        elif isinstance(schemanode, oamap.schema.Record):
            pass

        elif isinstance(schemanode, oamap.schema.Tuple):
            pass

        elif isinstance(schemanode, oamap.schema.Pointer):
            positionsrole = oamap.generator.PositionsRole(self.arrayname(), self.namespace)
            roles2arrays = {positionsrole: arrays[0]}
            schemanode.positions = str(positionsrole)

        else:
            raise AssertionError(schemanode)

        if schemanode.nullable:
            maskrole = oamap.generator.MaskRole(self.arrayname(), self.namespace, roles2arrays)
            roles2arrays = dict(list(roles2arrays.items()) + [(maskrole, arrays[-1])])
            schemanode.mask = str(maskrole)

        schemanode.namespace = self.namespace
        self.putall(roles2arrays)

    def putall(self, roles2arrays):
        if hasattr(self.new, "putall"):
            self.new.putall(roles2arrays)
        else:
            for n, x in roles2arrays.items():
                self.new[str(n)] = x

    def close(self):
        if hasattr(self.old, "close"):
            self.old.close()
        if hasattr(self.new, "close"):
            self.new.close()

################################################################ project/keep/drop

def project(data, path):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema().project(path)
        if schema is None:
            raise TypeError("projection resulted in no schema")
        out = schema(data._arrays)
        if isinstance(data, oamap.proxy.ListProxy):
            out._whence, out._stride, out._length = data._whence, data._stride, data._length
        elif isinstance(data, oamap.proxy.RecordProxy):
            out._index = data._index
        elif isinstance(data, oamap.proxy.TupleProxy):
            out._index = data._index
        else:
            raise AssertionError(type(data))
        return out

    else:
        raise TypeError("project can only be applied to an OAMap proxy (List, Record, Tuple)")

def keep(data, *paths):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema().keep(*paths)
        if schema is None:
            raise TypeError("keep operation resulted in no schema")
        out = schema(data._arrays)
        if isinstance(data, oamap.proxy.ListProxy):
            out._whence, out._stride, out._length = data._whence, data._stride, data._length
        elif isinstance(data, oamap.proxy.RecordProxy):
            out._index = data._index
        elif isinstance(data, oamap.proxy.TupleProxy):
            out._index = data._index
        else:
            raise AssertionError(type(data))
        return out

    else:
        raise TypeError("keep can only be applied to an OAMap proxy (List, Record, Tuple)")

def drop(data, *paths):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema().drop(*paths)
        if schema is None:
            raise TypeError("drop operation resulted in no schema")
        out = schema(data._arrays)
        if isinstance(data, oamap.proxy.ListProxy):
            out._whence, out._stride, out._length = data._whence, data._stride, data._length
        elif isinstance(data, oamap.proxy.RecordProxy):
            out._index = data._index
        elif isinstance(data, oamap.proxy.TupleProxy):
            out._index = data._index
        else:
            raise AssertionError(type(data))
        return out

    else:
        raise TypeError("drop can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ mask

def mask(data, path, low, high=None):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema()
        nodes = schema.path(path, parents=True)
        while isinstance(nodes[0], oamap.schema.List):
            nodes = (nodes[0].content,) + nodes
        node = nodes[0]

        arrays = DualSource(data._arrays, data._generator.namespaces())

        if isinstance(node, oamap.schema.Primitive):
            generator = data._generator.findbynames("Primitive", data=node.data, mask=node.mask)

            primitive = generator._getdata(data._arrays, data._cache).copy()
            if node.nullable:
                mask = generator._getmask(data._arrays, data._cache).copy()
            else:
                node.nullable = True
                mask = numpy.arange(len(primitive), dtype=oamap.generator.Masked.maskdtype)

            if high is None:
                if math.isnan(low):
                    selection = numpy.isnan(primitive)
                else:
                    selection = (primitive == low)
            else:
                if math.isnan(low) or math.isnan(high):
                    raise ValueError("if a range is specified, neither of the endpoints can be NaN")
                selection = (primitive >= low)
                numpy.bitwise_and(selection, (primitive <= high), selection)

            mask[selection] = oamap.generator.Masked.maskedvalue

            arrays.put(node, primitive, mask)

        else:
            raise NotImplementedError("mask operation only defined on primitive fields; {0} matches:\n\n    {1}".format(repr(path), node.__repr__(indent="    ")))

        out = schema(arrays)
        if isinstance(data, oamap.proxy.ListProxy):
            out._whence, out._stride, out._length = data._whence, data._stride, data._length
        elif isinstance(data, oamap.proxy.RecordProxy):
            out._index = data._index
        elif isinstance(data, oamap.proxy.TupleProxy):
            out._index = data._index
        else:
            raise AssertionError(type(data))
        return out

    else:
        raise TypeError("mask can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ flatten

def flatten(data):
    if isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1 and isinstance(data._generator.content, oamap.generator.ListGenerator):
        if isinstance(data._generator, oamap.generator.Masked) or isinstance(data._generator.content, oamap.generator.Masked):
            raise NotImplementedError("nullable; need to merge masks")

        schema = data._generator.namedschema()
        schema.content = schema.content.content

        starts, stops = data._generator.content._getstartsstops(data._arrays, data._cache)

        arrays = DualSource(data._arrays, data._generator.namespaces())

        if numpy.array_equal(starts[1:], stops[:-1]):
            # important special case: contiguous
            arrays.put(schema, starts[:1], stops[-1:])
            return schema(arrays)
        else:
            raise NotImplementedError("non-contiguous arrays: have to do some sort of concatenation")

    else:
        raise TypeError("flatten can only be applied to List(List(...))")

################################################################ filter

def filter(data, fcn, args=(), fieldname=None, numba=True):
    if not isinstance(args, tuple):
        args = tuple(args)

    if fieldname is None and isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1:
        if isinstance(data._generator, oamap.generator.Masked):
            raise NotImplementedError("nullable; need to merge masks")            

        schema = oamap.schema.List(oamap.schema.Pointer(data._generator.namedschema().content))

        params = fcn.__code__.co_varnames[:fcn.__code__.co_argcount]
        avoid = set(params)
        fcnname = newvar(avoid, "fcn")
        fillname = newvar(avoid, "fill")

        fcn = trycompile(numba)(fcn)
        env = {fcnname: fcn}
        exec("""
def {fill}({data}, {pointers}{params}):
    {i} = 0
    {numitems} = 0
    for {datum} in {data}:
        if {fcn}({datum}{params}):
            {pointers}[{numitems}] = {i}
            {numitems} += 1
        {i} += 1
    return {numitems}
""".format(fill=fillname,
           data=newvar(avoid, "data"),
           pointers=newvar(avoid, "pointers"),
           params="".join("," + x for x in params[1:]),
           i=newvar(avoid, "i"),
           numitems=newvar(avoid, "numitems"),
           datum=newvar(avoid, "datum"),
           fcn=fcnname), env)
        fill = trycompile(numba)(env[fillname])

        pointers = numpy.empty(data._length, dtype=oamap.generator.PointerGenerator.posdtype)
        numitems = fill(*((data, pointers) + args))
        offsets = numpy.array([0, numitems], dtype=data._generator.posdtype)
        pointers = pointers[:numitems]

        if isinstance(data._generator.content, oamap.generator.PointerGenerator):
            if isinstance(data._generator.content, oamap.generator.Masked):
                raise NotImplementedError("nullable; need to merge masks")
            innerpointers = data._generator.content._getpositions(data._arrays, data._cache)
            pointers = innerpointers[pointers]
            schema.content.target = schema.content.target.target

        arrays = DualSource(data._arrays, data._generator.namespaces())
        arrays.put(schema, offsets[:-1], offsets[1:])
        arrays.put(schema.content, pointers)
        return schema(arrays)

    elif fieldname is not None and isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1 and isinstance(data._generator.content, oamap.generator.RecordGenerator) and fieldname in data._generator.content.fields and isinstance(data._generator.content.fields[fieldname], oamap.generator.ListGenerator):
        if isinstance(data._generator, oamap.generator.Masked) or isinstance(data._generator.content, oamap.generator.Masked) or isinstance(data._generator.content.fields[fieldname], oamap.generator.Masked):
            raise NotImplementedError("nullable; need to merge masks")            

        schema = data._generator.namedschema()
        schema.content[fieldname] = oamap.schema.List(oamap.schema.Pointer(schema.content[fieldname].content))

        params = fcn.__code__.co_varnames[:fcn.__code__.co_argcount]
        avoid = set(params)
        fcnname = newvar(avoid, "fcn")
        fillname = newvar(avoid, "fill")

        fcn = trycompile(numba)(fcn)
        env = {fcnname: fcn}
        exec("""
def {fill}({data}, {innerstarts}, {stops}, {pointers}{params}):
    {i} = 0
    {numitems} = 0
    for {outer} in {data}:
        {index} = {innerstarts}[{i}]
        for {inner} in {outer}.{fieldname}:
            if {fcn}({inner}{params}):
                {pointers}[{numitems}] = {index}
                {numitems} += 1
            {index} += 1
        {stops}[{i}] = {numitems}
        {i} += 1
    return {numitems}
""".format(fill=fillname,
           data=newvar(avoid, "data"),
           innerstarts=newvar(avoid, "innerstarts"),
           stops=newvar(avoid, "stops"),
           pointers=newvar(avoid, "pointers"),
           params="".join("," + x for x in params[1:]),
           i=newvar(avoid, "i"),
           numitems=newvar(avoid, "numitems"),
           outer=newvar(avoid, "outer"),
           index=newvar(avoid, "index"),
           inner=newvar(avoid, "inner"),
           fieldname=fieldname,
           fcn=fcnname), env)
        fill = trycompile(numba)(env[fillname])

        innerstarts, innerstops = data._generator.content.fields[fieldname]._getstartsstops(data._arrays, data._cache)
        offsets = numpy.empty(data._length + 1, dtype=data._generator.content.fields[fieldname].posdtype)
        offsets[0] = 0
        pointers = numpy.empty(innerstops.max(), dtype=oamap.generator.PointerGenerator.posdtype)
        numitems = fill(*((data, innerstarts, offsets[1:], pointers) + args))
        pointers = pointers[:numitems]

        if isinstance(data._generator.content.fields[fieldname].content, oamap.generator.PointerGenerator):
            if isinstance(data._generator.content.fields[fieldname].content, oamap.generator.Masked):
                raise NotImplementedError("nullable; need to merge masks")
            innerpointers = data._generator.content.fields[fieldname].content._getpositions(data._arrays, data._cache)
            pointers = innerpointers[pointers]
            schema.content[fieldname].content.target = schema.content[fieldname].content.target.target

        arrays = DualSource(data._arrays, data._generator.namespaces())
        arrays.put(schema.content[fieldname], offsets[:-1], offsets[1:])
        arrays.put(schema.content[fieldname].content, pointers)
        return schema(arrays)
        
    elif fieldname is None:
        raise TypeError("filter without fieldname can only be applied to a List(...)")

    else:
        raise TypeError("filter with fieldname can only be applied to a top-level List(Record({{{0}: List(...)}}))".format(repr(fieldname)))

################################################################ define

def define(data, fieldname, fcn, args=(), at="", fieldtype=oamap.schema.Primitive(numpy.float64), numba=True):
    if not isinstance(args, tuple):
        args = tuple(args)

    if isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1:
        schema = data._generator.namedschema()
        nodes = schema.path(at, parents=True)
        while isinstance(nodes[0], oamap.schema.List):
            nodes = (nodes[0].content,) + nodes
        if not isinstance(nodes[0], oamap.schema.Record):
            raise TypeError("path {0} does not refer to a record:\n\n    {1}".format(repr(at), nodes[0].__repr__(indent="    ")))
        if len(nodes) < 2 or not isinstance(nodes[1], oamap.schema.List):
            raise TypeError("path {0} does not refer to a record in a list:\n\n    {1}".format(repr(at), nodes[1].__repr__(indent="    ")))
        recordnode = nodes[0]
        listnode = nodes[1]
        if recordnode.nullable or listnode.nullable:
            raise NotImplementedError("nullable; need to merge masks")

        recordnode[fieldname] = fieldtype.deepcopy()

        listgenerator = data._generator.findbynames("List", starts=listnode.starts, stops=listnode.stops)
        viewstarts, viewstops = listgenerator._getstartsstops(data._arrays, data._cache)
        viewschema = listgenerator.namedschema()
        viewarrays = DualSource(data._arrays, data._generator.namespaces())
        if numpy.array_equal(viewstarts[1:], viewstops[:-1]):
            viewarrays.put(viewschema, viewstarts[:1], viewstops[-1:])
        else:
            raise NotImplementedError("non-contiguous arrays: have to do some sort of concatenation")
        view = viewschema(viewarrays)

        params = fcn.__code__.co_varnames[:fcn.__code__.co_argcount]
        avoid = set(params)
        fcnname = newvar(avoid, "fcn")
        fillname = newvar(avoid, "fill")

        if isinstance(fieldtype, oamap.schema.Primitive) and not fieldtype.nullable:
            fcn = trycompile(numba)(fcn)
            env = {fcnname: fcn}
            exec("""
def {fill}({view}, {primitive}{params}):
    {i} = 0
    for {datum} in {view}:
        {primitive}[{i}] = {fcn}({datum}{params})
        {i} += 1
""".format(fill=fillname,
           view=newvar(avoid, "view"),
           primitive=newvar(avoid, "primitive"),
           params="".join("," + x for x in params[1:]),
           i=newvar(avoid, "i"),
           datum=newvar(avoid, "datum"),
           fcn=fcnname), env)
            fill = trycompile(numba)(env[fillname])

            primitive = numpy.empty(len(view), dtype=fieldtype.dtype)
            fill(*((view, primitive) + args))

            arrays = DualSource(data._arrays, data._generator.namespaces())
            arrays.put(recordnode[fieldname], primitive)
            return schema(arrays)

        elif isinstance(fieldtype, oamap.schema.Primitive):
            fcn = trycompile(numba)(fcn)
            env = {fcnname: fcn}
            exec("""
def {fill}({view}, {primitive}, {mask}{params}):
    {i} = 0
    {numitems} = 0
    for {datum} in {view}:
        {tmp} = {fcn}({datum}{params})
        if {tmp} is None:
            {mask}[{i}] = {maskedvalue}
        else:
            {mask}[{i}] = {numitems}
            {primitive}[{numitems}] = {tmp}
            {numitems} += 1
        {i} += 1
    return {numitems}
""".format(fill=fillname,
           view=newvar(avoid, "view"),
           primitive=newvar(avoid, "primitive"),
           mask=newvar(avoid, "mask"),
           params="".join("," + x for x in params[1:]),
           i=newvar(avoid, "i"),
           numitems=newvar(avoid, "numitems"),
           datum=newvar(avoid, "datum"),
           tmp=newvar(avoid, "tmp"),
           fcn=fcnname,
           maskedvalue=oamap.generator.Masked.maskedvalue), env)
            fill = trycompile(numba)(env[fillname])
            
            primitive = numpy.empty(len(view), dtype=fieldtype.dtype)
            mask = numpy.empty(len(view), dtype=oamap.generator.Masked.maskdtype)
            fill(*((view, primitive, mask) + args))

            arrays = DualSource(data._arrays, data._generator.namespaces())
            arrays.put(recordnode[fieldname], primitive, mask)
            return schema(arrays)

        else:
            raise NotImplementedError("define not implemented for fieldtype:\n\n    {0}".format(fieldtype.__repr__(indent="    ")))

    else:
        raise TypeError("define can only be applied to a top-level List(...)")

################################################################ quick test

# from oamap.schema import *

# dataset = List(Record(dict(x=List("int"), y=List("double")))).fromdata([{"x": [1, 2, 3], "y": [1.1, numpy.nan]}, {"x": [], "y": []}, {"x": [4, 5], "y": [3.3]}])

# dataset = List(Record(dict(x="int", y="double"))).fromdata([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])

# dataset = List(Record(dict(x=List(Record({"xx": "int"})), y="double"))).fromdata([{"x": [{"xx": 1}, {"xx": 2}], "y": 1.1}, {"x": [], "y": 2.2}, {"x": [{"xx": 3}], "y": 3.3}])

# dataset = List(List("int")).fromdata([[1, 2, 3], [], [4, 5]])

# dataset = List(List(List("int"))).fromdata([[[1, 2, 3], [4, 5], []], [], [[6], [7, 8]]])

# def f(x, y):
#   return len(x) == y

# filter(dataset, f, (0,), numba=False)
