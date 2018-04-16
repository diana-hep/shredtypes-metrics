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

################################################################ utilities

def _newvar(avoid, trial=None):
    while trial is None or trial in avoid:
        trial = "v" + str(len(avoid))
    avoid.add(trial)
    return trial

def paramtypes(args):
    try:
        import numba as nb
    except ImportError:
        return None
    else:
        return tuple(nb.typeof(x) for x in args)

def trycompile(fcn, paramtypes=None, numba=True):
    if numba is None or numba is False:
        return fcn

    try:
        import numba as nb
    except ImportError:
        return fcn

    if numba is True:
        numbaopts = {}
    else:
        numbaopts = numba

    if isinstance(fcn, nb.dispatcher.Dispatcher):
        fcn = fcn.py_fcn

    if paramtypes is None:
        return nb.jit(**numbaopts)(fcn)
    else:
        return nb.jit(paramtypes, **numbaopts)(fcn)

def returntype(fcn, paramtypes):
    try:
        import numba as nb
    except ImportError:
        return None

    if isinstance(fcn, nb.dispatcher.Dispatcher):
        overload = fcn.overloads.get(paramtypes, None)
        if overload is None:
            return None
        else:
            return overload.signature.return_type

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

def _setindexes(input, output):
    if isinstance(input, oamap.proxy.ListProxy):
        output._whence, output._stride, output._length = input._whence, input._stride, input._length
    elif isinstance(input, oamap.proxy.RecordProxy):
        output._index = input._index
    elif isinstance(input, oamap.proxy.TupleProxy):
        output._index = input._index
    else:
        raise AssertionError(type(input))
    return output
    
################################################################ fieldname/recordname

def fieldname(data, path, newname):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema()
        nodes = schema.path(path, parents=True)
        if len(nodes) < 2:
            raise TypeError("path {0} did not match a field in a record".format(repr(path)))

        for n, x in nodes[1].fields.items():
            if x is nodes[0]:
                oldname = n
                break

        del nodes[1][oldname]
        nodes[1][newname] = nodes[0]
        return _setindexes(data, schema(data._arrays))
        
    else:
        raise TypeError("fieldname can only be applied to an OAMap proxy (List, Record, Tuple)")

def recordname(data, path, newname):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema()
        nodes = schema.path(path, parents=True)
        while isinstance(nodes[0], oamap.schema.List):
            nodes = (nodes[0].content,) + nodes
        if not isinstance(nodes[0], oamap.schema.Record):
            raise TypeError("path {0} did not match a record".format(repr(path)))

        nodes[0].name = newname
        return _setindexes(data, schema(data._arrays))
        
    else:
        raise TypeError("fieldname can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ project/keep/drop

def project(data, path):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema().project(path)
        if schema is None:
            raise TypeError("projection resulted in no schema")
        return _setindexes(data, schema(data._arrays))
    else:
        raise TypeError("project can only be applied to an OAMap proxy (List, Record, Tuple)")

def keep(data, *paths):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema().keep(*paths)
        if schema is None:
            raise TypeError("keep operation resulted in no schema")
        return _setindexes(data, schema(data._arrays))
    else:
        raise TypeError("keep can only be applied to an OAMap proxy (List, Record, Tuple)")

def drop(data, *paths):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema().drop(*paths)
        if schema is None:
            raise TypeError("drop operation resulted in no schema")
        return _setindexes(data, schema(data._arrays))
    else:
        raise TypeError("drop can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ split

def split(data, *paths):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema()

        for path in paths:
            for nodes in schema.paths(path, parents=True):
                if len(nodes) < 4 or not isinstance(nodes[1], oamap.schema.Record) or not isinstance(nodes[2], oamap.schema.List) or not isinstance(nodes[3], oamap.schema.Record):
                    raise TypeError("path {0} matches a field that is not in a Record(List(Record({{field: ...}})))".format(repr(path)))

                datanode, innernode, listnode, outernode = nodes[0], nodes[1], nodes[2], nodes[3]
                for n, x in innernode.fields.items():
                    if x is datanode:
                        innername = n
                        break
                for n, x in outernode.fields.items():
                    if x is listnode:
                        outername = n
                        break

                del innernode[innername]
                if len(innernode.fields) == 0:
                    del outernode[outername]

                outernode[innername] = listnode.copy(content=datanode)

        return schema(data._arrays)

    else:
        raise TypeError("split can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ merge

def merge(data, container, *paths):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema()

        constructed = False
        try:
            nodes = schema.path(container, parents=True)

        except ValueError:
            try:
                slash = container.rindex("/")
            except ValueError:
                nodes = (schema,)
                tomake = container
            else:
                tofind, tomake = container[:slash], container[slash + 1:]
                nodes = schema.path(tofind, parents=True)
                container = tofind

            while isinstance(nodes[0], oamap.schema.List):
                nodes = (nodes[0].content,) + nodes
            if not isinstance(nodes[0], oamap.schema.Record):
                raise TypeError("container parent {0} is not a record".format(repr(container)))
            nodes[0][tomake] = oamap.schema.List(oamap.schema.Record({}))
            nodes = (nodes[0][tomake].content, nodes[0][tomake]) + nodes
            constructed = True

        else:
            while isinstance(nodes[0], oamap.schema.List):
                nodes = (nodes[0].content,) + nodes
            
        if len(nodes) < 2 or not isinstance(nodes[0], oamap.schema.Record) or not isinstance(nodes[1], oamap.schema.List):
            raise TypeError("container must be a List(Record(...))")
        
        containerrecord, containerlist = nodes[0], nodes[1]
        parents = nodes[2:]
        listnodes = []
        if not constructed:
            listnodes.append(containerlist)

        for path in paths:
            for nodes in schema.paths(path, parents=True):
                if len(nodes) < 2 or not isinstance(nodes[0], oamap.schema.List) or nodes[1:] != parents:
                    raise TypeError("".format(repr(path)))

                listnode, outernode = nodes[0], nodes[1]
                listnodes.append(listnode)
                
                for n, x in outernode.fields.items():
                    if x is listnode:
                        outername = n
                        break

                del outernode[outername]
                containerrecord[outername] = listnode.content

        if len(listnodes) == 0:
            raise TypeError("at least one path must match schema elements")

        if not all(x.namespace == listnodes[0].namespace and x.starts == listnodes[0].starts and x.stops == listnodes[0].stops for x in listnodes[1:]):
            starts1, stops1 = data._generator.findbynames("List", listnodes[0].namespace, starts=listnodes[0].starts, stops=listnodes[0].stops)._getstartsstops(data._arrays, data._cache)
            for x in listnodes[1:]:
                starts2, stops2 = data._generator.findbynames("List", x.namespace, starts=x.starts, stops=x.stops)._getstartsstops(data._arrays, data._cache)
                if not (starts1 is starts2 or numpy.array_equal(starts1, starts2)) and not (stops1 is stops2 or numpy.array_equal(stops1, stops2)):
                    raise ValueError("some of the paths refer to lists of different lengths")

        if constructed:
            containerlist.namespace = listnodes[0].namespace
            containerlist.starts = listnodes[0].starts
            containerlist.stops = listnodes[0].stops

        return schema(data._arrays)

    else:
        raise TypeError("merge can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ mask

def mask(data, at, low, high=None):
    if isinstance(data, oamap.proxy.Proxy):
        schema = data._generator.namedschema()
        nodes = schema.path(at, parents=True)
        while isinstance(nodes[0], oamap.schema.List):
            nodes = (nodes[0].content,) + nodes
        node = nodes[0]

        arrays = DualSource(data._arrays, data._generator.namespaces())

        if isinstance(node, oamap.schema.Primitive):
            generator = data._generator.findbynames("Primitive", node.namespace, data=node.data, mask=node.mask)

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
            raise NotImplementedError("mask operation only defined on primitive fields; {0} matches:\n\n    {1}".format(repr(at), node.__repr__(indent="    ")))

        return _setindexes(data, schema(arrays))

    else:
        raise TypeError("mask can only be applied to an OAMap proxy (List, Record, Tuple)")

################################################################ flatten

def flatten(data, at=""):
    if (isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1) or (isinstance(data, oamap.proxy.Proxy) and data._index == 0):
        schema = data._generator.namedschema()
        outernode = schema.path(at)
        if not isinstance(outernode, oamap.schema.List) or not isinstance(outernode.content, oamap.schema.List):
            raise TypeError("path {0} does not refer to a list within a list:\n\n    {1}".format(repr(at), outernode.__repr__(indent="    ")))
        innernode = outernode.content
        if outernode.nullable or innernode.nullable:
            raise NotImplementedError("nullable; need to merge masks")

        outergenerator = data._generator.findbynames("List", outernode.namespace, starts=outernode.starts, stops=outernode.stops)
        outerstarts, outerstops = outergenerator._getstartsstops(data._arrays, data._cache)
        innergenerator = data._generator.findbynames("List", innernode.namespace, starts=innernode.starts, stops=innernode.stops)
        innerstarts, innerstops = innergenerator._getstartsstops(data._arrays, data._cache)

        if not numpy.array_equal(innerstarts[1:], innerstops[:-1]):
            raise NotImplementedError("inner arrays are not contiguous: flatten would require the creation of pointers")

        starts = innerstarts[outerstarts]
        stops  = innerstops[outerstops - 1]

        outernode.content = innernode.content

        arrays = DualSource(data._arrays, data._generator.namespaces())
        arrays.put(outernode, starts, stops)
        return schema(arrays)

    else:
        raise TypeError("flatten can only be applied to a top-level OAMap proxy (List, Record, Tuple)")


def pairs(data, fieldname=None):
    '''
    data = 1D numpy array, list , tuple, or a List() or Tuple(). 
    fieldname = output numpy array. Will be returned from the function.
    Usage: out_array = pairs(data, fieldname)
    Works with Lists or tuples as data input only.
    Fieldname not included yet. 
    '''
    from oamap.schema import *
    if isinstance(data, oamap.proxy.ListProxy) or isinstance(data, oamap.proxy.TupleProxy) or isinstance(data, list) or isinstance(data, tuple) or isinstance(data, numpy.ndarray):
        if fieldname is None:
            # No fieldname given, We are free to choose out own. Let's return a nested List
            arr = numpy.array(data)
            arr1 = numpy.ndarray.tolist(arr[numpy.transpose(numpy.triu_indices(len(arr), 1))])
            schema = List(List('float'))
            # For returning tuples, uncomment. List comprehension is unfortunately required, maybe I am missing soemthing?
            #schema = List(Tuple(['float','float']))
            #arr2 = [tuple(l) for l in arr1]
            #obj = schema.fromdata(arr2)
            obj = schema.fromdata(arr1)
            return obj
        else:
            raise NotImplementedError("Will be available when records are available")
    else:
        raise NotImplementedError("Works with Lists or Tuple data only")
################################################################ filter

def filter(data, fcn, args=(), at="", numba=True):
    if not isinstance(args, tuple):
        try:
            args = tuple(args)
        except TypeError:
            args = (args,)

    if (isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1) or (isinstance(data, oamap.proxy.Proxy) and data._index == 0):
        schema = data._generator.namedschema()
        listnode = schema.path(at)
        if not isinstance(listnode, oamap.schema.List):
            raise TypeError("path {0} does not refer to a list:\n\n    {1}".format(repr(at), listnode.__repr__(indent="    ")))
        if listnode.nullable:
            raise NotImplementedError("nullable; need to merge masks")

        listgenerator = data._generator.findbynames("List", listnode.namespace, starts=listnode.starts, stops=listnode.stops)
        viewstarts, viewstops = listgenerator._getstartsstops(data._arrays, data._cache)
        viewschema = listgenerator.namedschema()
        viewarrays = DualSource(data._arrays, data._generator.namespaces())
        viewoffsets = numpy.array([viewstarts.min(), viewstops.max()], dtype=oamap.generator.ListGenerator.posdtype)
        viewarrays.put(viewschema, viewoffsets[:1], viewoffsets[-1:])
        view = viewschema(viewarrays)

        params = fcn.__code__.co_varnames[:fcn.__code__.co_argcount]
        avoid = set(params)
        fcnname = _newvar(avoid, "fcn")
        fillname = _newvar(avoid, "fill")
        lenname = _newvar(avoid, "len")
        rangename = _newvar(avoid, "range")

        ptypes = paramtypes(args)
        if ptypes is not None:
            import numba as nb
            from oamap.compiler import typeof_generator
            ptypes = (typeof_generator(view._generator.content),) + ptypes
        fcn = trycompile(fcn, paramtypes=ptypes, numba=numba)
        rtype = returntype(fcn, ptypes)
        if rtype is not None:
            if rtype != nb.types.boolean:
                raise TypeError("filter function must return boolean, not {0}".format(rtype))

        env = {fcnname: fcn, lenname: len, rangename: range if sys.version_info[0] > 2 else xrange}
        exec("""
def {fill}({view}, {viewstarts}, {viewstops}, {stops}, {pointers}{params}):
    {numitems} = 0
    for {i} in {range}({len}({viewstarts})):
        for {j} in {range}({viewstarts}[{i}], {viewstops}[{i}]):
            {datum} = {view}[{j}]
            if {fcn}({datum}{params}):
                {pointers}[{numitems}] = {j}
                {numitems} += 1
        {stops}[{i}] = {numitems}
    return {numitems}
""".format(fill=fillname,
           view=_newvar(avoid, "view"),
           viewstarts=_newvar(avoid, "viewstarts"),
           viewstops=_newvar(avoid, "viewstops"),
           stops=_newvar(avoid, "stops"),
           pointers=_newvar(avoid, "pointers"),
           params="".join("," + x for x in params[1:]),
           numitems=_newvar(avoid, "numitems"),
           i=_newvar(avoid, "i"),
           range=rangename,
           len=lenname,
           j=_newvar(avoid, "j"),
           datum=_newvar(avoid, "datum"),
           fcn=fcnname), env)
        fill = trycompile(env[fillname], numba=numba)

        offsets = numpy.empty(len(viewstarts) + 1, dtype=oamap.generator.ListGenerator.posdtype)
        offsets[0] = 0
        pointers = numpy.empty(len(view), dtype=oamap.generator.PointerGenerator.posdtype)
        numitems = fill(*((view, viewstarts, viewstops, offsets[1:], pointers) + args))
        pointers = pointers[:numitems]

        listnode.content = oamap.schema.Pointer(listnode.content)

        if isinstance(listgenerator.content, oamap.generator.PointerGenerator):
            if isinstance(listgenerator.content, oamap.generator.Masked):
                raise NotImplementedError("nullable; need to merge masks")
            innerpointers = listgenerator.content._getpositions(data._arrays, data._cache)
            pointers = innerpointers[pointers]
            listnode.content.target = listnode.content.target.target

        arrays = DualSource(data._arrays, data._generator.namespaces())
        arrays.put(listnode, offsets[:-1], offsets[1:])
        arrays.put(listnode.content, pointers)
        return schema(arrays)

    else:
        raise TypeError("filter can only be applied to a top-level OAMap proxy (List, Record, Tuple)")

################################################################ define

def define(data, fieldname, fcn, args=(), at="", fieldtype=oamap.schema.Primitive(numpy.float64), numba=True):
    if not isinstance(args, tuple):
        try:
            args = tuple(args)
        except TypeError:
            args = (args,)

    if (isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1) or (isinstance(data, oamap.proxy.Proxy) and data._index == 0):
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

        listgenerator = data._generator.findbynames("List", listnode.namespace, starts=listnode.starts, stops=listnode.stops)
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
        fcnname = _newvar(avoid, "fcn")
        fillname = _newvar(avoid, "fill")

        ptypes = paramtypes(args)
        if ptypes is not None:
            import numba as nb
            from oamap.compiler import typeof_generator
            ptypes = (typeof_generator(view._generator.content),) + ptypes
        fcn = trycompile(fcn, paramtypes=ptypes, numba=numba)
        rtype = returntype(fcn, ptypes)

        if isinstance(fieldtype, oamap.schema.Primitive) and not fieldtype.nullable:
            if rtype is not None:
                if rtype == nb.types.pyobject:
                    raise TypeError("numba could not prove that the function's output type is:\n\n    {0}".format(fieldtype.__repr__(indent="    ")))
                elif rtype != nb.from_dtype(fieldtype.dtype):
                    raise TypeError("function returns {0} but fieldtype is set to:\n\n    {1}".format(rtype, fieldtype.__repr__(indent="    ")))

            env = {fcnname: fcn}
            exec("""
def {fill}({view}, {primitive}{params}):
    {i} = 0
    for {datum} in {view}:
        {primitive}[{i}] = {fcn}({datum}{params})
        {i} += 1
""".format(fill=fillname,
           view=_newvar(avoid, "view"),
           primitive=_newvar(avoid, "primitive"),
           params="".join("," + x for x in params[1:]),
           i=_newvar(avoid, "i"),
           datum=_newvar(avoid, "datum"),
           fcn=fcnname), env)
            fill = trycompile(env[fillname], numba=numba)

            primitive = numpy.empty(len(view), dtype=fieldtype.dtype)
            fill(*((view, primitive) + args))

            arrays = DualSource(data._arrays, data._generator.namespaces())
            arrays.put(recordnode[fieldname], primitive)
            return schema(arrays)

        elif isinstance(fieldtype, oamap.schema.Primitive):
            if rtype is not None:
                if rtype != nb.types.optional(nb.from_dtype(fieldtype.dtype)):
                    raise TypeError("function returns {0} but fieldtype is set to:\n\n    {1}".format(rtype, fieldtype.__repr__(indent="    ")))

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
           view=_newvar(avoid, "view"),
           primitive=_newvar(avoid, "primitive"),
           mask=_newvar(avoid, "mask"),
           params="".join("," + x for x in params[1:]),
           i=_newvar(avoid, "i"),
           numitems=_newvar(avoid, "numitems"),
           datum=_newvar(avoid, "datum"),
           tmp=_newvar(avoid, "tmp"),
           fcn=fcnname,
           maskedvalue=oamap.generator.Masked.maskedvalue), env)
            fill = trycompile(env[fillname], numba=numba)
            
            primitive = numpy.empty(len(view), dtype=fieldtype.dtype)
            mask = numpy.empty(len(view), dtype=oamap.generator.Masked.maskdtype)
            fill(*((view, primitive, mask) + args))

            arrays = DualSource(data._arrays, data._generator.namespaces())
            arrays.put(recordnode[fieldname], primitive, mask)
            return schema(arrays)

        else:
            raise NotImplementedError("define not implemented for fieldtype:\n\n    {0}".format(fieldtype.__repr__(indent="    ")))

    else:
        raise TypeError("define can only be applied to a top-level OAMap proxy (List, Record, Tuple)")

################################################################ map

def map(data, fcn, args=(), at="", names=None, numba=True):
    if not isinstance(args, tuple):
        try:
            args = tuple(args)
        except TypeError:
            args = (args,)

    if (isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1) or (isinstance(data, oamap.proxy.Proxy) and data._index == 0):
        listnode = data._generator.namedschema().path(at)
        if not isinstance(listnode, oamap.schema.List):
            raise TypeError("path {0} does not refer to a list:\n\n    {1}".format(repr(at), listnode.__repr__(indent="    ")))
        if listnode.nullable:
            raise NotImplementedError("nullable; need to merge masks")

        listgenerator = data._generator.findbynames("List", listnode.namespace, starts=listnode.starts, stops=listnode.stops)

        viewstarts, viewstops = listgenerator._getstartsstops(data._arrays, data._cache)
        viewschema = listgenerator.namedschema()
        viewarrays = DualSource(data._arrays, data._generator.namespaces())
        viewoffsets = numpy.array([viewstarts.min(), viewstops.max()], dtype=oamap.generator.ListGenerator.posdtype)
        viewarrays.put(viewschema, viewoffsets[:1], viewoffsets[-1:])
        view = viewschema(viewarrays)

        params = fcn.__code__.co_varnames[:fcn.__code__.co_argcount]
        avoid = set(params)
        fcnname = _newvar(avoid, "fcn")
        fillname = _newvar(avoid, "fill")

        ptypes = paramtypes(args)
        if ptypes is not None:
            import numba as nb
            from oamap.compiler import typeof_generator
            ptypes = (typeof_generator(view._generator.content),) + ptypes
        fcn = trycompile(fcn, paramtypes=ptypes, numba=numba)
        rtype = returntype(fcn, ptypes)

        print "rtype", rtype, type(rtype)

        if rtype is None:
            first = fcn(*((view[0],) + args))

            if isinstance(first, numbers.Real):
                out = numpy.empty(len(view), dtype=(numpy.int64 if isinstance(first, numbers.Integral) else numpy.float64))

            elif isinstance(first, tuple) and len(first) > 0 and all(isinstance(x, (numbers.Real, bool, numpy.bool_)) for x in first):
                if names is None:
                    names = ["f" + str(i) for i in range(len(first))]
                if len(names) != len(first):
                    raise TypeError("names has length {0} but function returns {1} numbers per row".format(len(names), len(first)))

                out = numpy.empty(len(view), dtype=zip(names, [numpy.bool_ if isinstance(x, (bool, numpy.bool_)) else numpy.int64 if isinstance(x, numbers.Integral) else numpy.float64 for x in first]))

            else:
                raise TypeError("function must return tuples of numbers (rows of a table)")

            out[0] = first
            i = 1
            if args == ():
                for datum in view[1:]:
                    out[i] = fcn(datum)
                    i += 1
            else:
                for datum in view[1:]:
                    out[i] = fcn(*((datum,) + args))
                    i += 1
                        
        elif isinstance(rtype, (nb.types.Integer, nb.types.Float)):
            out = numpy.empty(len(view), dtype=numpy.dtype(rtype.name))
            env = {fcnname: fcn}
            exec("""
def {fill}({view}, {out}{params}):
    {i} = 0
    for {datum} in {view}:
        {out}[{i}] = {fcn}({datum}{params})
        {i} += 1
""".format(fill=fillname,
           view=_newvar(avoid, "view"),
           out=_newvar(avoid, "out"),
           params="".join("," + x for x in params[1:]),
           i=_newvar(avoid, "i"),
           datum=_newvar(avoid, "datum"),
           fcn=fcnname), env)
            fill = trycompile(env[fillname], numba=numba)
            fill(*((view, out) + args))

        elif isinstance(rtype, (nb.types.Tuple, nb.types.UniTuple)) and len(rtype.types) > 0 and all(isinstance(x, (nb.types.Integer, nb.types.Float, nb.types.Boolean)) for x in rtype.types):
            if names is None:
                names = ["f" + str(i) for i in range(len(rtype.types))]
            if len(names) != len(rtype.types):
                raise TypeError("names has length {0} but function returns {1} numbers per row".format(len(names), len(rtype.types)))

            out = numpy.empty(len(view), dtype=zip(names, [numpy.dtype(x.name) for x in rtype.types]))
            outs = tuple(out[n] for n in names)

            outnames = [_newvar(avoid, "out" + str(i)) for i in range(len(names))]
            iname = _newvar(avoid, "i")
            tmpname = _newvar(avoid, "tmp")
            env = {fcnname: fcn}
            exec("""
def {fill}({view}, {outs}{params}):
    {i} = 0
    for {datum} in {view}:
        {tmp} = {fcn}({datum}{params})
        {assignments}
        {i} += 1
""".format(fill=fillname,
           view=_newvar(avoid, "view"),
           outs=",".join(outnames),
           params="".join("," + x for x in params[1:]),
           i=iname,
           datum=_newvar(avoid, "datum"),
           tmp=tmpname,
           fcn=fcnname,
           assignments="\n        ".join("{out}[{i}] = {tmp}[{j}]".format(out=out, i=iname, tmp=tmpname, j=j) for j, out in enumerate(outnames))), env)
            fill = trycompile(env[fillname], numba=numba)
            fill(*((view,) + outs + args))

        else:
            raise TypeError("function must return tuples of numbers (rows of a table)")

        return out

    else:
        raise TypeError("map can only be applied to a top-level OAMap proxy (List, Record, Tuple)")

################################################################ reduce

def reduce(data, tally, fcn, args=(), at="", numba=True):
    if not isinstance(args, tuple):
        try:
            args = tuple(args)
        except TypeError:
            args = (args,)

    if (isinstance(data, oamap.proxy.ListProxy) and data._whence == 0 and data._stride == 1) or (isinstance(data, oamap.proxy.Proxy) and data._index == 0):
        listnode = data._generator.schema.path(at)
        if not isinstance(lsitnode, oamap.schema.List):
            raise TypeError("path {0} does not refer to a list:\n\n    {1}".format(repr(at), listnode.__repr__(indent="    ")))
        if listnode.nullable:
            raise NotImplementedError("nullable; need to merge masks")

        listgenerator = data._generator.findbynames("List", listnode.namespace, starts=listnode.starts, stops=listnode.stops)
        viewstarts, viewstops = listgenerator._getstartsstops(data._arrays, data._cache)
        viewschema = listgenerator.namedschema()
        viewarrays = DualSource(data._arrays, data._generator.namespaces())
        viewoffsets = numpy.array([viewstarts.min(), viewstops.max()], dtype=oamap.generator.ListGenerator.posdtype)
        viewarrays.put(viewschema, viewoffsets[:1], viewoffsets[-1:])
        view = viewschema(viewarrays)

        params = fcn.__code__.co_varnames[:fcn.__code__.co_argcount]
        avoid = set(params)
        fcnname = _newvar(avoid, "fcn")
        fillname = _newvar(avoid, "fill")

        ptypes = paramtypes(args)
        if ptypes is not None:
            import numba as nb
            from oamap.compiler import typeof_generator
            ptypes = (typeof_generator(view._generator.content), numba.typeof(tally)) + ptypes
        fcn = trycompile(fcn, paramtypes=ptypes, numba=numba)
        rtype = returntype(fcn, ptypes)

        if rtype is not None:
            if numba.typeof(tally) != rtype:
                raise TypeError("function should return the same type as tally")

        env = {fcnname: fcn}
        exec("""
def {fill}({view}, {tally}{params}):
    for {datum} in {view}:
        {tally} = {fcn}({datum}, {tally}{params})
    return {tally}
""".format(fill=fillname,
           view=_newvar(avoid, "view"),
           tally=_newvar(avoid, "tally"),
           params="".join("," + x for x in params[1:]),
           datum=_newvar(avoid, "datum"),
           fcn=fcnname), env)
        fill = trycompile(env[fillname], numba=numba)
        return fill(*((view, tally) + args))

    else:
        raise TypeError("reduce can only be applied to a top-level OAMap proxy (List, Record, Tuple)")
