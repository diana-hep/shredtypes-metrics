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

import bisect
import codecs
import copy
import fnmatch
import json
import numbers
import re
import sys
from types import ModuleType

import numpy

import oamap.generator
import oamap.inference
import oamap.backend.packing
import oamap.extension.common
import oamap.proxy
import oamap.util
from oamap.util import OrderedDict

if sys.version_info[0] > 2:
    basestring = str
    unicode = str

# Common extensions
from oamap.extension.common import ByteString
from oamap.extension.common import UTF8String

# The "PLURTP" type system: Primitives, Lists, Unions, Records, Tuples, and Pointers

class Schema(object):
    _identifier = re.compile("[a-zA-Z][a-zA-Z_0-9]*")   # forbid starting with underscore in field names
    _baddelimiter = re.compile("[a-zA-Z_0-9]")          # could be confused with field names or integers

    def __init__(self, *args, **kwds):
        raise TypeError("Kind cannot be instantiated directly")

    @property
    def nullable(self):
        return self._nullable

    @nullable.setter
    def nullable(self, value):
        if value is not True and value is not False:
            raise TypeError("nullable must be True or False, not {0}".format(repr(value)))
        self._nullable = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("mask must be None or an array name (string), not {0}".format(repr(value)))
        self._mask = value

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, value):
        if not isinstance(value, basestring):
            raise TypeError("namespace must be a string, not {0}".format(repr(value)))
        self._namespace = value

    @property
    def packing(self):
        return self._packing

    @packing.setter
    def packing(self, value):
        if not (value is None or isinstance(value, oamap.backend.packing.PackedSource)):
            raise TypeError("packing must be None or a PackedSource, not {0}".format(repr(value)))
        self._packing = value

    def _packingcopy(self):
        if self._packing is None:
            return None
        else:
            return self._packing.copy()

    def _packingtojson(self):
        if self._packing is None:
            return None
        else:
            return self._packing.tojson()

    @staticmethod
    def _packingfromjson(packing):
        if packing is None:
            return None
        else:
            return oamap.backend.packing.PackedSource.fromjson(packing)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = value
            return
        if isinstance(value, basestring):
            match = self._identifier.match(value)
            if match is not None and len(match.group(0)) == len(value):
                self._name = value
                return
        raise TypeError("name must be None or a string matching /{0}/, not {1}".format(self._identifier.pattern, repr(value)))

    @property
    def doc(self):
        return self._doc

    @doc.setter
    def doc(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("doc must be None or a string, not {0}".format(repr(value)))
        self._doc = value

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def _labels(self):
        labels = []
        self._collectlabels(set(), labels)
        return labels
        
    def _label(self, labels):
        for index, label in enumerate(labels):
            if label is self:
                return "#{0}".format(index)
        return None

    def __ne__(self, other):
        return not self.__eq__(other)

    def show(self, stream=sys.stdout):
        out = self.__repr__(indent="")
        if stream is None:
            return out
        else:
            stream.write(out)
            stream.write("\n")

    def defaultnames(self, prefix="object", delimiter="-"):
        memo = {}
        pointers = []
        self._defaultnames(prefix, delimiter, memo, pointers, set())

        for pointer in pointers:
            if id(pointer.target) in memo:
                # internal
                pointer._positions = pointer._get_positions(memo[id(pointer)], delimiter) + delimiter + memo[id(pointer.target)]
            else:
                # external
                pointer._positions = pointer._get_positions(memo[id(pointer)], delimiter)
                pointer._target.defaultnames(pointer._get_external(memo[id(pointer)], delimiter), delimiter)

    @property
    def hasarraynames(self):
        return self._hasarraynames(set())

    def tojsonfile(self, file, explicit=False, *args, **kwds):
        json.dump(self.tojson(explicit=explicit), file, *args, **kwds)

    def tojsonstring(self, explicit=False, *args, **kwds):
        return json.dumps(self.tojson(explicit=explicit), *args, **kwds)

    def tojson(self, explicit=False):
        return self._tojson(explicit, self._labels(), set())

    @staticmethod
    def fromjsonfile(file, *args, **kwds):
        return Schema.fromjson(json.load(file, *args, **kwds))

    @staticmethod
    def fromjsonstring(data, *args, **kwds):
        return Schema.fromjson(json.loads(data, *args, **kwds))

    @staticmethod
    def fromjson(data):
        if isinstance(data, (basestring, dict)):
            labels = {}
            out = Schema._fromjson(data, labels)
            if not isinstance(out, Schema):
                raise TypeError("unresolved label: {0}".format(repr(out)))
            out._finalizefromjson(labels)
            return out
        else:
            raise TypeError("JSON for a Schema must be a string or a dict, not {0}".format(repr(data)))

    @staticmethod
    def _fromjson(data, labels):
        if isinstance(data, basestring) and data.startswith("#"):
            return data

        elif isinstance(data, basestring):
            return Primitive._fromjson(data, labels)

        elif isinstance(data, dict):
            tpe = data.get("type", "primitive")
            if tpe == "primitive":
                return Primitive._fromjson(data, labels)
            elif tpe == "list":
                return List._fromjson(data, labels)
            elif tpe == "union":
                return Union._fromjson(data, labels)
            elif tpe == "record":
                return Record._fromjson(data, labels)
            elif tpe == "tuple":
                return Tuple._fromjson(data, labels)
            elif tpe == "pointer":
                return Pointer._fromjson(data, labels)
            else:
                raise TypeError("unrecognized type argument for Schema from JSON: {0}".format(repr(tpe)))

        else:
            raise TypeError("unrecognized type for Schema from JSON: {0}".format(repr(data)))

    def deepcopy(self, **replacements):
        return self.replace(lambda x: x, **replacements)

    def path(self, path, parents=False):
        out = None
        for nodes in self._path((), path, (), set()):
            if out is None:
                if parents:
                    out = nodes
                else:
                    out = nodes[0]
            else:
                raise ValueError("path {0} matches more than one field in schema".format(repr(path)))

        if out is None:
            raise ValueError("path {0} does not match any fields in the schema".format(repr(path)))
        else:
            return out

    def paths(self, *paths, **options):
        parents = options.pop("parents", False)
        if len(options) > 0:
            raise TypeError("unrecognized options: {0}".format(", ".join(options)))
        for path in paths:
            for nodes in self._path((), path, (), set()):
                if parents:
                    yield nodes
                else:
                    yield nodes[0]

    def _path(self, loc, path, parents, memo):
        if fnmatch.fnmatchcase("/".join(loc), path):
            yield (self,) + parents

    def project(self, path):
        return self._keep((), [path], True, {})

    def keep(self, *paths):
        return self._keep((), paths, False, {})

    def drop(self, *paths):
        return self._drop((), paths, {})

    def contains(self, schema):
        return self._contains(schema, set())

    def _normalize_extension(self, extension):
        if isinstance(extension, ModuleType):
            recurse = False
            extension = extension.__dict__
        else:
            recurse = True

        if isinstance(extension, dict):
            extension = [extension[n] for n in sorted(extension)]

        try:
            iter(extension)
        except TypeError:
            raise TypeError("extension must be a module containing ExtendedGenerator classes or a dict or list (recursively) containing ExtendedGenerator classes")
        else:
            out = []
            for x in extension:
                if isinstance(x, type) and issubclass(x, oamap.generator.ExtendedGenerator):
                    out.append(x)
                elif recurse:
                    out.extend(self._normalize_extension(x))
            return out

    def fromdata(self, value, pointer_fromequal=False):
        import oamap.fill
        return self(oamap.fill.fromdata(value, generator=self, pointer_fromequal=pointer_fromequal))

    def fromiterdata(self, values, limit=lambda entries, arrayitems, arraybytes: False, pointer_fromequal=False):
        import oamap.fill
        return self(oamap.fill.fromiterdata(values, generator=self, limit=limit, pointer_fromequal=pointer_fromequal))

    def __call__(self, arrays, prefix="object", delimiter="-", extension=oamap.extension.common):
        return self.generator(prefix=prefix, delimiter=delimiter, extension=self._normalize_extension(extension))(arrays)

    def generator(self, prefix="object", delimiter="-", extension=oamap.extension.common):
        if self._baddelimiter.match(delimiter) is not None:
            raise ValueError("delimiters must not contain /{0}/".format(self._baddelimiter.pattern))
        cacheidx = [0]
        memo = OrderedDict()
        extension = self._normalize_extension(extension)
        return self._finalizegenerator(self._generator(prefix, delimiter, cacheidx, memo, set(), extension), cacheidx, memo, extension)

    def _get_name(self, prefix, delimiter):
        if self._name is not None:
            return prefix + delimiter + "N" + self._name
        else:
            return prefix

    def _get_mask(self, prefix, delimiter):
        if self._mask is None:
            return self._get_name(prefix, delimiter) + delimiter + "M"
        else:
            return self._mask

    def _finalizegenerator(self, out, cacheidx, memo, extension):
        allgenerators = list(memo.values())
        for generator in memo.values():
            if isinstance(generator, oamap.generator.PointerGenerator):
                # only assign pointer targets after all other types have been resolved
                target, prefix, delimiter = generator.target
                if id(target) in memo:
                    # the target points elsewhere in the type tree: link to that
                    generator._internal = True
                    if generator.schema.positions is None:
                        generator.positions = generator.positions + delimiter + memo[id(target)].derivedname
                    generator.target = memo[id(target)]
                    generator.schema.target = generator.target.schema
                else:
                    # the target is not in the type tree: resolve it now
                    memo2 = OrderedDict()   # new memo, but same cacheidx
                    generator._internal = False
                    generator.target = target._finalizegenerator(target._generator(generator.schema._get_external(prefix, delimiter), delimiter, cacheidx, memo2, set(), extension), cacheidx, memo2, extension)
                    generator.schema.target = generator.target.schema
                    for generator2 in memo2.values():
                        allgenerators.append(generator2)

        for generator in allgenerators:
            generator._cachelen = cacheidx[0]

        return out

    def case(self, obj):
        return obj in self

    def cast(self, obj):
        if obj in self:
            return obj
        else:
            raise TypeError("object is not a member of {0}".format(self))

################################################################ Primitives can be any Numpy type

class Primitive(Schema):
    def __init__(self, dtype, nullable=False, data=None, mask=None, namespace="", packing=None, name=None, doc=None, metadata=None):
        self.dtype = dtype
        self.nullable = nullable
        self.data = data
        self.mask = mask
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.doc = doc
        self.metadata = metadata

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if not isinstance(value, numpy.dtype):
            value = numpy.dtype(value)
        if value.hasobject:
            raise TypeError("dtypes containing objects are not allowed")
        if value.names is not None:
            for n in value.names:
                if self._identifier.match(n) is None:
                    raise TypeError("dtype names must be identifier strings; the name {0} is not an identifier (/{1}/)".format(repr(n), self._identifier.pattern))
            raise NotImplementedError("record-array dtypes are not supported yet")
        if value.subdtype is not None:
            raise NotImplementedError("multidimensional dtypes are not supported yet")
        self._dtype = value

    _byteorder_transform = {"!": True, ">": True, "<": False, "|": False, "=": numpy.dtype(">f8").isnative}

    @staticmethod
    def _dtype2str(dtype, delimiter):
        if dtype.names is not None:
            return delimiter.join(Primitive._dtype2str(dtype[n], delimiter) + delimiter + n for n in dtype.names)
        if dtype.subdtype is not None:
            subdtype, dims = dtype.subdtype
        else:
            subdtype, dims = dtype, ()
        return "D" + "".join(repr(x) + delimiter for x in dims) + (subdtype.kind.upper() if Primitive._byteorder_transform[subdtype.byteorder] else subdtype.kind) + repr(subdtype.itemsize)

    @staticmethod
    def _str2dtype(string, delimiter):
        out = []
        for _, dims, _, kind, itemsize, name in re.findall("(D(([1-9][0-9]*{0})*)([a-zA-Z])([1-9][0-9]*)({0}[a-zA-Z][a-zA-Z_0-9]*)?)".format(delimiter), string):
            if dims == "":
                dims = ()
            else:
                dims = tuple(int(x) for x in dims[:-len(delimiter)].split(delimiter))
            itemsize = itemsize
            name = name[len(delimiter):]
            if ord("A") <= ord(kind) <= ord("Z"):
                byteorder = ">"
            else:
                byteorder = "<"
            if kind == "S":
                descr = (kind + itemsize, dims)
            else:
                descr = (byteorder + kind.lower() + itemsize, dims)
            if name == "":
                out.append(descr)
            else:
                out.append((name,) + descr)
        if len(out) == 1:
            return numpy.dtype(out[0])
        else:
            return numpy.dtype(out)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("data must be None or an array name (string), not {0}".format(repr(value)))
        self._data = value

    def _hasarraynames(self, memo):
        return self._data is not None and (not self._nullable or self._mask is not None)

    def __repr__(self, labels=None, shown=None, indent=None):
        eq = "="

        if labels is None:
            labels = self._labels()
            shown = set()
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))

            args = [repr(self._dtype)]
            if self._nullable is not False:
                args.append("nullable" + eq + repr(self._nullable))
            if self._data is not None:
                args.append("data" + eq + repr(self._data))
            if self._mask is not None:
                args.append("mask" + eq + repr(self._mask))
            if self._namespace != "":
                args.append("namespace" + eq + repr(self._namespace))
            if self._packing is not None:
                args.append("packing" + eq + repr(self._packing))
            if self._name is not None:
                args.append("name" + eq + repr(self._name))
            if self._doc is not None:
                args.append("doc" + eq + repr(self._doc))
            if self._metadata is not None:
                args.append("metadata" + eq + repr(self._metadata))

            if label is None:
                return "Primitive(" + ", ".join(args) + ")"
            else:
                return label + ": Primitive(" + ", ".join(args) + ")"

        else:
            return label

    def _collectlabels(self, collection, labels):
        if id(self) not in collection:
            collection.add(id(self))
        else:
            labels.append(self)

    def _tojson(self, explicit, labels, shown):
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))
            if not explicit and self._nullable is False and self._data is None and self._mask is None and self._namespace == "" and self._packing is None and self._name is None and self._doc is None and self._metadata is None:
                return str(self._dtype)
            else:
                out = OrderedDict([("type", "primitive"), ("dtype", self._dtype2str(self._dtype, "-"))])
                if explicit or self._nullable is not False:
                    out["nullable"] = self._nullable
                if explicit or self._data is not None:
                    out["data"] = self._data
                if explicit or self._mask is not None:
                    out["mask"] = self._mask
                if explicit or self._namespace != "":
                    out["namespace"] = self._namespace
                if explicit or self._packing is not None:
                    out["packing"] = self._packingtojson()
                if explicit or self._name is not None:
                    out["name"] = self._name
                if explicit or self._doc is not None:
                    out["doc"] = self._doc
                if explicit or self._metadata is not None:
                    out["metadata"] = oamap.util.python2json(self._metadata)
                if explicit or label is not None:
                    out["label"] = label
                return out
        else:
            return label

    @staticmethod
    def _fromjson(data, labels):
        if isinstance(data, basestring):
            return Primitive(data)
        else:
            if "dtype" not in data:
                raise TypeError("Primitive Schema from JSON is missing argument 'dtype'")
            out = Primitive(Primitive._str2dtype(data["dtype"], "-"), nullable=data.get("nullable", False), data=data.get("data", None), mask=data.get("mask", None), namespace=data.get("namespace", ""), packing=Schema._packingfromjson(data.get("packing", None)), name=data.get("name", None), doc=data.get("doc", None), metadata=oamap.util.json2python(data.get("metadata", None)))
            if "label" in data:
                labels[data["label"]] = out
            return out

    def _finalizefromjson(self, labels):
        pass

    def copy(self, **replacements):
        if "dtype" not in replacements:
            replacements["dtype"] = self._dtype
        if "nullable" not in replacements:
            replacements["nullable"] = self._nullable
        if "data" not in replacements:
            replacements["data"] = self._data
        if "mask" not in replacements:
            replacements["mask"] = self._mask
        if "namespace" not in replacements:
            replacements["namespace"] = self._namespace
        if "packing" not in replacements:
            replacements["packing"] = self._packing
        if "name" not in replacements:
            replacements["name"] = self._name
        if "doc" not in replacements:
            replacements["doc"] = self._doc
        if "metadata" not in replacements:
            replacements["metadata"] = self._metadata
        return Primitive(**replacements)

    def replace(self, fcn, *args, **kwds):
        return fcn(Primitive(self._dtype, nullable=self._nullable, data=self._data, mask=self._mask, namespace=self._namespace, packing=self._packingcopy(), name=self._name, doc=self._doc, metadata=copy.deepcopy(self._metadata)), *args, **kwds)

    def _keep(self, loc, paths, project, memo):
        return self.deepcopy()

    def _drop(self, loc, paths, memo):
        return self.deepcopy()

    def _contains(self, schema, memo):
        return self == schema

    def __hash__(self):
        return hash((Primitive, self._dtype, self._nullable, self._data, self._mask, self._namespace, self._packing, self._name, self._doc, oamap.util.python2hashable(self._metadata)))

    def __eq__(self, other, memo=None):
        return isinstance(other, Primitive) and self._dtype == other._dtype and self._nullable == other._nullable and self._data == other._data and self._mask == other._mask and self._namespace == other._namespace and self._packing == other._packing and self._name == other._name and self._doc == other._doc and self._metadata == other._metadata

    def __contains__(self, value, memo=None):
        if value is None:
            return self.nullable

        def recurse(value, dims):
            if dims == ():
                if issubclass(self.dtype.type, (numpy.bool_, numpy.bool)):
                    return value is True or value is False

                elif issubclass(self.dtype.type, numpy.integer):
                    iinfo = numpy.iinfo(self.dtype.type)
                    return isinstance(value, numbers.Integral) and iinfo.min <= value <= iinfo.max

                elif issubclass(self.dtype.type, numpy.floating):
                    return isinstance(value, numbers.Real)

                elif issubclass(self.dtype.type, numpy.complex):
                    return isinstance(value, numbers.Complex)

                else:
                    raise TypeError("unexpected dtype: {0}".format(self.dtype))

            else:
                try:
                    iter(value)
                    len(value)
                except TypeError:
                    return False
                else:
                    return len(value) == dims[0] and all(recurse(x, dims[1:]) for x in value)

        if self._dtype.subdtype is None:
            return recurse(value, ())
        else:
            subdtype, dims = self._dtype.subdtype
            return recurse(value, dims)

    def _get_data(self, prefix, delimiter):
        if self._data is None:
            return self._get_name(prefix, delimiter) + delimiter + self._dtype2str(self._dtype, delimiter)
        else:
            return self._data

    def _defaultnames(self, prefix, delimiter, memo, pointers, nesting):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        if self._nullable:
            self._mask = self._get_mask(prefix, delimiter)
        self._data = self._get_data(prefix, delimiter)
        memo[id(self)] = prefix

    def _generator(self, prefix, delimiter, cacheidx, memo, nesting, extension):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        args = []

        if self._nullable:
            cls = oamap.generator.MaskedPrimitiveGenerator
            args.append(self._get_mask(prefix, delimiter))
            args.append(cacheidx[0]); cacheidx[0] += 1
        else:
            cls = oamap.generator.PrimitiveGenerator

        args.append(self._get_data(prefix, delimiter))
        args.append(cacheidx[0]); cacheidx[0] += 1

        args.append(self._dtype)
        args.append(self._namespace)
        args.append(self._packingcopy())
        args.append(self._name)
        args.append(prefix)
        args.append(self.copy(packing=self._packingcopy()))

        for ext in extension:
            if ext.matches(self):
                args.insert(0, cls)
                cls = ext
                break

        memo[id(self)] = cls(*args)
        return memo[id(self)]

################################################################ Lists may have arbitrary length

class List(Schema):
    def __init__(self, content, nullable=False, starts=None, stops=None, mask=None, namespace="", packing=None, name=None, doc=None, metadata=None):
        self.content = content
        self.nullable = nullable
        self.starts = starts
        self.stops = stops
        self.mask = mask
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.doc = doc
        self.metadata = metadata

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        if isinstance(value, basestring):
            value = Primitive(value)
        if not isinstance(value, Schema):
            raise TypeError("content must be a Schema, not {0}".format(repr(value)))
        self._content = value

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("starts must be None or an array name (string), not {0}".format(repr(value)))
        self._starts = value

    @property
    def stops(self):
        return self._stops

    @stops.setter
    def stops(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("stops must be None or an array name (string), not {0}".format(repr(value)))
        self._stops = value

    def _hasarraynames(self, memo):
        if id(self) in memo:
            return True
        else:
            memo.add(id(self))
            return self._starts is not None and self._stops is not None and (not self._nullable or self._mask is not None) and self._content._hasarraynames(memo)

    def __repr__(self, labels=None, shown=None, indent=None):
        eq = "=" if indent is None else " = "

        if labels is None:
            labels = self._labels()
            shown = set()
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))

            args = []
            if indent is None:
                args.append(self._content.__repr__(labels, shown, indent))
            if self._nullable is not False:
                args.append("nullable" + eq + repr(self._nullable))
            if self._starts is not None:
                args.append("starts" + eq + repr(self._starts))
            if self._stops is not None:
                args.append("stops" + eq + repr(self._stops))
            if self._mask is not None:
                args.append("mask" + eq + repr(self._mask))
            if self._namespace != "":
                args.append("namespace" + eq + repr(self._namespace))
            if self._packing is not None:
                args.append("packing" + eq + repr(self._packing))
            if self._name is not None:
                args.append("name" + eq + repr(self._name))
            if self._doc is not None:
                args.append("doc" + eq + repr(self._doc))
            if self._metadata is not None:
                args.append("metadata" + eq + repr(self._metadata))

            if indent is None:
                argstr = ", ".join(args)
            else:
                args.append("content" + eq + self._content.__repr__(labels, shown, indent + "  ").lstrip() + "\n" + indent)
                args[0] = "\n" + indent + "  " + args[0]
                argstr = ("," + "\n" + indent + "  ").join(args)

            if label is None:
                return "List(" + argstr + ")"
            else:
                return label + ": List(" + argstr + ")"

        else:
            return label

    def _tojson(self, explicit, labels, shown):
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))
            out = OrderedDict([("type", "list"), ("content", self._content._tojson(explicit, labels, shown))])
            if explicit or self._nullable is not False:
                out["nullable"] = self._nullable
            if explicit or self._starts is not None:
                out["starts"] = self._starts
            if explicit or self._stops is not None:
                out["stops"] = self._stops
            if explicit or self._mask is not None:
                out["mask"] = self._mask
            if explicit or self._namespace != "":
                out["namespace"] = self._namespace
            if explicit or self._packing is not None:
                out["packing"] = self._packingtojson()
            if explicit or self._name is not None:
                out["name"] = self._name
            if explicit or self._doc is not None:
                out["doc"] = self._doc
            if explicit or self._metadata is not None:
                out["metadata"] = oamap.util.python2json(self._metadata)
            if explicit or label is not None:
                out["label"] = label
            return out
        else:
            return label

    @staticmethod
    def _fromjson(data, labels):
        if "content" not in data:
            raise TypeError("List Schema from JSON is missing argument 'content'")
        out = List.__new__(List)
        out._content = Schema._fromjson(data["content"], labels)
        out.nullable = data.get("nullable", False)
        out.starts = data.get("starts", None)
        out.stops = data.get("stops", None)
        out.mask = data.get("mask", None)
        out.namespace = data.get("namespace", "")
        out.packing = Schema._packingfromjson(data.get("packing", None))
        out.name = data.get("name", None)
        out.doc = data.get("doc", None)
        out.metadata = oamap.util.json2python(data.get("metadata", None))
        if "label" in data:
            labels[data["label"]] = out
        return out

    def _finalizefromjson(self, labels):
        if isinstance(self._content, basestring):
            if self._content not in labels:
                raise TypeError("unresolved label: {0}".format(repr(self._content)))
            self._content = labels[self._content]
        else:
            self._content._finalizefromjson(labels)

    def _collectlabels(self, collection, labels):
        if id(self) not in collection:
            collection.add(id(self))
            self._content._collectlabels(collection, labels)
        else:
            labels.append(self)

    def copy(self, **replacements):
        if "content" not in replacements:
            replacements["content"] = self._content
        if "nullable" not in replacements:
            replacements["nullable"] = self._nullable
        if "starts" not in replacements:
            replacements["starts"] = self._starts
        if "stops" not in replacements:
            replacements["stops"] = self._stops
        if "mask" not in replacements:
            replacements["mask"] = self._mask
        if "namespace" not in replacements:
            replacements["namespace"] = self._namespace
        if "packing" not in replacements:
            replacements["packing"] = self._packing
        if "name" not in replacements:
            replacements["name"] = self._name
        if "doc" not in replacements:
            replacements["doc"] = self._doc
        if "metadata" not in replacements:
            replacements["metadata"] = self._metadata
        return List(**replacements)

    def replace(self, fcn, *args, **kwds):
        return fcn(List(self._content.replace(fcn, *args, **kwds), nullable=self._nullable, starts=self._starts, stops=self._stops, mask=self._mask, namespace=self._namespace, packing=self._packingcopy(), name=self._name, doc=self._doc, metadata=copy.deepcopy(self._metadata)), *args, **kwds)

    def _path(self, loc, path, parents, memo):
        nodes = None
        for nodes in Schema._path(self, loc, path, parents, memo):
            yield nodes
        if nodes is None:
            for nodes in self._content._path(loc, path, (self,) + parents, memo):
                yield nodes

    def _keep(self, loc, paths, project, memo):
        content = self.content._keep(loc, paths, project, memo)
        if content is None:
            return None
        else:
            return self.copy(content=content)

    def _drop(self, loc, paths, memo):
        content = self.content._drop(loc, paths, memo)
        if content is None:
            return None
        else:
            return self.copy(content=content)

    def _contains(self, schema, memo):
        if self == schema:
            return True
        else:
            return self._content._contains(schema, memo)

    def __hash__(self):
        return hash((List, self._content, self._nullable, self._starts, self._stops, self._mask, self._namespace, self._packing, self._name, self._doc, oamap.util.python2hashable(self._metadata)))

    def __eq__(self, other, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)] == id(other)
        if not (isinstance(other, List) and self._nullable == other._nullable and self._starts == other._starts and self._stops == other._stops and self._mask == other._mask and self._namespace == other._namespace and self._packing == other._packing and self._name == other._name and self._doc == other._doc and self._metadata == other._metadata):
            return False
        memo[id(self)] = id(other)
        return self.content.__eq__(other.content, memo)

    def __contains__(self, value, memo=None):
        if memo is None:
            memo = {}
        if value is None:
            return self.nullable
        try:
            iter(value)
        except TypeError:
            return False
        else:
            for x in value:
                memo2 = dict(memo) if len(memo) > 0 else memo
                if not self.content.__contains__(x, memo2):
                    return False
            return True

    def _get_starts(self, prefix, delimiter):
        if self._starts is None:
            return self._get_name(prefix, delimiter) + delimiter + "B"
        else:
            return self._starts

    def _get_stops(self, prefix, delimiter):
        if self._stops is None:
            return self._get_name(prefix, delimiter) + delimiter + "E"
        else:
            return self._stops

    def _get_content(self, prefix, delimiter):
        return self._get_name(prefix, delimiter) + delimiter + "L"

    def _defaultnames(self, prefix, delimiter, memo, pointers, nesting):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        if self._nullable:
            self._mask = self._get_mask(prefix, delimiter)
        self._starts = self._get_starts(prefix, delimiter)
        self._stops = self._get_stops(prefix, delimiter)
        self._content._defaultnames(self._get_content(prefix, delimiter), delimiter, memo, pointers, nesting.union(set([id(self)])))
        memo[id(self)] = prefix

    def _generator(self, prefix, delimiter, cacheidx, memo, nesting, extension):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        args = []

        if self._nullable:
            cls = oamap.generator.MaskedListGenerator
            args.append(self._get_mask(prefix, delimiter))
            args.append(cacheidx[0]); cacheidx[0] += 1
        else:
            cls = oamap.generator.ListGenerator

        args.append(self._get_starts(prefix, delimiter))
        args.append(cacheidx[0]); cacheidx[0] += 1

        args.append(self._get_stops(prefix, delimiter))
        args.append(cacheidx[0]); cacheidx[0] += 1

        contentgen = self._content._generator(self._get_content(prefix, delimiter), delimiter, cacheidx, memo, nesting.union(set([id(self)])), extension)
        args.append(contentgen)
        args.append(self._namespace)
        args.append(self._packingcopy())
        args.append(self._name)
        args.append(prefix)
        args.append(self.copy(content=contentgen.schema, packing=self._packingcopy()))

        for ext in extension:
            if ext.matches(self):
                args.insert(0, cls)
                cls = ext
                break

        memo[id(self)] = cls(*args)
        return memo[id(self)]

################################################################ Unions may be one of several types

class Union(Schema):
    def __init__(self, possibilities, nullable=False, tags=None, offsets=None, mask=None, namespace="", packing=None, name=None, doc=None, metadata=None):
        self.possibilities = possibilities
        self.nullable = nullable
        self.tags = tags
        self.offsets = offsets
        self.mask = mask
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.doc = doc
        self.metadata = metadata

    @property
    def possibilities(self):
        return tuple(self._possibilities)

    @possibilities.setter
    def possibilities(self, value):
        self._extend(value, [])

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("tags must be None or an array name (string), not {0}".format(repr(value)))
        self._tags = value

    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("offsets must be None or an array name (string), not {0}".format(repr(value)))
        self._offsets = value

    def _extend(self, possibilities, start):
        trial = []
        try:
            for i, x in enumerate(possibilities):
                if isinstance(x, basestring):
                    x = Primitive(x)
                assert isinstance(x, Schema), "possibilities must be an iterable of Schemas; item at {0} is {1}".format(i, repr(x))
                trial.append(x)
        except TypeError:
            raise TypeError("possibilities must be an iterable of Schemas, not {0}".format(repr(possibilities)))
        except AssertionError as err:
            raise TypeError(err.message)
        self._possibilities = start + trial

    def append(self, possibility):
        if isinstance(possibility, basestring):
            possibility = Primitive(possibility)
        if not isinstance(possibility, Schema):
            raise TypeError("possibilities must be Schemas, not {0}".format(repr(possibility)))
        self._possibilities.append(possibility)

    def insert(self, index, possibility):
        if isinstance(possibility, basestring):
            possibility = Primitive(possibility)
        if not isinstance(possibility, Schema):
            raise TypeError("possibilities must be Schemas, not {0}".format(repr(possibility)))
        self._possibilities.insert(index, possibility)

    def extend(self, possibilities):
        self._extend(possibilities, self._possibilities)

    def __getitem__(self, index):
        return self._possibilities[index]

    def __setitem__(self, index, value):
        if not isinstance(index, numbers.Integral):
            raise TypeError("possibility index must be an integer, not {0}".format(repr(index)))
        if isinstance(value, basestring):
            value = Primitive(value)
        if not isinstance(value, Schema):
            raise TypeError("possibilities must be Schemas, not {0}".format(repr(value)))
        self._possibilities[index] = value

    def _hasarraynames(self, memo):
        if id(self) in memo:
            return True
        else:
            memo.add(id(self))
            return self._tags is not None and self._offsets is not None and (not self._nullable or self._mask is not None) and all(x._hasarraynames(memo) for x in self._possibilities)

    def __repr__(self, labels=None, shown=None, indent=None):
        eq = "=" if indent is None else " = "

        if labels is None:
            labels = self._labels()
            shown = set()
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))

            args = []
            if indent is None:
                args.append("[" + ", ".join(x.__repr__(labels, shown, indent) for x in self._possibilities) + "]")
            if self._nullable is not False:
                args.append("nullable" + eq + repr(self._nullable))
            if self._tags is not None:
                args.append("tags" + eq + repr(self._tags))
            if self._offsets is not None:
                args.append("offsets" + eq + repr(self._offsets))
            if self._mask is not None:
                args.append("mask" + eq + repr(self._mask))
            if self._namespace != "":
                args.append("namespace" + eq + repr(self._namespace))
            if self._packing is not None:
                args.append("packing" + eq + repr(self._packing))
            if self._name is not None:
                args.append("name" + eq + repr(self._name))
            if self._doc is not None:
                args.append("doc" + eq + repr(self._doc))
            if self._metadata is not None:
                args.append("metadata" + eq + repr(self._metadata))

            if indent is None:
                argstr = ", ".join(args)
            else:
                args.append("possibilities" + eq + "[\n" + indent + "    " + (",\n" + indent + "    ").join(x.__repr__(labels, shown, indent + "    ").lstrip() for x in self._possibilities) + "\n" + indent + "  ]")
                args[0] = "\n" + indent + "  " + args[0]
                argstr = ("," + "\n" + indent + "  ").join(args)

            if label is None:
                return "Union(" + argstr + ")"
            else:
                return label + ": Union(" + argstr + ")"

        else:
            return label

    def _tojson(self, explicit, labels, shown):
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))
            out = OrderedDict([("type", "union"), ("possibilities", [x._tojson(explicit, labels, shown) for x in self._possibilities])])
            if explicit or self._nullable is not False:
                out["nullable"] = self._nullable
            if explicit or self._tags is not None:
                out["tags"] = self._tags
            if explicit or self._offsets is not None:
                out["offsets"] = self._offsets
            if explicit or self._mask is not None:
                out["mask"] = self._mask
            if explicit or self._namespace != "":
                out["namespace"] = self._namespace
            if explicit or self._packing is not None:
                out["packing"] = self._packingtojson()
            if explicit or self._name is not None:
                out["name"] = self._name
            if explicit or self._doc is not None:
                out["doc"] = self._doc
            if explicit or self._metadata is not None:
                out["metadata"] = oamap.util.python2json(self._metadata)
            if explicit or label is not None:
                out["label"] = label
            return out
        else:
            return label

    @staticmethod
    def _fromjson(data, labels):
        if "possibilities" not in data:
            raise TypeError("Union Schema from JSON is missing argument 'possibilities'")
        if not isinstance(data["possibilities"], list):
            raise TypeError("argument 'possibilities' for Union Schema from JSON should be a list, not {0}".format(repr(data["possibilities"])))
        out = Union.__new__(Union)
        out.possibilities = [Schema._fromjson(x, labels) for x in data["possibilities"]]
        out.nullable = data.get("nullable", False)
        out.tags = data.get("tags", None)
        out.offsets = data.get("offsets", None)
        out.mask = data.get("mask", None)
        out.namespace = data.get("namespace", "")
        out.packing = Schema._packingfromjson(data.get("packing", None))
        out.name = data.get("name", None)
        out.doc = data.get("doc", None)
        out.metadata = oamap.util.json2python(data.get("metadata", None))
        if "label" in data:
            labels[data["label"]] = out
        return out

    def _finalizefromjson(self, labels):
        for i in range(len(self._possibilities)):
            if isinstance(self._possibilities[i], basestring):
                if self._possibilities[i] not in labels:
                    raise TypeError("unresolved label: {0}".format(repr(self._possibilities[i])))
                self._possibilities[i] = labels[self._possibilities[i]]
            else:
                self._possibilities[i]._finalizefromjson(labels)

    def _collectlabels(self, collection, labels):
        if id(self) not in collection:
            collection.add(id(self))
            for possibility in self._possibilities:
                possibility._collectlabels(collection, labels)
        else:
            labels.append(self)

    def copy(self, **replacements):
        if "possibilities" not in replacements:
            replacements["possibilities"] = self._possibilities
        if "nullable" not in replacements:
            replacements["nullable"] = self._nullable
        if "tags" not in replacements:
            replacements["tags"] = self._tags
        if "offsets" not in replacements:
            replacements["offsets"] = self._offsets
        if "mask" not in replacements:
            replacements["mask"] = self._mask
        if "namespace" not in replacements:
            replacements["namespace"] = self._namespace
        if "packing" not in replacements:
            replacements["packing"] = self._packing
        if "name" not in replacements:
            replacements["name"] = self._name
        if "doc" not in replacements:
            replacements["doc"] = self._doc
        if "metadata" not in replacements:
            replacements["metadata"] = self._metadata
        return Union(**replacements)

    def replace(self, fcn, *args, **kwds):
        return fcn(Union([x.replace(fcn, *args, **kwds) for x in self._possibilities], nullable=self._nullable, tags=self._tags, offsets=self._offsets, mask=self._mask, namespace=self._namespace, packing=self._packingcopy(), name=self._name, doc=self._doc, metadata=copy.deepcopy(self._metadata)), *args, **kwds)

    def _path(self, loc, path, parents, memo):
        nodes = None
        for nodes in Schema._path(self, loc, path, parents, memo):
            yield nodes
        if nodes is None:
            for possibility in self._possibilities:
                for nodes in possibility._path(loc, path, (self,) + parents, memo):
                    yield nodes

    def _keep(self, loc, paths, project, memo):
        possibilities = []
        for x in self._possibilities:
            p = self._keep(loc, paths, project, memo)
            if p is None:
                return None
            else:
                possibilities.append(p)
        return self.copy(possibilities)

    def _drop(self, loc, paths, memo):
        possibilities = []
        for x in self._possibilities:
            p = self._drop(loc, paths, memo)
            if p is None:
                return None
            else:
                possibilities.append(p)
        return self.copy(possibilities)

    def _contains(self, schema, memo):
        if self == schema:
            return True
        else:
            return any(x._contains(schema, memo) for x in self._possibilities)

    def __hash__(self):
        return hash((Union, self._possibilities, self._nullable, self._tags, self._offsets, self._mask, self._namespace, self._packing, self._name, self._doc, oamap.util.python2hashable(self._metadata)))

    def __eq__(self, other, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)] == id(other)
        if not (isinstance(other, Union) and len(self._possibilities) == len(other._possibilities) and self._nullable == other._nullable and self._tags == other._tags and self._offsets == other._offsets and self._mask == other._mask and self._namespace == other._namespace and self._packing == other._packing and self._name == other._name and self._doc == other._doc and self._metadata == other._metadata):
            return False
        memo[id(self)] = id(other)
        return all(x.__eq__(y, memo) for x, y in zip(self.possibilities, other.possibilities))

    def __contains__(self, value, memo=None):
        if memo is None:
            memo = {}
        if value is None:
            return self._nullable or any(x._nullable for x in self._possibilities)
        return any(x.__contains__(value, memo) for x in self.possibilities)

    def _get_tags(self, prefix, delimiter):
        if self._tags is None:
            return self._get_name(prefix, delimiter) + delimiter + "T"
        else:
            return self._tags

    def _get_offsets(self, prefix, delimiter):
        if self._offsets is None:
            return self._get_name(prefix, delimiter) + delimiter + "O"
        else:
            return self._offsets

    def _get_possibility(self, prefix, delimiter, i):
        return self._get_name(prefix, delimiter) + delimiter + "U" + repr(i)

    def _defaultnames(self, prefix, delimiter, memo, pointers, nesting):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        if self._nullable:
            self._mask = self._get_mask(prefix, delimiter)
        self._tags = self._get_tags(prefix, delimiter)
        self._offsets = self._get_offsets(prefix, delimiter)
        for i, x in enumerate(self._possibilities):
            x._defaultnames(self._get_possibility(prefix, delimiter, i), delimiter, memo, pointers, nesting.union(set([id(self)])))
        memo[id(self)] = prefix

    def _generator(self, prefix, delimiter, cacheidx, memo, nesting, extension):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        args = []

        if self._nullable:
            cls = oamap.generator.MaskedUnionGenerator
            args.append(self._get_mask(prefix, delimiter))
            args.append(cacheidx[0]); cacheidx[0] += 1
        else:
            cls = oamap.generator.UnionGenerator

        args.append(self._get_tags(prefix, delimiter))
        args.append(cacheidx[0]); cacheidx[0] += 1

        args.append(self._get_offsets(prefix, delimiter))
        args.append(cacheidx[0]); cacheidx[0] += 1

        possibilitiesgen = [x._generator(self._get_possibility(prefix, delimiter, i), delimiter, cacheidx, memo, nesting.union(set([id(self)])), extension) for i, x in enumerate(self._possibilities)]
        args.append(possibilitiesgen)
        args.append(self._namespace)
        args.append(self._packingcopy())
        args.append(self._name)
        args.append(prefix)
        args.append(self.copy(possibilities=[x.schema for x in possibilitiesgen], packing=self._packingcopy()))

        for ext in extension:
            if ext.matches(self):
                args.insert(0, cls)
                cls = ext
                break

        memo[id(self)] = cls(*args)
        return memo[id(self)]

################################################################ Records contain fields of known types

class Record(Schema):
    def __init__(self, fields, nullable=False, mask=None, namespace="", packing=None, name=None, doc=None, metadata=None):
        self.fields = fields
        self.nullable = nullable
        self.mask = mask
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.doc = doc
        self.metadata = metadata

    @property
    def fields(self):
        return dict(self._fields)

    @fields.setter
    def fields(self, value):
        self._extend(value, [])

    def _extend(self, fields, start):
        trial = []
        try:
            for n, x in fields.items():
                assert isinstance(n, basestring), "fields must be a dict from identifier strings to Schemas; the key {0} is not a string".format(repr(n))
                matches = self._identifier.match(n)
                assert matches is not None and len(matches.group(0)) == len(n), "fields must be a dict from identifier strings to Schemas; the key {0} is not an identifier (/{1}/)".format(repr(n), self._identifier.pattern)
                if isinstance(x, basestring):
                    x = Primitive(x)
                assert isinstance(x, Schema), "fields must be a dict from identifier strings to Schemas; the value at key {0} is {1}".format(repr(n), repr(x))
                trial.append((n, x))
        except AttributeError:
            raise TypeError("fields must be a dict from strings to Schemas; {0} is not a dict".format(repr(fields)))
        except AssertionError as err:
            raise TypeError(err.message)
        self._fields = OrderedDict(start + trial)

    def __getitem__(self, index):
        return self._fields[index]

    def __setitem__(self, index, value):
        if not isinstance(index, basestring):
            raise TypeError("field keys must be strings, not {0}".format(repr(index)))
        if isinstance(value, basestring):
            value = Primitive(value)
        if not isinstance(value, Schema):
            raise TypeError("field values must be Schemas, not {0}".format(repr(value)))
        self._fields[index] = value

    def __delitem__(self, index):
        del self._fields[index]

    def rename(self, fromfield, tofield):
        if not self.hasarraynames:
            raise ValueError("cannot rename a field in a schema without fixed array names; try calling defaultnames() to assign fixed array names")
        renamed = []    # but maintain order
        found = False
        for n, x in self._fields.items():
            if n == fromfield:
                renamed.append((tofield, x))
                found = True
            else:
                renamed.append((n, x))
        if not found:
            raise KeyError("field not found: {0}".format(repr(fromfield)))
        self._fields = OrderedDict(renamed)

    def _hasarraynames(self, memo):
        if id(self) in memo:
            return True
        else:
            memo.add(id(self))
            return (not self._nullable or self._mask is not None) and all(x._hasarraynames(memo) for x in self._fields.values())

    def __repr__(self, labels=None, shown=None, indent=None):
        eq = "=" if indent is None else " = "

        if labels is None:
            labels = self._labels()
            shown = set()
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))

            args = []
            if indent is None:
                args.append("{" + ", ".join("{0}: {1}".format(repr(n), x.__repr__(labels, shown, indent)) for n, x in self._fields.items()) + "}")
            if self._nullable is not False:
                args.append("nullable" + eq + repr(self._nullable))
            if self._mask is not None:
                args.append("mask" + eq + repr(self._mask))
            if self._namespace != "":
                args.append("namespace" + eq + repr(self._namespace))
            if self._packing is not None:
                args.append("packing" + eq + repr(self._packing))
            if self._name is not None:
                args.append("name" + eq + repr(self._name))
            if self._doc is not None:
                args.append("doc" + eq + repr(self._doc))
            if self._metadata is not None:
                args.append("metadata" + eq + repr(self._metadata))

            if indent is None:
                argstr = ", ".join(args)
            else:
                args.append("fields" + eq + "{\n" + indent + "    " + (",\n" + indent + "    ").join("{0}: {1}".format(repr(n), x.__repr__(labels, shown, indent + "    ").lstrip()) for n, x in self._fields.items()) + "\n" + indent + "  }")
                args[0] = "\n" + indent + "  " + args[0]
                argstr = ("," + "\n" + indent + "  ").join(args)

            if label is None:
                return "Record(" + argstr + ")"
            else:
                return label + ": Record(" + argstr + ")"

        else:
            return label

    def _tojson(self, explicit, labels, shown):
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))
            out = OrderedDict([("type", "record"), ("fields", [[n, x._tojson(explicit, labels, shown)] for n, x in self._fields.items()])])
            if explicit or self._nullable is not False:
                out["nullable"] = self._nullable
            if explicit or self._mask is not None:
                out["mask"] = self._mask
            if explicit or self._namespace != "":
                out["namespace"] = self._namespace
            if explicit or self._packing is not None:
                out["packing"] = self._packingtojson()
            if explicit or self._name is not None:
                out["name"] = self._name
            if explicit or self._doc is not None:
                out["doc"] = self._doc
            if explicit or self._metadata is not None:
                out["metadata"] = oamap.util.python2json(self._metadata)
            if explicit or label is not None:
                out["label"] = label
            return out
        else:
            return label

    @staticmethod
    def _fromjson(data, labels):
        if "fields" not in data:
            raise TypeError("Record Schema from JSON is missing argument 'fields'")
        out = Record.__new__(Record)
        if isinstance(data["fields"], list) and all(len(x) == 2 and isinstance(x[0], basestring) for x in data["fields"]):
            out._fields = OrderedDict((n, Schema._fromjson(x, labels)) for n, x in data["fields"])
        elif isinstance(data["fields"], dict) and all(isinstance(x, basestring) for x in data["fields"]):
            out._fields = OrderedDict((n, Schema._fromjson(data["fields"][n], labels)) for n in sorted(data["fields"]))
        else:
            raise TypeError("argument 'fields' for Record Schema from JSON should be a list or dict of key-value pairs (in which the keys are strings), not {0}".format(repr(data["fields"])))
        out.nullable = data.get("nullable", False)
        out.mask = data.get("mask", None)
        out.namespace = data.get("namespace", "")
        out.packing = Schema._packingfromjson(data.get("packing", None))
        out.name = data.get("name", None)
        out.doc = data.get("doc", None)
        out.metadata = oamap.util.json2python(data.get("metadata", None))
        if "label" in data:
            labels[data["label"]] = out
        return out

    def _finalizefromjson(self, labels):
        for n in list(self._fields.keys()):
            if isinstance(self._fields[n], basestring):
                if self._fields[n] not in labels:
                    raise TypeError("unresolved label: {0}".format(repr(self._fields[n])))
                self._fields[n] = labels[self._fields[n]]
            else:
                self._fields[n]._finalizefromjson(labels)

    def _collectlabels(self, collection, labels):
        if id(self) not in collection:
            collection.add(id(self))
            for field in self._fields.values():
                field._collectlabels(collection, labels)
        else:
            labels.append(self)

    def copy(self, **replacements):
        if "fields" not in replacements:
            replacements["fields"] = self._fields
        if "nullable" not in replacements:
            replacements["nullable"] = self._nullable
        if "mask" not in replacements:
            replacements["mask"] = self._mask
        if "namespace" not in replacements:
            replacements["namespace"] = self._namespace
        if "packing" not in replacements:
            replacements["packing"] = self._packing
        if "name" not in replacements:
            replacements["name"] = self._name
        if "doc" not in replacements:
            replacements["doc"] = self._doc
        if "metadata" not in replacements:
            replacements["metadata"] = self._metadata
        return Record(**replacements)

    def replace(self, fcn, *args, **kwds):
        return fcn(Record(OrderedDict((n, x.replace(fcn, *args, **kwds)) for n, x in self._fields.items()), nullable=self._nullable, mask=self._mask, namespace=self._namespace, packing=self._packingcopy(), name=self._name, doc=self._doc, metadata=copy.deepcopy(self._metadata)), *args, **kwds)

    def _path(self, loc, path, parents, memo):
        nodes = None
        for nodes in Schema._path(self, loc, path, parents, memo):
            yield nodes
        if nodes is None:
            for n, x in self._fields.items():
                for nodes in x._path(loc + (n,), path, (self,) + parents, memo):
                    yield nodes

    def _keep(self, loc, paths, project, memo):
        fields = OrderedDict()
        for n, x in self._fields.items():
            if any(fnmatch.fnmatchcase("/".join(loc + (n,)), p) for p in paths):
                fields[n] = x
            elif any(fnmatch.fnmatchcase("/".join(loc + (n,)), "/".join(p.split("/")[:len(loc) + 1])) for p in paths):
                f = x._keep(loc + (n,), paths, project, memo)
                if f is not None:
                    fields[n] = f
        if len(fields) == 0:
            return None
        elif project and len(fields) == 1:
            out, = fields.values()
            return out
        else:
            return self.copy(fields=fields)

    def _drop(self, loc, paths, memo):
        fields = OrderedDict()
        for n, x in self._fields.items():
            if not any(fnmatch.fnmatchcase("/".join(loc + (n,)), p) for p in paths):
                f = x._drop(loc + (n,), paths, memo)
                if f is not None:
                    fields[n] = f
        if len(fields) == 0:
            return None
        else:
            return self.copy(fields=fields)

    def _contains(self, schema, memo):
        if self == schema:
            return True
        else:
            return any(x._contains(schema, memo) for x in self._fields.values())

    def __hash__(self):
        return hash((Record, tuple(self._fields.items()), self._nullable, self._mask, self._namespace, self._packing, self._name, self._doc, oamap.util.python2hashable(self._metadata)))

    def __eq__(self, other, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)] == id(other)
        if not (isinstance(other, Record) and set(self._fields) == set(other._fields) and self._nullable == other._nullable and self._mask == other._mask and self._namespace == other._namespace and self._packing == other._packing and self._name == other._name and self._doc == other._doc and self._metadata == other._metadata):
            return False
        memo[id(self)] = id(other)
        return all(self._fields[n].__eq__(other._fields[n], memo) for n in self._fields)

    def __contains__(self, value, memo=None):
        if memo is None:
            memo = {}
        if value is None:
            return self.nullable
        if isinstance(value, dict):
            return all(n in value and x.__contains__(value[n], memo) for n, x in self._fields.items())
        elif isinstance(value, tuple) and hasattr(value, "_fields"):
            return all(n in value._fields and x.__contains__(getattr(value, n), memo) for n, x in self._fields.items())
        elif isinstance(value, (list, tuple)):
            return False
        else:
            return all(hasattr(value, n) and x.__contains__(getattr(value, n), memo) for n, x in self._fields.items())

    def _get_field(self, prefix, delimiter, n):
        return self._get_name(prefix, delimiter) + delimiter + "F" + n

    def _defaultnames(self, prefix, delimiter, memo, pointers, nesting):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        if self._nullable:
            self._mask = self._get_mask(prefix, delimiter)
        for n, x in self._fields.items():
            x._defaultnames(self._get_field(prefix, delimiter, n), delimiter, memo, pointers, nesting.union(set([id(self)])))
        memo[id(self)] = prefix

    def _generator(self, prefix, delimiter, cacheidx, memo, nesting, extension):
        if len(self._fields) == 0:
            raise TypeError("Record has no fields")
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        args = []

        if self._nullable:
            cls = oamap.generator.MaskedRecordGenerator
            args.append(self._get_mask(prefix, delimiter))
            args.append(cacheidx[0]); cacheidx[0] += 1
        else:
            cls = oamap.generator.RecordGenerator

        fieldsgen = OrderedDict([(n, self._fields[n]._generator(self._get_field(prefix, delimiter, n), delimiter, cacheidx, memo, nesting.union(set([id(self)])), extension)) for n in sorted(self._fields)])
        args.append(fieldsgen)
        args.append(self._namespace)
        args.append(self._packingcopy())
        args.append(self._name)
        args.append(prefix)
        args.append(self.copy(fields=OrderedDict((n, x.schema) for n, x in fieldsgen.items()), packing=self._packingcopy()))

        for ext in extension:
            if ext.matches(self):
                args.insert(0, cls)
                cls = ext
                break

        memo[id(self)] = cls(*args)
        return memo[id(self)]

################################################################ Tuples are like records but with an order instead of field names

class Tuple(Schema):
    def __init__(self, types, nullable=False, mask=None, namespace="", packing=None, name=None, doc=None, metadata=None):
        self.types = types
        self.nullable = nullable
        self.mask = mask
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.doc = doc
        self.metadata = metadata

    @property
    def types(self):
        return tuple(self._types)

    @types.setter
    def types(self, value):
        self._extend(value, [])

    def _extend(self, types, start):
        trial = []
        try:
            for i, x in enumerate(types):
                if isinstance(x, basestring):
                    x = Primitive(x)
                assert isinstance(x, Schema), "types must be an iterable of Schemas; item at {0} is {1}".format(i, repr(x))
                trial.append(x)
        except TypeError:
            raise TypeError("types must be an iterable of Schemas, not {0}".format(repr(types)))
        except AssertionError as err:
            raise TypeError(err.message)
        self._types = start + trial

    def append(self, item):
        if isinstance(item, basestring):
            item = Primitive(item)
        if not isinstance(item, Schema):
            raise TypeError("types must be Schemas, not {0}".format(repr(item)))
        self._types.append(item)

    def insert(self, index, item):
        if isinstance(item, basestring):
            item = Primitive(item)
        if not isinstance(item, Schema):
            raise TypeError("types must be Schemas, not {0}".format(repr(item)))
        self._types.insert(index, item)

    def extend(self, types):
        self._extend(types, self._types)

    def __getitem__(self, index):
        return self._types[index]

    def __setitem__(self, index, value):
        if not isinstance(index, numbers.Integral):
            raise TypeError("types index must be an integer, not {0}".format(repr(index)))
        if isinstance(value, basestring):
            value = Primitive(value)
        if not isinstance(item, Schema):
            raise TypeError("types must be Schemas, not {0}".format(repr(value)))
        self._types[index] = value

    def _hasarraynames(self, memo):
        if id(self) in memo:
            return True
        else:
            memo.add(id(self))
            return (not self._nullable or self._mask is not None) and all(x._hasarraynames(memo) for x in self._types)

    def __repr__(self, labels=None, shown=None, indent=None):
        eq = "=" if indent is None else " = "

        if labels is None:
            labels = self._labels()
            shown = set()
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))

            args = []
            if indent is None:
                args.append("[" + ", ".join(x.__repr__(labels, shown) for x in self._types) + "]")
            if self._nullable is not False:
                args.append("nullable" + eq + repr(self._nullable))
            if self._mask is not None:
                args.append("mask" + eq + repr(self._mask))
            if self._namespace != "":
                args.append("namespace" + eq + repr(self._namespace))
            if self._packing is not None:
                args.append("packing" + eq + repr(self._packing))
            if self._name is not None:
                args.append("name" + eq + repr(self._name))
            if self._doc is not None:
                args.append("doc" + eq + repr(self._doc))
            if self._metadata is not None:
                args.append("metadata" + eq + repr(self._metadata))

            if indent is None:
                argstr = ", ".join(args)
            else:
                args.append("types" + eq + "[\n" + indent + "    " + (",\n" + indent + "    ").join(x.__repr__(labels, shown, indent + "    ").lstrip() for x in self._types) + "\n" + indent + "  ]")
                args[0] = "\n" + indent + "  " + args[0]
                argstr = ("," + "\n" + indent + "  ").join(args)

            if label is None:
                return "Tuple(" + argstr + ")"
            else:
                return label + ": Tuple(" + argstr + ")"

        return label

    def _tojson(self, explicit, labels, shown):
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))
            out = OrderedDict([("type", "tuple"), ("types", [x._tojson(explicit, labels, shown) for x in self._types])])
            if explicit or self._nullable is not False:
                out["nullable"] = self._nullable
            if explicit or self._mask is not None:
                out["mask"] = self._mask
            if explicit or self._namespace != "":
                out["namespace"] = self._namespace
            if explicit or self._packing is not None:
                out["packing"] = self._packingtojson()
            if explicit or self._name is not None:
                out["name"] = self._name
            if explicit or self._doc is not None:
                out["doc"] = self._doc
            if explicit or self._metadata is not None:
                out["metadata"] = oamap.util.python2json(self._metadata)
            if explicit or label is not None:
                out["label"] = label
            return out
        else:
            return label

    @staticmethod
    def _fromjson(data, labels):
        if "types" not in data:
            raise TypeError("Tuple Schema from JSON is missing argument 'types'")
        if not isinstance(data["types"], list):
            raise TypeError("argument 'types' for Tuple Schema from JSON should be a list, not {0}".format(repr(data["types"])))
        out = Tuple.__new__(Tuple)
        out._types = [Schema._fromjson(x, labels) for x in data["types"]]
        out.nullable = data.get("nullable", False)
        out.mask = data.get("mask", None)
        out.namespace = data.get("namespace", "")
        out.packing = Schema._packingfromjson(data.get("packing", None))
        out.name = data.get("name", None)
        out.doc = data.get("doc", None)
        out.metadata = oamap.util.json2python(data.get("metadata", None))
        if "label" in data:
            labels[data["label"]] = out
        return out

    def _finalizefromjson(self, labels):
        for i in range(len(self._types)):
            if isinstance(self._types[i], basestring):
                if self._types[i] not in labels:
                    raise TypeError("unresolved label: {0}".format(repr(self._types[i])))
                self._types[i] = labels[self._types[i]]
            else:
                self._types[i]._finalizefromjson(labels)

    def _collectlabels(self, collection, labels):
        if id(self) not in collection:
            collection.add(id(self))
            for item in self._types:
                item._collectlabels(collection, labels)
        else:
            labels.append(self)

    def copy(self, **replacements):
        if "types" not in replacements:
            replacements["types"] = self._types
        if "nullable" not in replacements:
            replacements["nullable"] = self._nullable
        if "mask" not in replacements:
            replacements["mask"] = self._mask
        if "namespace" not in replacements:
            replacements["namespace"] = self._namespace
        if "packing" not in replacements:
            replacements["packing"] = self._packing
        if "name" not in replacements:
            replacements["name"] = self._name
        if "doc" not in replacements:
            replacements["doc"] = self._doc
        if "metadata" not in replacements:
            replacements["metadata"] = self._metadata
        return Tuple(**replacements)

    def replace(self, fcn, *args, **kwds):
        return fcn(Tuple([x.replace(fcn, *args, **kwds) for x in self._types], nullable=self._nullable, mask=self._mask, namespace=self._namespace, packing=self._packingcopy(), name=self._name, doc=self._doc, metadata=copy.deepcopy(self._metadata)), *args, **kwds)

    def _path(self, loc, path, parents, memo):
        nodes = None
        for nodes in Schema._path(self, loc, path, parents, memo):
            yield nodes
        if nodes is None:
            for i, x in enumerate(self._types):
                for nodes in x._path(loc + (str(i),), path, (self,) + parents, memo):
                    yield nodes

    def _keep(self, loc, paths, project, memo):
        types = []
        for i, x in enumerate(self._types):
            n = str(i)
            if any(fnmatch.fnmatchcase("/".join(loc + (n,)), p) for p in paths):
                types.append(x)
            elif any(fnmatch.fnmatchcase("/".join(loc + (n,)), "/".join(p.split("/")[:len(loc) + 1])) for p in paths):
                f = x._keep(loc + (n,), paths, project, memo)
                if f is not None:
                    types.append(f)
        if len(types) == 0:
            return None
        elif project and len(fields) == 1:
            out, = fields.values()
            return out
        else:
            return self.copy(types=types)

    def _drop(self, loc, paths, memo):
        types = []
        for i, x in enumerate(self._types):
            n = str(i)
            if not any(fnmatch.fnmatchcase("/".join(loc + (n,)), p) for p in paths):
                f = x._drop(loc + (n,), paths, memo)
                if f is not None:
                    types.append(f)
        if len(types) == 0:
            return None
        else:
            return self.copy(types=types)

    def _contains(self, schema, memo):
        if self == schema:
            return True
        else:
            return any(x._contains(schema, memo) for x in self._types)

    def __hash__(self):
        return hash((Tuple, self._types, self._nullable, self._mask, self._namespace, self._packing, self._name, self._doc, oamap.util.python2hashable(self._metadata)))

    def __eq__(self, other, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)] == id(other)
        if not (isinstance(other, Tuple) and len(self._types) == len(other._types) and self._nullable == other._nullable and self._mask == other._mask and self._namespace == other._namespace and self._packing == other._packing and self._name == other._name and self._doc == other._doc and self._metadata == other._metadata):
            return False
        memo[id(self)] = id(other)
        return all(x.__eq__(y, memo) for x, y in zip(self._types, other._types))

    def __contains__(self, value, memo=None):
        if memo is None:
            memo = {}
        if value is None:
            return self.nullable
        if isinstance(value, tuple) and len(value) == len(self._types):
            return all(x.__contains__(v, memo) for v, x in zip(value, self._types))
        else:
            return False

    def _get_field(self, prefix, delimiter, i):
        return self._get_name(prefix, delimiter) + delimiter + "F" + repr(i)

    def _defaultnames(self, prefix, delimiter, memo, pointers, nesting):
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        if self._nullable:
            self._mask = self._get_mask(prefix, delimiter)
        for i, x in enumerate(self._types):
            x._defaultnames(self._get_field(prefix, delimiter, i), delimiter, memo, pointers, nesting.union(set([id(self)])))
        memo[id(self)] = prefix

    def _generator(self, prefix, delimiter, cacheidx, memo, nesting, extension):
        if len(self._types) == 0:
            raise TypeError("Tuple has no types")
        if id(self) in nesting:
            raise TypeError("types may not be defined in terms of themselves:\n\n    {0}".format(repr(self)))
        args = []

        if self._nullable:
            cls = oamap.generator.MaskedTupleGenerator
            args.append(self._get_mask(prefix, delimiter))
            args.append(cacheidx[0]); cacheidx[0] += 1
        else:
            cls = oamap.generator.TupleGenerator

        typesgen = [x._generator(self._get_field(prefix, delimiter, i), delimiter, cacheidx, memo, nesting.union(set([id(self)])), extension) for i, x in enumerate(self._types)]
        args.append(typesgen)
        args.append(self._namespace)
        args.append(self._packingcopy())
        args.append(self._name)
        args.append(prefix)
        args.append(self.copy(types=[x.schema for x in typesgen], packing=self._packingcopy()))

        for ext in extension:
            if ext.matches(self):
                args.insert(0, cls)
                cls = ext
                break

        memo[id(self)] = cls(*args)
        return memo[id(self)]

################################################################ Pointers redirect to the contents of other types

class Pointer(Schema):
    def __init__(self, target, nullable=False, positions=None, mask=None, namespace="", packing=None, name=None, doc=None, metadata=None):
        self.target = target
        self.nullable = nullable
        self.positions = positions
        self.mask = mask
        self.namespace = namespace
        self.packing = packing
        self.name = name
        self.doc = doc
        self.metadata = metadata

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        if isinstance(value, basestring):
            value = Primitive(value)
        if not (value is None or isinstance(value, Schema)):
            raise TypeError("target must be None or a Schema, not {0}".format(repr(value)))
        if value is self:
            raise TypeError("Pointer may not point directly at itself (it would never resolve to a value)")
        self._target = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        if not (value is None or isinstance(value, basestring)):
            raise TypeError("positions must be None or an array name (string), not {0}".format(repr(value)))
        self._positions = value

    def _hasarraynames(self, memo):
        if id(self) in memo:
            return True
        else:
            memo.add(id(self))
            return self._positions is not None and (not self._nullable or self._mask is not None) and self._target._hasarraynames(memo)

    def __repr__(self, labels=None, shown=None, indent=None):
        eq = "=" if indent is None else " = "

        if labels is None:
            labels = self._labels()
            shown = set()
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))

            args = []
            if indent is None:
                if self._target is None:
                    args.append(repr(None))
                else:
                    args.append(self._target.__repr__(labels, shown, indent))
            if self._nullable is not False:
                args.append("nullable" + eq + repr(self._nullable))
            if self._positions is not None:
                args.append("positions" + eq + repr(self._positions))
            if self._mask is not None:
                args.append("mask" + eq + repr(self._mask))
            if self._namespace != "":
                args.append("namespace" + eq + repr(self._namespace))
            if self._packing is not None:
                args.append("packing" + eq + repr(self._packing))
            if self._name is not None:
                args.append("name" + eq + repr(self._name))
            if self._doc is not None:
                args.append("doc" + eq + repr(self._doc))
            if self._metadata is not None:
                args.append("metadata" + eq + repr(self._metadata))

            if indent is None:
                argstr = ", ".join(args)
            else:
                if self._target is None:
                    args.append("target" + eq + repr(None) + "\n" + indent)
                else:
                    args.append("target" + eq + self._target.__repr__(labels, shown, indent + "  ").lstrip() + "\n" + indent)
                args[0] = "\n" + indent + "  " + args[0]
                argstr = ("," + "\n" + indent + "  ").join(args)
                
            if label is None:
                return "Pointer(" + argstr + ")"
            else:
                return label + ": Pointer(" + argstr + ")"

        else:
            return label

    def _tojson(self, explicit, labels, shown):
        label = self._label(labels)

        if label is None or id(self) not in shown:
            shown.add(id(self))
            if self._target is None:
                raise TypeError("pointer target is still None; must be resolved before it can be stored")
            out = OrderedDict([("type", "pointer"), ("target", self._target._tojson(explicit, labels, shown))])
            if explicit or self._nullable is not False:
                out["nullable"] = self._nullable
            if explicit or self._positions is not None:
                out["positions"] = self._positions
            if explicit or self._mask is not None:
                out["mask"] = self._mask
            if explicit or self._namespace != "":
                out["namespace"] = self._namespace
            if explicit or self._packing is not None:
                out["packing"] = self._packingtojson()
            if explicit or self._name is not None:
                out["name"] = self._name
            if explicit or self._doc is not None:
                out["doc"] = self._doc
            if explicit or self._metadata is not None:
                out["metadata"] = oamap.util.python2json(self._metadata)
            if explicit or label is not None:
                out["label"] = label
            return out
        else:
            return label

    @staticmethod
    def _fromjson(data, labels):
        if "target" not in data:
            raise TypeError("Pointer Schema from JSON is missing argument 'target'")
        out = Pointer.__new__(Pointer)
        out._target = Schema._fromjson(data["target"], labels)
        out.nullable = data.get("nullable", False)
        out.positions = data.get("positions", None)
        out.mask = data.get("mask", None)
        out.namespace = data.get("namespace", "")
        out.packing = Schema._packingfromjson(data.get("packing", None))
        out.name = data.get("name", None)
        out.doc = data.get("doc", None)
        out.metadata = oamap.util.json2python(data.get("metadata", None))
        if "label" in data:
            labels[data["label"]] = out
        return out

    def _finalizefromjson(self, labels):
        if isinstance(self._target, basestring):
            if self._target not in labels:
                raise TypeError("unresolved label: {0}".format(repr(self._target)))
            self._target = labels[self._target]
        else:
            self._target._finalizefromjson(labels)

    def _collectlabels(self, collection, labels):
        if id(self) not in collection:
            collection.add(id(self))
            if self._target is not None:
                self._target._collectlabels(collection, labels)
        else:
            labels.append(self)

    def copy(self, **replacements):
        if "target" not in replacements:
            replacements["target"] = self._target
        if "nullable" not in replacements:
            replacements["nullable"] = self._nullable
        if "positions" not in replacements:
            replacements["positions"] = self._positions
        if "mask" not in replacements:
            replacements["mask"] = self._mask
        if "namespace" not in replacements:
            replacements["namespace"] = self._namespace
        if "packing" not in replacements:
            replacements["packing"] = self._packing
        if "name" not in replacements:
            replacements["name"] = self._name
        if "doc" not in replacements:
            replacements["doc"] = self._doc
        if "metadata" not in replacements:
            replacements["metadata"] = self._metadata
        return Pointer(**replacements)

    def replace(self, fcn, *args, **kwds):
        return fcn(Pointer(self._target.replace(fcn, *args, **kwds), nullable=self._nullable, positions=self._positions, mask=self._mask, namespace=self._namespace, packing=self._packingcopy(), name=self._name, doc=self._doc, metadata=copy.deepcopy(self._metadata)), *args, **kwds)

    def _path(self, loc, path, parents, memo):
        nodes = None
        for nodes in Schema._path(self, loc, path, parents, memo):
            yield nodes
        if nodes is None:
            if id(self) not in memo:
                memo.add(id(self))
                for nodes in self._target._path(loc, path, (self,) + parents, memo):
                    yield nodes
        
    def _keep(self, loc, paths, project, memo):
        if id(self) in memo:
            return memo[id(self)]
        memo[id(self)] = self.copy(target=None)
        target = self._target._keep(loc, paths, project, memo)
        if target is None:
            return None
        else:
            memo[id(self)]._target = target
            return memo[id(self)]

    def _drop(self, loc, paths, memo):
        if id(self) in memo:
            return memo[id(self)]
        memo[id(self)] = self.copy(target=None)
        target = self._target._drop(loc, paths, memo)
        if target is None:
            return None
        else:
            memo[id(self)]._target = target
            return memo[id(self)]

    def _contains(self, schema, memo):
        if id(self) in memo:
            return False
        memo.add(id(self))
        if self == schema:
            return True
        else:
            return self._target._contains(schema, memo)

    def __hash__(self):
        return hash((Pointer, self._target, self._nullable, self._positions, self._mask, self._namespace, self._packing, self._name, self._doc, oamap.util.python2hashable(self._metadata)))

    def __eq__(self, other, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)] == id(other)
        if not (isinstance(other, Pointer) and self._nullable == other._nullable and self._positions == other._positions and self._mask == other._mask and self._namespace == other._namespace and self._packing == other._packing and self._name == other._name and self._doc == other._doc and self._metadata == other._metadata):
            return False
        memo[id(self)] = id(other)
        return self.target.__eq__(other.target, memo)

    def __contains__(self, value, memo=None):
        if memo is None:
            memo = {}
        if id(value) in memo:
            return memo[id(value)] == id(self)
        memo[id(value)] = id(self)
        if value is None:
            return self._nullable
        return self.target.__contains__(value, memo)

    def _get_positions(self, prefix, delimiter):
        if self._positions is None:
            return self._get_name(prefix, delimiter) + delimiter + "P"
        else:
            return self._positions

    def _get_external(self, prefix, delimiter):
        return self._get_name(prefix, delimiter) + delimiter + "X"

    def _defaultnames(self, prefix, delimiter, memo, pointers, nesting):
        if self._nullable:
            self._mask = self._get_mask(prefix, delimiter)
        # not done: see Schema.defaultnames
        memo[id(self)] = prefix
        pointers.append(self)
        
    def _generator(self, prefix, delimiter, cacheidx, memo, nesting, extension):
        if self._target is None:
            raise TypeError("when creating a Pointer type from a Pointer schema, target must be set to a value other than None")
        args = []

        if self._nullable:
            cls = oamap.generator.MaskedPointerGenerator
            args.append(self._get_mask(prefix, delimiter))
            args.append(cacheidx[0]); cacheidx[0] += 1
        else:
            cls = oamap.generator.PointerGenerator

        args.append(self._get_positions(prefix, delimiter))
        args.append(cacheidx[0]); cacheidx[0] += 1

        args.append((self._target, prefix, delimiter))  # placeholder! see _finalizegenerator!
        args.append(self._namespace)
        args.append(self._packingcopy())
        args.append(self._name)
        args.append(prefix)
        args.append(self.copy(packing=self._packingcopy()))

        for ext in extension:
            if ext.matches(self):
                args.insert(0, cls)
                cls = ext
                break

        memo[id(self)] = cls(*args)
        return memo[id(self)]
