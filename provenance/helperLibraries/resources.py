from visitor import walk, transform
import os.path
import fileutil
import urllib
#import urllib.parse
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

#import urllib.request
try:
    from urllib import request
except ImportError:
    from urllib2 import urlopen as urlrequest

import mimetypes
class Resource:
    def __init__(self, key, data, mimetype):
        self.key = key
        self._data = data
        self.mimetype = mimetype

    @staticmethod
    def from_file(key, path):
        mimetype = Resource.path_to_mimetype(path)
        data = fileutil.read_binary(path)
        return Resource(key, data, mimetype)

    @staticmethod
    def from_uri(key, path):
        mimetype = Resource.path_to_mimetype(path)
        data = path
        return ResourceIndirect(key, data, mimetype)

    @staticmethod
    def path_to_mimetype(path):
        mimetype, encoding = mimetypes.guess_type(path)
        if mimetype is None or mimetype == '':
                mimetype = 'application/octet-stream'
        return mimetype

    @property
    def extension(self):
        extension = mimetypes.guess_extension(self.mimetype)
        if extension is None:
            extension = ''
        return extension

    @property
    def data(self):
        return self._data


class ResourceIndirect(Resource):
    def __init__(self, key, target, mimetype):
        super().__init__(key, target, mimetype)

    @property
    def target(self):
        return super().data

    def open(self):
        return urllib.request.urlopen(self.target)

    @property
    def data(self):
        with self.open() as o:
            data = o.read()
            return data


def is_resource_ref(d):
    return isinstance(d, dict) and '$resource' in d


def is_resource_indirect_ref(d):
    return isinstance(d, dict) and '$indirect' in d


def find_resources(data):
    resources = {}

    def post(d):
        if isinstance(d, Resource):
            resources[d.key] = d

    walk(data, post=post)
    return resources


def find_resource_references(data):
    keys = []

    def post(d):
        if is_resource_ref(d):
            keys.append(d['$resource'])
        elif is_resource_indirect_ref(d):
            keys.append(d['$indirect'])

    walk(data, post=post)

    return keys


def replace_resource_references(data, resources):
    resource_instances = {}

    def post(d):
        if is_resource_ref(d):
            key = d['$resource']
            if key not in resource_instances:
                resource = resources[key]
                resource_instances[key] = Resource(key, resource['data'], resource['mimetype'])
            return resource_instances[key]
        elif is_resource_indirect_ref(d):
            key = d['$indirect']
            if key not in resource_instances:
                resource = resources[key]
                resource_instances[key] = ResourceIndirect(key, resource['data'].decode('utf-8'), resource['mimetype'])
            return resource_instances[key]
        else:
            return d

    return transform(data, post=post)


def extract_resources(data):
    resources = {}

    def post(d):
        if isinstance(d, Resource):
            assert d.key not in resources

            if isinstance(d, ResourceIndirect):
                resources[d.key] = {'data': d.target, 'mimetype': d.mimetype}
                return {'$indirect': d.key}
            else:
                resources[d.key] = {'data': d.data, 'mimetype': d.mimetype}
                return {'$resource': d.key}
        return d

    return transform(data, post=post), resources

