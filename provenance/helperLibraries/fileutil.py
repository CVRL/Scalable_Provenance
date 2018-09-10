"""Provides helper functions for interacting with files."""


def read_binary(path):
    with open(path, 'rb') as f:
        return f.read()


def write_binary(path, data):
    with open(path, 'wb') as f:
        f.write(data)


def read_text(path):
    with open(path, 'r') as f:
        return f.read()


def write_text(path, text):
    with open(path, 'w') as f:
        f.write(text)


def read_lines(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def write_lines(path, lines):
    with open(path, 'w') as f:
        for i, line in enumerate(lines):
            if i > 0:
                f.write('\n')
            f.write(line)


def read_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, obj):
    import json
    with open(path, 'w') as f:
        json.dump(obj, f)


def read_csv(path):
    import csv
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def read_pickle(path, mode='rb'):
    import pickle
    with open(path, mode) as f:
        return pickle.load(f)


def write_pickle(data, path, mode='wb'):
    import pickle
    with open(path, mode) as f:
        pickle.dump(data, f)


def make_sure_path_exists(path):
    import errno
    import os
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def safe_remove_file(path):
    import logging
    import os
    import time
    while True:
        try:
            os.remove(path)
            break
        except OSError:
            logging.exception("Failed to delete {}".format(path))
            pass
        time.sleep(0.1)
        if not os.path.isfile(path):
            break


def safe_remove_directory(path):
    import logging
    import os
    import shutil
    import time
    while True:
        try:
            shutil.rmtree(path)
            break
        except OSError:
            logging.exception("Failed to delete {}".format(path))
            pass
        time.sleep(0.1)
        if not os.path.isdir(path):
            break
def getResourcePath(algname):
    import os
    rootname = 'provenance'
    ab = os.path.abspath('.')
    parts = ab.split(rootname)
    final = os.path.join(parts[0],rootname,algname,'algorithmResources')
    make_sure_path_exists(final)
    return final
rawFileNames=[".3fr",".ari",".arw",".bay",".crw",".cr2", ".cr3",".cap",".data",".dcs",".dcr",".dng",".drf",".eip",".erf",".fff",".gpr",".iiq",".k25",".kdc",".mdc",".mef",".mos",".mrw",".nef",".nrw",".obm",".orf",".pef",".ptx",".pxn",".r3d",".raf",".raw",".rwl",".rw2",".rwz",".sr2",".srf",".srw",".tif",".x3f"]
def isRaw(fileName):
    if fileName.endswith('.jpg') or fileName.endswith('.png') or fileName.endswith('.JPG') or fileName.endswith('.bmp') or fileName.endswith('.gif'):
        return False
    for ex in rawFileNames:
        if fileName.endswith(ex): return True
    return False
# deserialize the index serialized by indexConstruction
# indexFileBuffer is a binarary string (i.e. produced by file.read())
def splitMultiFileByteArray(byteData, sizeMetaDataLength, numFilesConcatinated):
    byteFile_Array = []
    startByteIndex = 0
    byteFile_ranges = []
    for i in range(numFilesConcatinated):
        byteFileDataLength = int(byteData[startByteIndex:startByteIndex + sizeMetaDataLength].decode('ascii'))
        byteEnd = startByteIndex + sizeMetaDataLength + byteFileDataLength
        # byteFileData = byteData[startByteIndex + sizeMetaDataLength:byteEnd]
        byteFile_ranges.append((startByteIndex+sizeMetaDataLength,byteEnd))
        # byteFile_Array.append(byteFileData)
        startByteIndex = byteEnd

    return byteFile_ranges
