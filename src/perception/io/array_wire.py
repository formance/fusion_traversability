"""Length-prefixed JSON metadata followed by typed contiguous binary arrays."""
import json
import struct
import numpy as np

MAX_BODY = 64 * 1024 * 1024

def pack_arrays(metadata, arrays):
    header = dict(metadata)
    descriptors = {}
    pieces = []
    offset = 0
    for name, array in arrays.items():
        array = np.ascontiguousarray(array)
        if array.dtype not in (np.dtype('uint8'), np.dtype('<f4')):
            raise ValueError('Unsupported wire dtype')
        raw = array.tobytes()
        descriptors[name] = dict(shape=list(array.shape), dtype=array.dtype.str, offset=offset, size=len(raw))
        pieces.append(raw)
        offset += len(raw)
    header['arrays'] = descriptors
    encoded = json.dumps(header, allow_nan=False).encode()
    body = struct.pack('<I', len(encoded)) + encoded + b''.join(pieces)
    if len(body) > MAX_BODY:
        raise ValueError('Binary message too large')
    return body

def unpack_arrays(body):
    if not 4 <= len(body) <= MAX_BODY:
        raise ValueError('Invalid binary message size')
    length = struct.unpack_from('<I', body)[0]
    if not 0 < length <= 128 * 1024 or 4 + length > len(body):
        raise ValueError('Invalid metadata size')
    header = json.loads(body[4:4+length])
    payload = memoryview(body)[4+length:]
    arrays = {}
    for name, desc in header.pop('arrays').items():
        dtype = np.dtype(desc['dtype'])
        shape = desc['shape']
        if dtype not in (np.dtype('uint8'), np.dtype('<f4')) or not 1 <= len(shape) <= 3 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid array type or shape')
        size = int(np.prod(shape, dtype=object)) * dtype.itemsize
        offset = desc['offset']
        if type(offset) is not int or offset < 0 or size != desc['size'] or offset + size > len(payload):
            raise ValueError('Invalid array bounds')
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=size//dtype.itemsize, offset=offset).reshape(shape)
    return header, arrays
