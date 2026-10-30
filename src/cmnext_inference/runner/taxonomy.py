"""Map network labels to the user-defined factory inspection taxonomy."""
import numpy as np

def build_lookup(mapping):
    count=mapping['target_class_count']
    unknown=mapping['unknown_target_id']
    if not 0<=unknown<count:raise ValueError('Invalid unknown target ID')
    lookup=np.full(256,unknown,dtype=np.uint8)
    source=mapping['source_to_target']
    if len(source)!=mapping['source_class_count']:raise ValueError('Source mapping length mismatch')
    for index,value in enumerate(source):
        if value is None:continue
        if not isinstance(value,int) or not 0<=value<count:raise ValueError('Invalid target class ID')
        lookup[index]=value
    return lookup

def map_labels(labels,lookup):
    if labels.dtype!=np.uint8 or labels.ndim!=2:raise ValueError('Expected mono8 source label image')
    return lookup[labels]
