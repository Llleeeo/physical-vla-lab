"""Offline recorded-observation prefix execution; does not simulate feedback."""
import numpy as np

def execute_prefix(chunks,lengths,prefix):
 chunks=np.asarray(chunks)
 if chunks.ndim!=3 or chunks.shape[-1]!=7 or not np.isfinite(chunks).all():raise ValueError('finite NxHx7 absolute chunks required')
 if not isinstance(prefix,int) or not 1<=prefix<=chunks.shape[1]:raise ValueError('invalid prefix')
 if any(n<1 for n in lengths) or sum(lengths)!=len(chunks):raise ValueError('episode lengths mismatch')
 stream=np.empty((len(chunks),7),dtype=chunks.dtype);anchors=np.empty(len(chunks),dtype=np.int64);horizons=anchors.copy();offset=0
 for n in lengths:
  for start in range(0,n,prefix):
   count=min(prefix,n-start);sl=slice(offset+start,offset+start+count)
   stream[sl]=chunks[offset+start,:count];anchors[sl]=offset+start;horizons[sl]=np.arange(count)
  offset+=n
 return stream,anchors,horizons
