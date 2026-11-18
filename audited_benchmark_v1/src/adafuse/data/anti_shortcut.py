"""Explicit optional experimental prefix mask; distinct from header anonymization."""
import numpy as np

def extra_prefix_mask(stats,sequence,stat_prefix=0,sequence_prefix=0):
    stats=np.asarray(stats).copy();sequence=np.asarray(sequence).copy()
    if stats.ndim!=2 or sequence.ndim!=2 or len(stats)!=len(sequence):raise ValueError('Expected matching batches')
    if not 0<=stat_prefix<=stats.shape[1] or not 0<=sequence_prefix<=sequence.shape[1]:raise ValueError('Invalid prefix mask')
    stats[:,:stat_prefix]=0;sequence[:,:sequence_prefix]=0
    return stats,sequence
