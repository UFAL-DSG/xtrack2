import math
import numpy as np
from collections import Counter

from utils import floatX, intX, shuffle, iter_data

def padded(seqs, pad_back=True, is_int=False, pad_by=None):
    if not pad_by:
        pad_by = [0]
    lens = map(len, seqs)
    max_len = max(lens)
    seqs_padded = []
    for seq, seq_len in zip(seqs, lens):
        n_pad = max_len - seq_len
        if pad_back:
            seq = seq + pad_by * n_pad
        else:
            seq = pad_by * n_pad + seq
        seqs_padded.append(seq)

    if is_int:
        return intX(seqs_padded)
    else:
        return floatX(seqs_padded)


class Padded(object):

    def __init__(self, seqs, targets, size=64, shuffle=True):
        self.seqs = seqs
        self.targets = targets
        self.size = size
        self.shuffle = shuffle

    def iter(self):
        
        if self.shuffle:
            self.seqs, self.targets = shuffle(self.seqs, self.targets)

        for i in range(0, len(self.seqs), self.size):
            xmb, ymb = self.seqs[i:i+self.size], self.targets[i:i+self.size]
            xmb = padded(xmb)
            ymb = floatX(ymb)
            yield xmb, ymb

class SortedPadded(object):
    def __init__(self, seqs, seq_ids, labels, size=64):
        self.seqs = seqs
        self.seq_ids = seq_ids
        self.labels = labels
        self.size = size

    def iter(self):
        for chunks in iter_data(*self.seqs, size=self.size*20):
            sort = np.argsort([len(x) for x in chunks[0]])

            for i in range(len(chunks)):
                chunks[i] = [chunks[i][idx] for idx in sort]

            # print range(len(x_chunk))[::self.size]
            mb_chunks = [[chunks[i][idx:idx+self.size] for i in range(len(chunks))] for idx in range(len(chunks[0]))[::self.size]]
            mb_chunks = shuffle(mb_chunks)
            for res in mb_chunks:
                res[0] = padded(res[0])

                yield res