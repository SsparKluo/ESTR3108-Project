import numpy as np
import sys
import os
from data_io import read_data_file

def padding_seq(seq, max_len = 501, default_char = 'N'):
    cur_len = len(seq)
    new_seq = ""
    if cur_len < max_len:
        new_seq = seq + default_char * (max_len - cur_len)
    else:
        new_seq = seq[:max_len]
    return new_seq

def split_seq(seq, overlap = 20, window_size = 101):
    seq_len = len(seq)
    splitted_seqs = []
    remain = ""

    if seq_len >= window_size:
        times = (seq_len - window_size) / (window_size - overlap) + 1
        remain_size = (seq_len - window_size) % (window_size - overlap)
        remain = seq[-remain_size:]
    else:
        times = 0
        remain = seq_len

    pointer = overlap #points the last index we had processed, initialied for the first run.
    
    for i in range(times):
        start = pointer - overlap
        pointer = start + window_size
        subseq = seq[start:pointer]
        splitted_seqs.append(subseq)

    remain_seq = padding_seq(remain, max_len = window_size)
    splitted_seqs.append(remain_seq)
    return splitted_seqs

def seq2array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    a = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        index = a.index(val)
        new_array[i][index] = 1

        #data[key] = new_array
    return new_array

def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        #bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_seq(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = seq2array(bag_seq)
        bag_subt = tri_fea

        bags.append(np.array(bag_subt))
    
        
    return bags, labels

def get_bag_data():
    pass

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = data["max_len"])

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label

if __name__ == "__main__":
    pass