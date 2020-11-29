from os import read
import numpy as np
import os
import sys

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    max_len = 0
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                if len(seq) > max_len:
                    max_len = len(seq)
                seq_list.append(seq)
                labels.append([1, 0] if label == 1 else [0, 1])
    
    return seq_list, labels, max_len

def read_data_file(posi_file, nega_file = None):
    data = dict()
    max_len0 = max_len1 = 0
    seqs, labels, max_len0 = read_seq_graphprot(posi_file, label = 1)
    if nega_file:
        seqs2, labels2, max_len1 = read_seq_graphprot(nega_file, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2

    max_len = max_len0 if max_len0 > max_len1 else max_len1    
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    data["max_len"] = max_len
    
    return data

def main():
    posi = input('positive test file path: ')
    nega = input('negative test file path: ')
    data = read_data_file(posi, nega_file= nega)
    print(data["seq"])
    print(data["Y"])



if __name__ == "__main__":
    main()
    