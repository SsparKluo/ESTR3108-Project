from os import read
import numpy as np
import os
import sys

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def read_data_file(posi_file, nega_file = None):
    data = dict()
    seqs, labels = read_seq_graphprot(posi_file, label = 1)
    if nega_file:
        seqs2, labels2 = read_seq_graphprot(nega_file, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def main():
    posi = input('positive test file path: ')
    nega = input('negative test file path: ')
    data = read_data_file(posi, nega_file= nega)
    print(data["seq"])
    print(data["Y"])



if __name__ == "__main__":
    main()
    