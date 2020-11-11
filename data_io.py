import numpy as np
import os
import gzip

def read_seq(seq_file):
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq)
            seq_list.append(seq_array) 
    
    return np.array(seq_list)

def get_RNA_seq_concolutional_array(seq, motif_len = 10):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'

    #motif: convolutional kernel
    half_len = motif_len/2
    row = (len(seq) + half_len *2 )
    new_array = np.zeros((row, 4))
    for i in range(half_len):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-half_len, row):
        new_array[i] = np.array([0.25]*4)
        
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue

        index = alpha.index(val)
        new_array[i][index] = 1

    return new_array

def load_data_file(inputfile, seq = True, onlytest = False):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    if len(path):
        path = './'
    data = dict()
    if seq: 
        tmp = []
        tmp.append(read_seq(inputfile))
        seq_onehot, structure = read_structure(inputfile, path)
        tmp.append(seq_onehot)
        data["seq"] = tmp
        data["structure"] = structure
    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = load_label_seq(inputfile)
        
    return data





if __name__ == "__main__":
    dataset = sys.argv[1]
    run_ideepv(dataset)
    