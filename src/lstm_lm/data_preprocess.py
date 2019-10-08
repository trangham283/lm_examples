import os
import argparse
import cPickle as pickle

# change bucketing scheme for clean fisher
BUCKETS = [20, 40]
#BUCKETS = [20, 40, 80]

def read_vocab(data_dir):
    vocab_path = os.path.join(data_dir, 'fisher.vocab')
    idx2word = open(vocab_path).readlines()
    idx2word = [x.strip() for x in idx2word]
    word2idx = dict(zip(idx2word, range(len(idx2word))))
    return idx2word, word2idx

def tokenize(data_path, idx2word, word2idx):
    """Tokenizes a text file."""
    assert os.path.exists(data_path)

    all_data = [[], [], []]
    #all_data = [[], [], [], []]

    # Tokenize file content and divide into buckets
    with open(data_path, 'r') as f:
        for line in f:
            ids = []
            words = ['<sos>'] + line.split() + ['<eos>']
            for word in words:
                if word in word2idx:
                    ids.append(word2idx[word])
                else:
                    ids.append(word2idx['<unk>'])
            if len(ids) <= BUCKETS[0]:
                all_data[0].append(ids)
            elif len(ids) <= BUCKETS[1]:
                all_data[1].append(ids)
            #elif len(ids) <= BUCKETS[2]:
            #    all_data[2].append(ids)
            else:
                all_data[2].append(ids)
                #all_data[3].append(ids)
                
    #ids = torch.LongTensor(ids)
    return all_data

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            'Preprocess Fisher training data, binning by sentence lengths')
    pa.add_argument('--data_dir', type=str, \
            default='/g/ssli/projects/disfluencies/ttmt001/', \
            help='data directory, including vocab file')
    pa.add_argument('--dtype', type=str, \
            default='disf', \
            help='disf or clean')
    pa.add_argument('--split', type=str, \
            default='valid', \
            help='train, valid, or test split')

    args = pa.parse_args()
    data_dir = os.path.join(args.data_dir, 'fisher_' + args.dtype)
    split = args.split

    idx2word, word2idx = read_vocab(data_dir)
    data_path = os.path.join(data_dir, split + '_with_unk.txt')

    data = tokenize(data_path, idx2word, word2idx)
    out_file = os.path.join(data_dir, split + '.pickle')
    pickle.dump(data, open(out_file, 'w'))

