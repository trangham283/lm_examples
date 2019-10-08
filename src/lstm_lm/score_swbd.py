# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import lm_model
import random
import cPickle as pickle
from data_preprocess import read_vocab, BUCKETS
import os
import cPickle as pickle

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data_path', type=str, \
        default='/g/ssli/projects/disfluencies/ttmt001',
        help='location of the data corpus')
parser.add_argument('--dtype', type=str, \
        default='disf',
        help='data type: disf or clean')
parser.add_argument('--model_type', type=str, default='LSTM',
        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
        help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=150,
        help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
        help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
        help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
        help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
        help='tie the word embedding and softmax weights')
parser.add_argument('--eval_path', type=str, \
        default='/g/ssli/projects/disfluencies/ttmt001', \
        help='location of the data to score (swbd)')
parser.add_argument('--ftype', type=str, \
        default='ms', \
        help='type of switchboard data to score (ms or ptb)')
parser.add_argument('--seed', type=int, default=1111, \
        help='random seed')
parser.add_argument('--eos', action='store_true', \
        help='keep score of eos token')
parser.add_argument('--cuda', action='store_true', \
        help='use CUDA')
parser.add_argument('--save', type=str,  \
        default='/g/ssli/projects/disfluencies/ttmt001/fisher/lstm-model.pt',\
        help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
idx2word, word2idx = read_vocab(os.path.join(args.data_path, \
        'fisher_' + args.dtype))
ntokens = len(word2idx)

eval_batch_size = 1
def read_data(eval_path, ftype):
    data_pairs = []
    if args.dtype == 'dtok':
        filename = os.path.join(eval_path, 'swbd_{}_sents_notok.txt'.format(ftype))
    else:
        filename = os.path.join(eval_path, 'swbd_{}_sents.txt'.format(ftype))
    swbd_data = open(filename).readlines()
    swbd_data = [x.strip() for x in swbd_data]
    for line in swbd_data:
        line_ids = []
        words = ['<sos>'] + line.split()
        for word in words:
            if word in word2idx:
                line_ids.append(word2idx[word])
            else:
                line_ids.append(word2idx['<unk>'])
        target_ids = line_ids[1:] + [word2idx['<eos>']]
        line_ids = torch.LongTensor(line_ids).to(device)
        target_ids = torch.LongTensor(target_ids).view(-1).to(device)
        data_pairs.append([line_ids, target_ids])
    return data_pairs

# Load the best saved model.
save = args.save
cuda = torch.cuda.is_available()

model = lm_model.RNNModel(args.model_type, ntokens, args.emsize, \
        args.nhid, args.nlayers, args.dropout, args.tied).to(device)

ckpt = save[:-2] + 'ckpt'
if cuda:
    checkpoint = torch.load(ckpt)
else:
    # Load GPU model on CPU
    checkpoint = torch.load(ckpt, \
            map_location=lambda storage, loc: storage)
start_epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (trained for {} epochs)".format(\
        ckpt, checkpoint['epoch']))
model.rnn.flatten_parameters()

#with open(save, 'rb') as f:
#    model = torch.load(f)
#    # after load the rnn params are not a continuous chunk of memory
#    # this makes them a continuous chunk, and will speed up forward pass
#    model.rnn.flatten_parameters()

eval_path = args.eval_path
ftype = args.ftype
data_pairs = read_data(eval_path, ftype)

model.eval()
scoring = nn.Softmax()
all_scores = []
with torch.no_grad():
    for data, targets in data_pairs:
        hidden = model.init_hidden(eval_batch_size)
        lengths = [len(data)]
        this_sent = []
        data = data.view(-1, len(data)) # reshape to batch_first
        output, hidden = model(data, hidden, lengths)
        output_flat = output.view(-1, ntokens)
        scores = softmax(output_flat, dim=1)
        for k, c in enumerate(targets):
            this_score = scores[k, c].item()
            this_sent.append(this_score)
        if not args.eos:
            all_scores.append(this_sent[:-1]) # skip score of <eos> token
        else:
            all_scores.append(this_sent) # keep score of <eos> token

# save all_scores
modelname = os.path.basename(save).split('.')[0]
outname = os.path.join(eval_path, 'swbd_{}_scores_{}.pickle'.format(ftype, \
        modelname))
pickle.dump(all_scores, open(outname, 'w'))


