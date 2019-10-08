# coding: utf-8
import argparse
import time, os, sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import lm_model
import random
import cPickle as pickle
from data_preprocess import read_vocab, BUCKETS

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
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
        help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
        help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
        help='random seed')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
        help='report interval')
parser.add_argument('--save', type=str,  \
        default='/g/ssli/projects/disfluencies/ttmt001/fisher/lstm-lm-0.pt',
        help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, you should run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data and divide into batches
###############################################################################
def batchify(data, bsz):
    batches = []
    for i in range(len(BUCKETS) + 1):
        this_bucket = data[i]
        if i == len(BUCKETS):
            max_len = max([len(x) for x in this_bucket])
        else:
            max_len = BUCKETS[i]
        for batch_offset in range(0, len(this_bucket), bsz):
            this_batch = this_bucket[batch_offset:batch_offset+bsz]
            this_batch = [torch.LongTensor(ids) for ids in this_batch]
            batches.append((max_len, this_batch))
    return batches

idx2word, word2idx = read_vocab(os.path.join(args.data_path, \
        'fisher_' + args.dtype))
ntokens = len(word2idx)

train_path = os.path.join(args.data_path, 'fisher_' + args.dtype, \
        'train.pickle')
valid_path = os.path.join(args.data_path, 'fisher_' + args.dtype, \
        'valid.pickle')
train_data = pickle.load(open(train_path))
valid_data = pickle.load(open(valid_path))
            
train_batches = batchify(train_data, args.batch_size)
valid_batches = batchify(valid_data, args.batch_size)

###############################################################################
# Build the model
###############################################################################
model = lm_model.RNNModel(args.model_type, ntokens, args.emsize, \
        args.nhid, args.nlayers, args.dropout, args.tied).to(device)

pad_idx = word2idx['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params, lr=args.lr)
cuda = torch.cuda.is_available()

###############################################################################
# Training code
###############################################################################

def get_batch(batch):
    current_batch_size = len(batch)
    # pad sequence, ensure correct dimensions
    # also need this to be in decreasing length order
    sorted_batch = sorted(batch, key=lambda x: len(x), reverse=True)
    padded_batch = pad_sequence(sorted_batch, batch_first=True, \
            padding_value=pad_idx)
    data = padded_batch[:, :-1]
    target = padded_batch[:, 1:]
    lengths = [len(x)-1 for x in sorted_batch]
    return data.to(device), target.to(device), current_batch_size, lengths

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, this_batch in enumerate(data_source):
            seq_len, batch = this_batch
            data, targets, current_batch_size, lengths = get_batch(batch)
            #print(i, data.shape)
            hidden = model.init_hidden(current_batch_size)
            output, hidden = model(data, hidden, lengths)
            output_flat = output.view(-1, ntokens)
            targets_flat = targets.contiguous().view(-1)           
            # accumulate loss per batch
            total_loss += criterion(output_flat, targets_flat).item()
    return total_loss / len(data_source) 

# Loop over epochs.
lr = args.lr
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    # shuffle at each epoch
    random.shuffle(train_batches)
    for i, this_batch in enumerate(train_batches):
        seq_len, batch = this_batch
        data, targets, current_batch_size, lengths = get_batch(batch)
        # My data is now [batch x seq_len x input_size], so it's ok to 
        # initialize hidden like this (instead of the repackage...)
        hidden = model.init_hidden(current_batch_size)
        model.zero_grad()
        optimizer.zero_grad()
        output, hidden = model(data, hidden, lengths)
        flat_outputs = output.view(-1, ntokens)
        flat_targets = targets.contiguous().view(-1)
        loss = criterion(flat_outputs, flat_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | '
            'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, i, \
                    len(train_batches), lr, elapsed * 1000 / args.log_interval,\
                    cur_loss, math.exp(cur_loss)))
            sys.stdout.flush()
            #with open(args.save, 'wb') as f:
            #    torch.save(model, f)
            total_loss = 0
            start_time = time.time()


#####################################
## debug
#print("debug eval")
#save = args.save
#with open(save, 'rb') as f:
#    model = torch.load(f)
#    model.rnn.flatten_parameters()

#val_loss = evaluate(valid_batches)
#print(val_loss)
#exit(0)
#####################################

# At any point you can hit Ctrl + C to break out of training early.
ckpt = args.save[:-2] + 'ckpt'
try:
    if os.path.isfile(ckpt):
        print("=> loading checkpoint '{}' ...".format(ckpt))
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
    else:
        start_epoch = 1
        best_val_loss = None

    for epoch in range(start_epoch, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(valid_batches)
        print('------------')
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, \
                        (time.time() - epoch_start_time), \
                        val_loss, math.exp(val_loss)))
        print('------------')
        sys.stdout.flush()
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            state = {'epoch': start_epoch + epoch - 1, \
                'state_dict': model.state_dict(), \
                'best_val_loss': best_val_loss}
            torch.save(state, ckpt)
        else:
            # Anneal the learning rate if no improvement has been seen 
            # in the validation dataset.
            print ("=> Validation loss did not improve")
            lr *= 0.9
except KeyboardInterrupt:
    print('--------------')
    print('Exiting from training early and saving current model')
    with open(args.save, 'wb') as f:
        torch.save(model, f)


