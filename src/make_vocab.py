#!/usr/bin/env python

import os
import sys
import argparse
import cPickle as pickle
from glob import glob

def make_vocab(in_file, threshold, out_file):
    c = {}
    f = open(in_file).readlines()
    print "Number of lines: ", len(f)
    for i, line in enumerate(f):
        words = line.strip().split()
        for w in words:
            if w not in c: c[w] = 0
            c[w] += 1
        if i % 10000 == 0:
            print "# Processed lines: ", i 
    fout = open(out_file, 'w')
    for k, v in c.iteritems():
        if v < threshold: continue
        print >> fout, k
    print >> fout, '<unk>'
    fout.close()
 
def prep_data(in_file, out_file, vocab_file):
    vocab = open(vocab_file).readlines()
    vocab = set([s.strip() for s in vocab])
    fout = open(out_file, 'w')
    f = open(in_file).readlines()
    for line in f:
        words = line.split()
        new_words = []
        for x in words:
            if x in vocab: new_words.append(x)
            else: new_words.append('<unk>')
        print >> fout, ' '.join(new_words)
    fout.close()

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Create vocabulary')
    pa.add_argument('--in_dir', \
            default='/g/ssli/projects/disfluencies/ttmt001', \
            help='project directory')
    pa.add_argument('--train_file', \
            default='train.txt', \
            help='file containing training data')
    pa.add_argument('--unk_split', \
            default='valid_splits', \
            help='split of files to convert to unk')
    pa.add_argument('--vocab_file', \
            default='fisher.vocab', \
            help='vocabulary file name')
    pa.add_argument('--step', \
            default='make_vocab', \
            help='make_vocab or prep_data')
    pa.add_argument('--dtype', \
            default='disf', \
            help='clean or disf')
    pa.add_argument('--threshold', \
            default=10, type=int, \
            help='vocabulary threshold before to convert to unk')

    args = pa.parse_args()
    unk_split = args.unk_split
    dtype = args.dtype
    vocab_file = os.path.join(args.in_dir, 'fisher_' + dtype, args.vocab_file)
    threshold = args.threshold
    if args.step == 'make_vocab':
        print "Creating vocabulary: "
        print vocab_file
        in_file = os.path.join(args.in_dir, 'fisher_' + dtype, args.train_file)
        print "from: ", in_file
        make_vocab(in_file, threshold, vocab_file)
    elif args.step == 'prep_data':
        data_dir = os.path.join(args.in_dir, 'fisher_' + dtype, unk_split)
        in_file = os.path.join(data_dir, args.train_file)
        out_file = in_file  + '_with_unk'
        prep_data(in_file, out_file, vocab_file)
    else:
        print "Need to specify option: make vocab or prep data"
        exit(0)
        
