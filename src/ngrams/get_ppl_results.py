#!/usr/bin/env python

import os
import sys
from collections import Counter
import argparse
import pandas as pd

def parse_results(sent, eos=True):
    all_items = sent.split('\n')
    sent_str = all_items[0]
    toks = sent_str.split()
    if not eos:
        # don't include EOS
        toks = sent_str.split()
        scores = [x for x in all_items if x.startswith('\t')][:-1]
    else:
        # include EOS
        toks = sent_str.split() + ['</s>']
        scores = [x for x in all_items if x.startswith('\t')]
    scores = [float(x.split(']')[1].split('[')[0].strip()) for x in scores]
    assert len(scores) == len(toks)
    return sent_str, scores

def get_lm_scores(in_file, out_file, df_idx, ftype):
    list_row = []
    a = open(in_file).read()
    a = a.split('\n\n')[:-1] # each sentence get stored in these chunks
    assert len(a) == len(df_idx)
    for i, sent in enumerate(a):
        sent_str, scores = parse_results(sent)
        row = df_idx.iloc[i]
        col_name = 'ms_sent' if ftype == 'ms' else 'tree_sent'
        assert row[col_name] == sent_str
        list_row.append({'file_num': row.file_num, \
                'speaker': row.speaker, \
                'turn': row.turn, \
                'sent_num': row.sent_num, \
                'scores': scores})
    df = pd.DataFrame(list_row)
    cols = ['file_num', 'speaker', 'turn', 'sent_num', 'scores']
    df.to_csv(out_file, sep='\t', index=False, columns=cols, header=True)

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Compute ngram prob scores')
    pa.add_argument('--common_dir', \
            default='/g/ssli/projects/disfluencies/ttmt001', \
            help='project directory')
    pa.add_argument('--in_dir', \
            default='/g/ssli/projects/disfluencies/ttmt001', \
            help='project directory')
    pa.add_argument('--out_dir', \
            default='/g/ssli/projects/disfluencies/ttmt001', \
            help='output directory')
    pa.add_argument('--ftype', \
            default='ms', type=str, \
            help='ms or ptb sentences')

    args = pa.parse_args()
    # swbd_ptb_sents_ppl.txt
    basename = "swbd_{}_sents_ppl.txt".format(args.ftype)
    in_file = os.path.join(args.in_dir, basename)
    outname = "swbd_{}_ngram_scores.tsv".format(args.ftype)
    out_file = os.path.join(args.out_dir, outname)

    # file with sentence indices
    if "dtok" in in_file: 
        idx_file = os.path.join(args.common_dir, 'swbd_sents_with_ann_notok.tsv')
    else:
        idx_file = os.path.join(args.common_dir, 'swbd_sents.tsv')
    print "idx common file:", idx_file
    df_idx = pd.read_csv(idx_file, sep='\t')

    get_lm_scores(in_file, out_file, df_idx, args.ftype)

