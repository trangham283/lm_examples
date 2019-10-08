#!/usr/bin/env python

from __future__ import division
import os
import re
import sys
import argparse
import cPickle as pickle
import pandas as pd
import numpy as np
from glob import glob
from itertools import groupby
from collections import Counter

# MS-State alignment columns
ERR = ['<SUB>', '<DEL>', '<INS>']
OTHER = ["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]
SLASH = ["//", "--", "-/", "/"]

def norm_laughter(word):
    if '[laughter-' in word:
        word = word.lstrip('[laughter').rstrip(']').lstrip('-')
    return word

def norm(word):
    word = norm_laughter(word)
    if word.startswith('a_') or word.startswith('b_'):
        word = word[2:]
    return word

def make_str(tokens):
    if not isinstance(tokens, basestring):
        print tokens, "an empty sent"
        return '<EMPTY>'
    else:
        all_str = tokens.strip().lstrip("['").rstrip("']")
        all_str = all_str.split()
        all_str = [x.rstrip("',").lstrip("'").rstrip('"').lstrip('"') \
                for x in all_str]
        all_str = [x for x in all_str if x not in SLASH]
        all_str = [norm(x) for x in all_str]
        if not all_str: all_str = ['<EMPTY>']
        return ' '.join(all_str)

def get_tokens(sentence):
    if not isinstance(sentence, str):
        toks = ['EMPTY_TREE']
    else:
        toks = sentence.strip().split()
    return toks

def norm_mrg(mrg):
    if not isinstance(mrg, str):
        return 'EMPTY_TREE'
    else:
        return mrg

def make_list(tokens):
    if not isinstance(tokens, basestring):
        return tokens
    else:
        all_str = tokens.strip().lstrip("['").rstrip("']")
        all_str = all_str.split()
        all_str = [x.rstrip("',").lstrip("'").rstrip('"').lstrip('"') \
                for x in all_str]
        return all_str

# file with "gold" sentence boundaries (slash units)
def preprocess_comb(project_dir, comb_suffix):
    fcomb = os.path.join(project_dir, comb_suffix)
    df_comb = pd.read_csv(fcomb, sep='\t')
    df_comb['file_num'] = df_comb['file'].apply(lambda x: \
            int(x.rstrip('.trans').lstrip('sw')))
    # THESE are files without turn id
    probs = [4103, 4108, 4171, 4329, 4617]
    for i, row in df_comb.iterrows():
        if row.file_num not in probs: continue
        # use ms names as first priority
        turn_toks = make_list(row.ms_names)
        turn_toks = [x for x in turn_toks if x != "None"]
        if not turn_toks: 
            # use ptb names instead
            turn_toks = make_list(row.names)
            turn_toks = [x for x in turn_toks if x != "None"]
        if not turn_toks: # completely empty case
            turn_num = int(df_comb.loc[i-1, 'turn']) + 1
        else:
            turn_num = int(turn_toks[0].split('_')[0][3:])
        if row.turn != turn_num:
            df_comb.loc[i, 'turn'] = turn_num
    df_comb['turn'] = df_comb.turn.apply(int)
    df_comb = df_comb.rename(columns={'sentence': 'ptb_tokens', \
            'ms_sentence': 'ms_tokens', \
            'names': 'ptb_tok_id', 'ms_names': 'ms_tok_id'})
    df_comb['tree_sent'] = df_comb.ptb_tokens.apply(make_str)
    df_comb['ms_sent'] = df_comb.ms_tokens.apply(make_str)
    cols = ['file_num', 'speaker', 'turn', 'sent_num', \
            'ms_sent', 'tree_sent', 'ms_tokens', 'ptb_tokens', \
            'ms_tok_id', 'ptb_tok_id', 'comb_ann', 'tags', \
            'ms_disfl', 'first_name', 'comb_sentence']
    df_comb.to_csv('/s0/ttmt001/swbd_sents_with_ann_notok.tsv', sep='\t', \
            index=False, columns=cols, header=True)

# df_comb columns: 
# ['speaker', 'turn', 'sent_num', 'file', 'sentence', 'ms_sentence',
# 'comb_sentence', 'names', 'ms_names', 'comb_ann', 'tags',
# 'first_name', 'ms_disfl', 'file_num']
def sort_keys(pw_names):
    pw_temp = []
    for pw in pw_names:
        turn = int(pw.split('_')[0][3:])
        sent_num = int(pw.split('_')[-1][2:])
        pw_temp.append([pw, (turn, sent_num)])
    sorted_keys = sorted(pw_temp, key=lambda x: x[1])
    sorted_keys = [x[0] for x in sorted_keys]
    return sorted_keys

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            'Get sentence-level examples to parse')
    pa.add_argument('--project_dir', \
            default='/g/ssli/projects/disfluencies/switchboard_data', \
            #default='/s0/ttmt001/speech_parsing', \
            help='project directory')
    pa.add_argument('--comb_suffix', \
            #default='treebank_msstate_combine_turns_uw_names_tt.tsv', \
            #default='treebank_msstate_combine_turns_uw_names_v2_with_ms_disfl.tsv', \
            default='treebank_msstate_combine_turns_uw_names_with_ms_disfl_no_tokenization.tsv', \
            help='sentence boundary combination file suffix')

    args = pa.parse_args()

    project_dir = args.project_dir
    comb_suffix = args.comb_suffix
    preprocess_comb(project_dir, comb_suffix)


