#!/bin/bash

ftype=$1
base_dir=/g/ssli/projects/disfluencies/ttmt001/fisher_$ftype
#train_txt=$base_dir/train.txt
train_txt=/s0/ttmt001/fisher/fisher_$ftype/train.txt

vocab_txt=$base_dir/fisher.vocab
order=3
model=fisher.trigram.model

# specifying -vocab and -unk makes sure unknown words are all mapped to  <unk>
# eos and sos are automatically inserted

# counts
ngram-count -unk -text $train_txt \
    -vocab $vocab_txt \
    -order $order > $base_dir/fisher.trigram.counts

# language model
ngram-count -unk -kndiscount \
    -read $base_dir/fisher.trigram.counts \
    -vocab $vocab_txt -order $order -lm - | ngram -lm - -order $order \
    -prune-lowprobs -write-lm $base_dir/$model

