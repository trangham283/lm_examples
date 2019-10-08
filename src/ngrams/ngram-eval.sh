#!/bin/bash

dtype=$1 # clean or disf or dtok
ftype=$2 # ms or ptb 
base_dir=/g/ssli/projects/disfluencies/ttmt001
vocab_txt=${base_dir}/fisher_${dtype}/fisher.vocab
order=3
model=fisher.trigram.model

# evaluate
#valid_text=$base_dir/valid.txt
#eval_output_text=$base_dir/ppl_valid.txt 
#valid_text=$base_dir/debug/fsh_100045_A
#eval_output_text=$base_dir/debug/fsh_100045_A.ppl

## compute for swbd sentences
if [ $dtype == "dtok" ] 
then
    valid_text=$base_dir/swbd_${ftype}_sents_notok.txt
else
    valid_text=$base_dir/swbd_${ftype}_sents.txt
fi
echo $valid_text

eval_output_text=$base_dir/fisher_${dtype}/swbd_${ftype}_sents_ppl.txt
ngram -unk -debug 2 \
    -order $order \
    -lm ${base_dir}/fisher_${dtype}/${model} -ppl ${valid_text} \
    -vocab ${vocab_txt} > ${eval_output_text}

# make table of scores:
python get_ppl_results.py --in_dir ${base_dir}/fisher_$dtype \
 --out_dir ${base_dir}/fisher_${dtype} --ftype ${ftype}

