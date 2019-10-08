#!/bin/bash

dtype=$1
modelname=lstm-lm-$2
ftype=$3
base_dir=/g/ssli/projects/disfluencies/ttmt001

# make table of scores:
python get_lstm_results.py --in_dir ${base_dir} \
 --out_dir ${base_dir}/fisher_${dtype} --ftype ${ftype} \
 --dtype $dtype --modelname $modelname

