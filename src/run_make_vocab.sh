#!/bin/bash

split=$1
dtype=$2
DATA_DIR=/g/ssli/projects/disfluencies/ttmt001
SENT_FILE=${DATA_DIR}/${split}_list

#FILES=`ls ${DATA_DIR}`
#for F in $FILES 

while IFS= read -r line
do
    echo $line
    python make_vocab.py --train_file $line --dtype ${dtype} \
        --unk_split ${split}_splits --step prep_data
done < ${SENT_FILE}

