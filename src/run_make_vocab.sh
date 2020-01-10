#!/bin/bash

split=$1 # train,valid,test
dtype=$2 # clean, disf, dtok
DATA_DIR=/g/ssli/projects/disfluencies/ttmt001
SENT_FILE=${DATA_DIR}/${split}_list 
# list of files in each split, you can get this by `ls ${split} >> sent_file` or something like that

#FILES=`ls ${DATA_DIR}`
#for F in $FILES 

while IFS= read -r line
do
    echo $line
    python make_vocab.py --train_file $line --dtype ${dtype} \
        --unk_split ${split}_splits --step prep_data
done < ${SENT_FILE}

