#!/bin/bash

SPLIT=$1

DATA_DIR=/s0/ttmt001/fisher/features_not_tokenized/$SPLIT
OUT_DIR=/s0/ttmt001/fisher/features_not_tokenized/${SPLIT}_words
FILES=`ls ${DATA_DIR}/fsh*`

for F in $FILES
do
    BNAME=$(basename $F)
    NF=$OUT_DIR/$BNAME
    echo $NF
    awk '{print $1}' $F > $NF
done

