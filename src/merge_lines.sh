#!/bin/bash

# convert one word per line to one sentence per line
SPLIT=$1

# "cleaned" is same as /g/ssli/projects/disfluencies/cleaned (from Vicky)
#DATA_DIR=/s0/ttmt001/fisher/cleaned/$SPLIT
#OUT_DIR=/s0/ttmt001/fisher/fisher_clean/$SPLIT
DATA_DIR=/s0/ttmt001/fisher/features_not_tokenized/${SPLIT}_words
OUT_DIR=/s0/ttmt001/fisher/fisher_dtok/$SPLIT
FILES=`ls ${DATA_DIR}/fsh*`

for F in $FILES
do
    BNAME=$(basename $F)
    NF=$OUT_DIR/$BNAME
    echo $NF
    cat $F | tr "_" " " | awk ' /^$/ { print; } /./ {printf("%s ", $0); } ' > $NF
done

