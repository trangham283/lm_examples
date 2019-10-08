#!/bin/bash

# GPU version
source /homes/ttmt001/transitory/pytorch-exps/pytorch-0.4.0-gpu/bin/activate
# CPU version
#source /homes/ttmt001/transitory/pytorch-exps/pytorch-0.4.0-cpu/bin/activate

RUN_ID=5002
SEED=3000
DTYPE=dtok
emsize=256
nhid=128

basedir=/homes/ttmt001/transitory/prosodic-anomalies/lstm_lm
cd $basedir
# Training
python train_lm.py \
    --epochs 50 --dtype $DTYPE \
    --emsize $emsize --nhid $nhid \
    --batch_size 128  --seed $SEED \
    --save /g/ssli/projects/disfluencies/ttmt001/fisher_$DTYPE/lstm-lm-${RUN_ID}.pt \
    --cuda >> $baseidr/logs/job${RUN_ID}.log

# score on switchboard data
for ftype in ms ptb
do
    python score_swbd.py \
        --dtype $DTYPE \
        --emsize $emsize \
        --nhid $nhid \
        --seed $SEED \
        --save /g/ssli/projects/disfluencies/ttmt001/fisher_$DTYPE/lstm-lm-${RUN_ID}.pt \
        --ftype $ftype --cuda --eos >> $basedir/logs/job${RUN_ID}.log
done
