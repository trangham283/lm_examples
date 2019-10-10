# lm_examples
Language model examples -- tutorial code for new grad students at TIAL lab.

# Dependencies
* SRI LM: /g/tial/sw/pkgs/srilm-1.7.1/bin/i686-m64/
* pytorch 0.4.0, cuda (or just use the CPU version)
* Python 2.7.x

# Fisher data set
* Original: /g/ssli/projects/disfluencies/fisher
* Various versions split to train and validation: `/g/ssli/projects/disfluencies/ttmt001/fisher_{clean,disf,dtok}`
    * disf: original version with disfluencies
    * clean: version cleaned of disfluencies
    * dtok: version with disfluencies, but not tokenized (e.g. "don't" is not split as "do" + "n't")
    * fsh_1* is in valid, the rest in train
* Stats: 
    * train: from 15522 files -- wc >> 1,181,752 sents; 14,363,366 tokens
    * valid: from 3137 files -- wc >> 268,991 sents; 2,841,422 tokens
Recommendation: use the dtok version.

# Steps
Some of these are already done (this is for documentation purposes only)

**1. Split to train/valid** (should be done already -- skip):

Raw data: 
```
mv fisher/text/fsh_1* fisher_disf/valid/
mv fisher/text/* fisher_disf/train/
```

**2. Preprocessings** (also should be done already -- skip):
    
   2a0. (if files have associcated features -- dtok set):
    `./src/grep_words.sh {train,valid}`

   2a1. (clean and dtok set): merge words into sentences; this takes individual files from `fisher/cleaned/{train,valid}` and puts them in `fisher/fisher_clean/{train,valid}`:
   
    ./src/merge_lines.sh {train,valid}

   2b. make big text file to be used in ngram models:
   
    
    cat train/* > train.txt
    cat valid/* > valid.txt
    

   2c. change "s" to "'s" in train.txt and valid.txt (clean and disf versions, not in dtok version):
    
    
    %s/\<s\>/'s/g
    %s/\<ll\>/'ll/g
    %s/\<m\>/'m/g
    %s/\<ve\>/'ve/g
    %s/\<d\>/'d/g
    %s/\<re\>/'re/g
    
    
   At this point only train.txt and valid.txt have this fix.

____________________________________________________
The corresponding data are in `/g/ssli/projects/disfluencies/ttmt001/fisher_{disf,clean,dtok}`. 

The outputs of the next steps are also in that directory, but I recommend studying and running the following steps 
for your own understanding and practice.


**3. Make vocabulary from training files** (specific to ngrams):

`python ngrams/make_vocab.py --step make_vocab --dtype {disf,clean,dtok}`

**4. Split to smaller chunks:** 
Need this for LSTM model only -- split train.txt and valid.txt into smaller chunks to facilitate parallelization (do this in the directory of your data):

```
split -d -n 40 train.txt
split -d -n 10 valid.txt
```

**5. Train ngrams:**

`./src/ngrams/ngram-lms.sh {disf,clean,dtok}`

**6. Prepare switchboard** (or other dataset) sentences to compute ppl score (should also be done already -- skip):

`python src/prep_lm_sentences.py`

This produces swbd_sents.tsv with turn, sent_num etc. info and ptb as well as ms versions of the sentences. 
* ptb = Penn Treebank version of transcripts
* ms = Mississippi State version of transcripts

For your purposes, you don't need to worry about the differences. Note, though that ms tokens don't split contractions while ptb ones do (e.g. "it's" in ms, "it 's" in ptb). So if you've been using the dtok version, it's better to choose ms; if you've been using disf or clean, use ptb.

For ngram score computations -- produce text files one sentence per line:

 ```
 cut -f5 swbd_sents.tsv > swbd_ms_sents.txt
 cut -f6 swbd_sents.tsv > swbd_ptb_sents.txt
 ```

 OR

 ```
 cut -f5 swbd_sents_with_ann_notok.tsv > swbd_ms_sents_notok.txt
 cut -f6 swbd_sents_with_ann_notok.tsv > swbd_ptb_sents_notok.txt
 ```
 
 Then remove header line.

**7. Compute ngram scores:**

`./src/ngrams/ngram-eval.sh {disf,clean,dtok} {ms,ptb}`

**8. Convert OOV tokens** to `<unk>` -- preparation step for LSTM LM models:
```
python src/make_vocab.py \
   --train_file {x01,..} \
   --dtype {disf,clean,dtok} \
   --unk_split {train,valid}_splits \
   --step prep_data
```

Batching this:

`src/run_make_vocab.sh {valid,train} {clean,disf,dtok}`

Then 

`cat x*_with_unk files > {train,valid}_with_unk.txt`

**9. Prepare bucketed data** for training LSTM LMs:

NOTE: need to add special words to both clean and disf fisher vocabs: 

`<eos>, <sos>, <pad>`

`python src/lstm_lm/data_preprocess.py --dtype {disf,clean,dtok} --split {valid,train}`

**10. Train LSTM language model** on fisher and score on SWBD:

`./src/lstm_lm/job{5000,5001,5002}.sh`

**11. Make table of scores** (optional):

```
lstm_lm/run_eval_lstm.sh disf 5000 {ms,ptb}
lstm_lm/run_eval_lstm.sh clean 5001 {ms,ptb}
lstm_lm/run_eval_lstm.sh dtok 5002 {ms,ptb}
```


