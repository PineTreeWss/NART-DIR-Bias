input_dir="../../data-bin/wmt14-de-en-origin-big-s-kd/"       # directory of raw text data
data_dir=../../data-bin/wmt16-de-en-original-skd-preprocessed   # directory of the generated binarized data
src=de                            # source language id
tgt=en                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.de-en --validpref ${input_dir}/valid.de-en --testpref ${input_dir}/test.de-en \
    --srcdict ${input_dir}/dict.${src}.txt --tgtdict ${input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 32