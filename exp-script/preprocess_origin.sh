input_dir=""       # directory of raw text data
data_dir=$PREPROCESSED_DIR$   # directory of the generated binarized data
src=de                            # source language id
tgt=en                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.de-en --validpref ${input_dir}/valid.de-en --testpref ${input_dir}/test.de-en \
    --srcdict ${input_dir}/dict.${src}.txt --tgtdict ${input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 32
