
data_dir="../../data-bin/wmt16-de-en-original-skd-preprocessed"
average_checkpoint_path=../../checkpoints/wmt16-de-en-bi-skd-P0-Upp8/checkpoint_bi_avg.pt

NUMEXPR_MAX_THREADS=32 fairseq-generate  ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --batch-size 16 --seed 0 --skip-invalid-size-inputs-valid-test\
    --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1.1, \
        \"argmax_token_num\":5, \
        \"decode_alpha\": 1.0, \"decode_gamma\": 0, \
        \"decode_lm_path\": None, \
        \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
        \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 32,  \"decode_dedup\": True, \"decode_direction\": \"bidirection\" }" \
    --path ${average_checkpoint_path} > Generate.Beam.Bidir.1.0.out
NUMEXPR_MAX_THREADS=32 fairseq-generate  ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --batch-size 16 --seed 0 --skip-invalid-size-inputs-valid-test\
    --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1.1, \
        \"argmax_token_num\":5, \
        \"decode_alpha\": 1.1, \"decode_gamma\": 0, \
        \"decode_lm_path\": None, \
        \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
        \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 32,  \"decode_dedup\": True, \"decode_direction\": \"bidirection\" }" \
    --path ${average_checkpoint_path} > Generate.Beam.Bidir.1.1.out
NUMEXPR_MAX_THREADS=32 fairseq-generate  ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --batch-size 16 --seed 0 --skip-invalid-size-inputs-valid-test\
    --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1.1, \
        \"argmax_token_num\":5, \
        \"decode_alpha\": 1.2, \"decode_gamma\": 0, \
        \"decode_lm_path\": None, \
        \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
        \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 32,  \"decode_dedup\": True, \"decode_direction\": \"bidirection\" }" \
    --path ${average_checkpoint_path} > Generate.Beam.Bidir.1.2.out
NUMEXPR_MAX_THREADS=32 fairseq-generate  ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --batch-size 16 --seed 0 --skip-invalid-size-inputs-valid-test\
    --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1.1, \
        \"argmax_token_num\":5, \
        \"decode_alpha\": 1.3, \"decode_gamma\": 0, \
        \"decode_lm_path\": None, \
        \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
        \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 32,  \"decode_dedup\": True, \"decode_direction\": \"bidirection\" }" \
    --path ${average_checkpoint_path} > Generate.Beam.Bidir.1.3.out
NUMEXPR_MAX_THREADS=32 fairseq-generate  ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --batch-size 16 --seed 0 --skip-invalid-size-inputs-valid-test\
    --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1.1, \
        \"argmax_token_num\":5, \
        \"decode_alpha\": 1.4, \"decode_gamma\": 0, \
        \"decode_lm_path\": None, \
        \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
        \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 32,  \"decode_dedup\": True, \"decode_direction\": \"bidirection\" }" \
    --path ${average_checkpoint_path} > Generate.Beam.Bidir.1.4.out
