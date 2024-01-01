
export CUDA_VISIBLE_DEVICES=0



fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
  --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
  --save-dir ${MULTILINGUAL_BACKBONE} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_m --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${PRETRAINED_ASR}





SAVE_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/save_dir/multi

data_dir=/workspace/chennan_tmp/s2t/mustc/en-de
TEXT_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin
CRESS_DIR=/workspace/chennan_tmp/s2t/cress
HU_BERT=/workspace/chennan_tmp/s2t/mustc/en-de/hu_bert


exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_825
fairseq-train $data_dir --text-data $TEXT_DIR --tgt-lang $tgt \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 3000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch deltalm_transformer --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --patience 10 \
  --fp16 \
  --st-training --mt-finetune \
  --hubert-model-path $HU_BERT/hubert_base_ls960.pt \
  --pretrained_deltalm_checkpoint /workspace/deltalm/save_dir/delta_s2t_ted_en_de_right/checkpoints/checkpoint7.pt \
  --max-source-positions 512 --max-target-positions 512
  
  # --tensorboard checkpoints/${exp}



exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_825
ckpt=/workspace/chennan_tmp/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_825
python /workspace/chennan_tmp/s2t/fairseq/scripts/average_checkpoints.py \
    --inputs $ckpt \
    --num-epoch-checkpoints  \
    --output $ckpt/avg_last_10_epoch.pt




ckpt=/workspace/chennan_tmp/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_825/checkpoint4.pt

data_dir=/workspace/chennan_tmp/s2t/mustc/en-de
TEXT_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin
CRESS_DIR=/workspace/chennan_tmp/s2t/cress
HU_BERT=/workspace/chennan_tmp/s2t/mustc/en-de/hu_bert

lang=de
lenpen=1.2
export CUDA_VISIBLE_DEVICES=0
fairseq-generate  $data_dir \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
  --path $ckpt \
  --max-source-positions 900000 \
  --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu
# ckpt=/workspace/chennan_tmp/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_test/checkpoint1.pt
# CRESS_DIR=/workspace/chennan_tmp/s2t/cress
# lang=de
# lenpen=1.2


# data_dir=/workspace/chennan_tmp/s2t/mustc/en-de
# TEXT_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin
# CRESS_DIR=/workspace/chennan_tmp/s2t/cress
# HU_BERT=/workspace/chennan_tmp/s2t/mustc/en-de/hu_bert

# export CUDA_VISIBLE_DEVICES=3
# fairseq-generate  $data_dir \
#   --user-dir $CRESS_DIR \
#   --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
#   --path $ckpt \
#   --max-source-positions 900000 \
#   --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu

