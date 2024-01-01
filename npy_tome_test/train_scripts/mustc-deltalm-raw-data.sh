
export CUDA_VISIBLE_DEVICES=1







tgt=de


data_dir=/workspace/chennan_tmp/s2t/mustc/en-de
TEXT_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin
CRESS_DIR=/workspace/chennan_tmp/s2t/cress_raw
HU_BERT=/workspace/chennan_tmp/s2t/mustc/en-de/hu_bert


exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_raw
fairseq-train $data_dir --text-data $TEXT_DIR --tgt-lang $tgt \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 3000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch deltalm_transformer_raw --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --patience 10 \
  --fp16 \
  --st-training --mt-finetune \
  --hubert-model-path $HU_BERT/hubert_base_ls960.pt \
  --pretrained_deltalm_checkpoint /workspace/deltalm/pretrain_model/deltalm-base.pt \
  --max-source-positions 512 --max-target-positions 512
  
  # --tensorboard checkpoints/${exp}



# exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_825
# ckpt=/workspace/chennan_tmp/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_825
python /workspace/chennan_tmp/s2t/fairseq/scripts/average_checkpoints.py \
    --inputs $ckpt \
    --num-epoch-checkpoints  \
    --output $ckpt/avg_last_10_epoch.pt



export CUDA_VISIBLE_DEVICES=0
ckpt=/workspace/chennan_tmp/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v3_raw/checkpoint10.pt

tgt=de


data_dir=/workspace/chennan_tmp/s2t/mustc/en-de
TEXT_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin
CRESS_DIR=/workspace/chennan_tmp/s2t/cress_raw
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

