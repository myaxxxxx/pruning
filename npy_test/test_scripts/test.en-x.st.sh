ckpt=/workspace/chennan_tmp/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt/avg_last_10_epoch.pt
CRESS_DIR=/workspace/chennan_tmp/s2t/cress
lang=de
lenpen=1.2


data_dir=/workspace/chennan_tmp/s2t/mustc/en-de
TEXT_DIR=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin
CRESS_DIR=/workspace/chennan_tmp/s2t/cress
HU_BERT=/workspace/chennan_tmp/s2t/mustc/en-de/hu_bert

export CUDA_VISIBLE_DEVICES=3
fairseq-generate  $data_dir \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
  --path $ckpt \
  --max-source-positions 900000 \
  --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu



# /home/SuXiangDong/SxdStu98/student/cress/fairseq/checkpoints/avg_last_10_epoch.pt