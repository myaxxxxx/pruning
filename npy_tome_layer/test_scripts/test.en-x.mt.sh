
ckpt=/home/SuXiangDong/SxdStu98/student/cress/fairseq/checkpoints/avg_last_10_epoch.pt
lang=de
lenpen=1.2

exp=en-$tgt.postln.wmt_pretrain

export CUDA_VISIBLE_DEVICES=2

SAVE_DIR=/home/SuXiangDong/SxdStu98/student/cress/fairseq/checkpoints
fairseq-generate data/mustc/en-$lang --text-data data/mustc/en-$lang/binary --tgt-lang $lang \
  --user-dir cress \
  --config-yaml config.yaml --gen-subset test --task speech_and_text_translation \
  --path $ckpt \
  --ext-mt-training \
  --max-tokens 2000000 --max-tokens-text 4096 --beam 8 --lenpen $lenpen --scoring sacrebleu