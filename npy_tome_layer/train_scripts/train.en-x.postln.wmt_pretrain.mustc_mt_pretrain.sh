tgt=$1

export CUDA_VISIBLE_DEVICES=1,2,3,4

SAVE_DIR=/home/SuXiangDong/SxdStu98/student/cress/fairseq/checkpoints_mustc_de

data_dir=/data01/FL_Grp/HuSL_81/data/mustc/en-$tgt

exp=en-$tgt.postln.wmt_pretrain.mustc_mt_deltalm
fairseq-train $data_dir --text-data data/mustc/en-$tgt/binary/ --tgt-lang $tgt \
  --user-dir cress \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 2000000 --max-tokens-text 8192 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt \
  --ddp-backend=legacy_ddp \
  --warmup-updates 8000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --layernorm-embedding \
  --patience 10 \
  --fp16 \
  --ext-mt-training \
  --hubert-model-path hu_bert/hubert_base_ls960.pt
  
  
  #  \  --load-pretrained-mt-encoder-decoder-from checkpoints/en-$tgt.postln.wmt_pretrain/avg_last_5_epoch.pt