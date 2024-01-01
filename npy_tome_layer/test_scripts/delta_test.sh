DATA_BIN=/workspace/chennan_tmp/s2t/mustc/en-de/binary
SAVE_DIR=/workspace/deltalm/save_dir/delta_s2t_ted_en_de_right_test
USER_DIR=/workspace/deltalm/deltalm/code/unilm/deltalm/prefix_deltalm_old
PRETRAINED_MODEL=/workspace/deltalm/pretrain_model/deltalm-base.pt



export CUDA_VISIBLE_DEVICES=3
PRETRAINED_MODEL=/workspace/deltalm/pretrain_model/deltalm-base.pt
DATA_BIN=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin


# python train.py $DATA_BIN \
#     --arch deltalm_base \
#     --user-dir $USER_DIR \
#     --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
#     --share-all-embeddings \
#     --max-source-positions 512 --max-target-positions 512 \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --lr-scheduler inverse_sqrt \
#     --lr 8e-5 \
#     --warmup-init-lr 1e-07 \
#     --stop-min-lr 1e-09 \
#     --warmup-updates 6000 \
#     --max-update 400000 \
#     --max-epoch 50 \
#     --max-tokens 8192 \
#     --update-freq 1 \
#     --seed 1 \
#     --tensorboard-logdir $SAVE_DIR/tensorboard \
#     --save-dir $SAVE_DIR/checkpoints \
#     --keep-last-epochs 50 --fp16



#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-best-checkpoints 5 --save-interval-updates 1000
#     --keep-interval-updates 5





export CUDA_VISIBLE_DEVICES=2
DATA_BIN=/workspace/chennan_tmp/s2t/mustc/en-de/delta_data_bin


for((i=1;i<=40;i++));do  
SAVE_DIR=/workspace/deltalm/save_dir/delta_s2t_ted_en_de_right/checkpoints/checkpoint$i.pt
nohup python generate.py $DATA_BIN \
    --user-dir $USER_DIR \
    --path $SAVE_DIR \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece > /workspace/deltalm/save_dir/delta_s2t_ted_en_de_right/result$i.txt

done 