task="ccnn_v7_t1_shb"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "v3 that only max pooling after concat 3 branch, no norm"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--no_norm \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--epochs 701 > logs/$task.log  &

echo logs/$task.log  # for convenience