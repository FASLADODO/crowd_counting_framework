task="ccnn_v7_t5_shb"

CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "v3 that only max pooling after concat 3 branch, with norm"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 2e-5 \
--decay 1e-4 \
--batch_size 20 \
--datasetname shanghaitech_more_rnd \
--epochs 1001 > logs/$task.log  &

echo logs/$task.log  # for convenience