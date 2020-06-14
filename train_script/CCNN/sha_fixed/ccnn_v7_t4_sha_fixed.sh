task="ccnn_v7_t4_sha_fixed"

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "l1 loss"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn L1 \
--datasetname shanghaitech_20p \
--epochs 1001 > logs/$task.log  &

echo logs/$task.log  # for convenience