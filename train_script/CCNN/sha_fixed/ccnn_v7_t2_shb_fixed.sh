task="ccnn_v7_t2_sha_fixed"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "1e-4 shb fix 15"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--epochs 901 > logs/$task.log  &

echo logs/$task.log  # for convenience