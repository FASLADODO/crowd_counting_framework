task="ccnn_v7_t11_shb_fixed"

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "mse mean"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 20 \
--loss_fn MSEmean \
--datasetname shanghaitech_rnd \
--epochs 901 > logs/$task.log  &

echo logs/$task.log  # for convenience