task="ccnn_v7_t10_shb"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=3 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "try new argmentation and train test split "  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 3 \
--loss_fn "MSEMean" \
--datasetname shanghaitech_non_overlap \
--epochs 1001 > logs/$task.log  &

echo logs/$task.log  # for convenience