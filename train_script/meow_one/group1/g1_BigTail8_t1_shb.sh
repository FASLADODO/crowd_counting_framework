task="g1_BigTail8_t1_shb"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -4 lr and decay, try downsample input for faster training"  \
--model "BigTail8" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 4 \
--loss_fn "MSEL1Sum" \
--datasetname shanghaitech_non_overlap_downsample \
--skip_train_eval \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience