task="adamw1_ccnnv7_t7_jhu.sh"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=6 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "lower lr and decay"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/jhu_crowd_plusplus  \
--lr 1e-4 \
--decay 0.001 \
--loss_fn "MSEL1Mean" \
--batch_size 60 \
--datasetname jhucrowd_downsample_512 \
--optim adamw \
--skip_train_eval \
--epochs 201 > logs/$task.log  &

echo logs/$task.log  # for convenience