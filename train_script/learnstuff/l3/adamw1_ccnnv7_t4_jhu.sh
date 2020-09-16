task="adamw1_CompactCNNV7_t4_jhu"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=6 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW with extrem high lr and decay, msel1mean on jhu, no more force eval cache"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/jhu_crowd_plusplus  \
--lr 1e-3 \
--decay 0.1 \
--loss_fn "MSEL1Mean" \
--batch_size 60 \
--datasetname jhucrowd_downsample_512 \
--optim adamw \
--skip_train_eval \
--epochs 401 > logs/$task.log  &

echo logs/$task.log  # for convenience