task="adamw1_bigtail13i_t1_jhu"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW with extrem high lr and decay, msel1mean on jhu"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/jhu_crowd_plusplus  \
--lr 1e-3 \
--decay 0.1 \
--loss_fn "MSEL1Mean" \
--batch_size 40 \
--datasetname jhucrowd_256 \
--optim adamw \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience