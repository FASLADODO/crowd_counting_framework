task="adamw1_bigtail13i_t1_bike"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW default high lr"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/mybikedata  \
--lr 1e-3 \
--decay 0.1 \
--loss_fn "MSEL1Mean" \
--batch_size 5 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/adamw1_bigtail13i_t1_shb/adamw1_bigtail13i_t1_shb_checkpoint_valid_mae=-7.574910521507263.pth \
--datasetname my_bike_non_overlap \
--optim adamw \
--cache \
--epochs 2001 > logs/$task.log  &

echo logs/$task.log  # for convenience