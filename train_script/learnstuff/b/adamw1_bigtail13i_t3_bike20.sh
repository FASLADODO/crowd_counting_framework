task="adamw1_bigtail13i_t3_bike20"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=3 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW default lower lr"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/data_bike_20_q100  \
--lr 1e-4 \
--decay 0.01 \
--loss_fn "MSEMean" \
--batch_size 3 \
--datasetname my_bike_non_overlap \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/adamw1_bigtail13i_t1_shb/adamw1_bigtail13i_t1_shb_checkpoint_valid_mae=-7.574910521507263.pth \
--optim adamw \
--cache \
--epochs 2101 > logs/$task.log  &

echo logs/$task.log  # for convenience