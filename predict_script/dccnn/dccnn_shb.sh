task="eval_bigtail13i_t2_shb"
# HTTPS_PROXY="http://10.60.28.99:86"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "eval dccnn adamw1_bigtail13i_t1_shb shb"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--eval_only  \
--batch_size 1 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/adamw1_bigtail13i_t1_shb/adamw1_bigtail13i_t1_shb_checkpoint_valid_mae=-7.574910521507263.pth \
--datasetname shanghaitech_non_overlap \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience