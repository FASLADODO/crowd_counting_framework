task="eval_adamw1_bigtail13i_t1_bike"
# HTTPS_PROXY="http://10.60.28.99:86"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "eval adamw1_bigtail13i_t1_bike"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/mybikedata  \
--eval_only  \
--batch_size 1 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/adamw1_bigtail13i_t1_bike/adamw1_bigtail13i_t1_bike_checkpoint_valid_mae=-2.6874878883361815.pth \
--datasetname shanghaitech_non_overlap_test_with_densitygt \
--eval_density \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience