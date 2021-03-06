task="performance_shb_BigTail13i_t7"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python debug/perfomance_test_on_shb.py  \
--task_id $task  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--datasetname shanghaitech_non_overlap \
--load_model saved_model_best/adamw1_bigtail13i_t1_shb/adamw1_bigtail13i_t1_shb_checkpoint_valid_mae=-7.574910521507263.pth  \
--skip_train_eval \
--cache \
--pin_memory \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience