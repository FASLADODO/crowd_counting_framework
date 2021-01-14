task="eval_l2_adamw2_bigtail13i_t12_sha"
# HTTPS_PROXY="http://10.60.28.99:86"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "eval dccnn l2_adamw2_bigtail13i_t12_sha sha"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A  \
--eval_only  \
--batch_size 1 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/l2_adamw2_bigtail13i_t12_sha/l2_adamw2_bigtail13i_t12_sha_checkpoint_valid_mae=-90.4343.pt \
--datasetname shanghaitech_non_overlap_test_with_densitygt \
--eval_density \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience