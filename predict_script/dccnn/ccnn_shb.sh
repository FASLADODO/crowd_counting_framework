task="eval_g1_ccnn_v7_t3_shb"
# HTTPS_PROXY="http://10.60.28.99:86"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "eval g1_ccnn_v7_t3_shb"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--eval_only  \
--batch_size 1 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/g1_ccnn_v7_t3_shb/g1_ccnn_v7_t3_shb_checkpoint_valid_mae=-8.881268501281738.pth \
--datasetname shanghaitech_non_overlap_test_with_densitygt \
--eval_density \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience