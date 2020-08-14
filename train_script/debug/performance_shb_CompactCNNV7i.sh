task="performance_shb_CompactCNNV7i_t1"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python debug/perfomance_test_on_shb.py  \
--task_id $task  \
--model "CompactCNNV7i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--datasetname shanghaitech_non_overlap \
--load_model  saved_model_best/g1_ccnn_v7_t3_shb/g1_ccnn_v7_t3_shb_checkpoint_valid_mae=-8.881268501281738.pth  \
--skip_train_eval \
--cache \
--pin_memory \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience