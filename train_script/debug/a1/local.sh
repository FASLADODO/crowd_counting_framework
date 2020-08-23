task="evaluation_shb_CompactCNNV7i_t1"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python debug/evaluate_shb.py  \
--model "CompactCNNV7i" \
--input /data/ShanghaiTech/part_A/test_data  \
--output visualize/$task  \
--load_model  saved_model_best/g1_ccnn_v7_t3_shb/g1_ccnn_v7_t3_shb_checkpoint_valid_mae=-8.881268501281738.pth  \
--meta_data logs/$task.txt  \
--datasetname shanghaitech_non_overlap \
 > logs/$task.log  &

echo logs/$task.log


"/data/ShanghaiTech/part_A/test_data"


--model "CompactCNNV7i" \
--input /data/ShanghaiTech/part_A/test_data
--output visualize/evaluation_shb_CompactCNNV7i_t1
--meta_data logs/evaluation_shb_CompactCNNV7i_t1.txt
--datasetname shanghaitech_non_overlap