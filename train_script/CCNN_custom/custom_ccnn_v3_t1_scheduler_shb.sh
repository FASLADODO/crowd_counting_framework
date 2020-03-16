CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn_lrscheduler.py  \
--task_id custom_ccnn_v3_t1_scheduler_shb  \
--note "train custom ccnn v1 branching 3 branch"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--batch_size 5 \
--epochs 400 > logs/custom_ccnn_v3_t1_scheduler_shb.log  &