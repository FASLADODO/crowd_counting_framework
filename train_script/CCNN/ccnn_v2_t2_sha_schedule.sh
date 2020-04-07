CUDA_VISIBLE_DEVICES=5 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn_lrscheduler.py  \
--task_id ccnn_v2_t2_sha_schedule  \
--note "ccnnv2 with decay and 20p augment, much lower lr"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_20p \
--epochs 602 > logs/ccnn_v2_t2_sha_schedule.log  &