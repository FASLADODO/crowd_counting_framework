CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn_lrscheduler.py  \
--task_id ccnn_v2_t5_sha_schedule  \
--note "ccnnv2 with decay and 20p augment, much lower lr"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--datasetname shanghaitech_256 \
--epochs 1500 > logs/ccnn_v2_t5_sha_schedule.log  &