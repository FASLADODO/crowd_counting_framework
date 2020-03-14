
CUDA_VISIBLE_DEVICES=2  HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn_lrscheduler.py  \
--task_id ccnn_v2_t1_scheduler  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 0 \
--datasetname shanghaitech_keepfull_and_crop \
--epochs 301 > logs/ccnn_v2_t1_scheduler.log  &