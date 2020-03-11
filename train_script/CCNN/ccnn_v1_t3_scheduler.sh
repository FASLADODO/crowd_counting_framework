CUDA_VISIBLE_DEVICES=2  HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn_lrscheduler.py  \
--task_id ccnn_v1_t3_scheduler  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 0 \
--datasetname shanghaitech \
--epochs 300 > logs/ccnn_v1_t3_scheduler.log  &