CUDA_VISIBLE_DEVICES=3 nohup python train_custom_compact_cnn_lrscheduler.py  \
--task_id ccnn_v1_t1_scheduler  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 0 \
--datasetname shanghaitech \
--epochs 401 > logs/ccnn_v1_t1_scheduler.log  &