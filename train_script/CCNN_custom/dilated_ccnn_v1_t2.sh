#CUDA_VISIBLE_DEVICES=5 nohup python train_custom_compact_cnn_lrscheduler.py  \
#--task_id dilated_ccnn_v1_t1  \
#--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
#--lr 1e-4 \
#--decay 5e-4 \
#--datasetname shanghaitech \
#--epochs 400 > logs/dilated_ccnn_v1_t1.log  &


CUDA_VISIBLE_DEVICES=5 nohup python train_custom_compact_cnn_lrscheduler.py  \
--task_id dilated_ccnn_v1_t2  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-4 \
--datasetname shanghaitech \
--epochs 400 > logs/dilated_ccnn_v1_t2.log  &