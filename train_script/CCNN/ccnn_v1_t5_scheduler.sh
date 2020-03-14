#CUDA_VISIBLE_DEVICES=2  HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn_lrscheduler.py  \
#--task_id ccnn_v1_t5_scheduler  \
#--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
#--lr 1e-4 \
#--decay 0 \
#--datasetname shanghaitech_keepfull_and_crop \
#--epochs 500 > logs/ccnn_v1_t5_scheduler.log  &



CUDA_VISIBLE_DEVICES=2  HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v1_t5_scheduler_c2  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-6 \
--decay 0 \
--load_model saved_model/ccnn_v1_t5_scheduler/ccnn_v1_t5_scheduler_checkpoint_597600.pth  \
--datasetname shanghaitech_keepfull_and_crop \
--epochs 700 > logs/ccnn_v1_t5_scheduler_c2.log  &