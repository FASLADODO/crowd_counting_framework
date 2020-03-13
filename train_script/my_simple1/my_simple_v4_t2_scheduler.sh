CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_attn_can_adcrowdnet_simple_lrscheduler.py  \
--task_id simple_v4_t2_scheduler  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull_and_crop \
--epochs 120 > logs/simple_v4_t2_scheduler.log  &