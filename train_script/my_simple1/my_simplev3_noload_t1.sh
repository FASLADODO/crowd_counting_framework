CUDA_VISIBLE_DEVICES=4 nohup python train_attn_can_adcrowdnet_simple.py  \
--task_id simple_v3_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull \
--epochs 22 > logs/simple_v3_t1.log  &