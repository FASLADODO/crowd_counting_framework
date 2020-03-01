CUDA_VISIBLE_DEVICES=3 nohup python train_attn_can_adcrowdnet_simple.py  \
--task_id simple_v2_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull \
--epochs 46 > logs/simple_v2_t2.log  &