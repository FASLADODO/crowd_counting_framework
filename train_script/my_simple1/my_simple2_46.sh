CUDA_VISIBLE_DEVICES=3 nohup python train_attn_can_adcrowdnet_simple.py  \
--task_id simple_2  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull \
--epochs 46 > logs/simple_2.log  &