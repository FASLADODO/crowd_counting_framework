CUDA_VISIBLE_DEVICES=3 nohup python train_attn_can_adcrowdnet_simple.py  \
--task_id simple_1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 5e-4 \
--datasetname shanghaitech \
--epochs 33 > logs/simple_1.log  &