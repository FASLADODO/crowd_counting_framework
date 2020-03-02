CUDA_VISIBLE_DEVICES=4 nohup python train_attn_can_adcrowdnet_simple.py  \
--task_id simple_v3_t3  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--load_model saved_model/simple_v3_t1/simple_v3_t2_checkpoint_64800.pth  \
--lr 1e-7 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull \
--epochs 91 > logs/simple_v3_t3.log  &