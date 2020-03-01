CUDA_VISIBLE_DEVICES=4 nohup python train_attn_can_adcrowdnet_simple.py  \
--task_id simple_v3_t2  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--load_model saved_model/simple_v3_t1/simple_v3_t1_checkpoint_22680.pth  \
--lr 1e-6 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull \
--epochs 60 > logs/simple_v3_t2.log  &