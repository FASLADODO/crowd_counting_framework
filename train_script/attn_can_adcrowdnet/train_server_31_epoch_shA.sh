CUDA_VISIBLE_DEVICES=4 nohup python train_attn_can_adcrowdnet.py  \
--task_id attn_can_adcrowdnet_default_shtA_31  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--output saved_model/attn_can_adcrowdnet_default_shtA_31 \
--datasetname shanghaitech_keepfull \
--epochs 31 > logs/attn_can_adcrowdnet_default_shtA_31_nohup.log  &