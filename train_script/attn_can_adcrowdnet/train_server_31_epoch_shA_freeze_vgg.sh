CUDA_VISIBLE_DEVICES=5 nohup python train_attn_can_adcrowdnet_freeze_vgg.py  \
--task_id attn_can_adcrowdnet_default_shtA_31_freeze_vgg  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--output saved_model/attn_can_adcrowdnet_default_shtA_31_freeze_vgg \
--datasetname shanghaitech \
--epochs 31 > logs/attn_can_adcrowdnet_default_shtA_31_freeze_vgg_nohup.log  &