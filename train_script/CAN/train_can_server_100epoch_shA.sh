CUDA_VISIBLE_DEVICES=4 nohup python train_context_aware_network.py  \
--task_id can_default_shtA_100  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--output saved_model/context_aware_network \
--datasetname shanghaitech_keepfull \
--epochs 100 > logs/can_default_shtA_100_nohup.log  &