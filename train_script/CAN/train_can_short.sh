#CUDA_VISIBLE_DEVICES=4 nohup python train_context_aware_network.py  \
#--task_id can_default_shtA_t1  \
#--lr 1e-4 \
#--decay 5e-4 \
#--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
#--output saved_model/can_default_shtA_t1 \
#--datasetname shanghaitech_keepfull \
#--epochs 46 > logs/can_default_shtA_t1.log  &
#

CUDA_VISIBLE_DEVICES=1 nohup python train_context_aware_network.py  \
--task_id can_default_shtA_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-4 \
--datasetname shanghaitech_keepfull \
--epochs 46 > logs/can_default_shtA_t1.log  &