#python /home/tt/project/crowd_counting_framework/main_pacnn.py --input /home/tt/project/crowd_counting_framework/data/ShanghaiTech/part_A

#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--epochs 151 \
#--task_id train_state1_attemp1

#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp1_10_checkpoint.pth.tar \
#--epochs 151 \
#--lr 1e-6 \
#--task_id train_state1_attemp3

# trained 30

#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp3_30_checkpoint.pth.tar \
#--epochs 151 \
#--lr 1e-7 \
#--task_id train_state1_attemp4


#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp4_35_checkpoint.pth.tar \
#--epochs 151 \
#--lr 1e-8 \
#--task_id train_state1_attemp5

################3

## TODO: train this
#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp5_40_checkpoint.pth.tar \
#--epochs 300 \
#--lr 1e-8 \
#--task_id train_state1_attemp6


#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp6_120_checkpoint.pth.tar \
#--epochs 300 \
#--lr 1e-9 \
#--task_id train_state1_attemp7

#### no loss for d1, d2, d3 but only count d_final
#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp7_180_checkpoint.pth.tar \
#--epochs 300 \
#--lr 1e-9 \
#--PACNN_MUTILPLE_SCALE_LOSS False \
#--task_id train_state1_attemp8_finalloss

####################

python main_pacnn.py \
--input data/ShanghaiTech/part_A \
--load_model saved_model/train_state1_attemp7_180_checkpoint.pth.tar \
--epochs 500 \
--lr 1e-9 \
--PACNN_PERSPECTIVE_AWARE_MODEL 1 \
--PACNN_MUTILPLE_SCALE_LOSS 1 \
--task_id train_state2_attemp1


#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state1_attemp7_180_checkpoint.pth.tar
#--epochs 500
#--lr 1e-9
#--PACNN_PERSPECTIVE_AWARE_MODEL 1
#--PACNN_MUTILPLE_SCALE_LOSS 1
#--task_id dev