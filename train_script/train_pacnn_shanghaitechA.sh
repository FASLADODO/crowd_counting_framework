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

python main_pacnn.py \
--input data/ShanghaiTech/part_A \
--load_model saved_model/train_state1_attemp3_30_checkpoint.pth.tar \
--epochs 151 \
--lr 1e-7 \
--task_id train_state1_attemp4