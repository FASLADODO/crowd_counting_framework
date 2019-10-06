#python main_pacnn.py \
#--input data/ShanghaiTech/part_A \
#--load_model saved_model/train_state2_attemp4_265_checkpoint.pth.tar \
#--PACNN_PERSPECTIVE_AWARE_MODEL 0 \
#--PACNN_MUTILPLE_SCALE_LOSS 0 \
#--test \
#--task_id test

python main_pacnn.py \
--input data/ShanghaiTech/part_A \
--load_model saved_model/train_state1_attemp7_180_checkpoint.pth.tar \
--PACNN_PERSPECTIVE_AWARE_MODEL 0 \
--PACNN_MUTILPLE_SCALE_LOSS 0 \
--test \
--task_id test

