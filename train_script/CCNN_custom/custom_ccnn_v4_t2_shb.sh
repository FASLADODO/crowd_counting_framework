CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn_sgd.py  \
--task_id custom_ccnn_v4_t2_shb  \
--note "let train with sgd"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 5e-5 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--batch_size 1 \
--epochs 50 > logs/custom_ccnn_v4_t1_shb.log  &


#python train_custom_compact_cnn.py  \
#--task_id custom_ccnn_v4_t1_shb  \
#--note "defromable"  \
#--input /data/ShanghaiTech/part_B/  \
#--lr 1e-5 \
#--decay 1e-5 \
#--datasetname shanghaitech_keepfull \
#--batch_size 2 \
#--epochs 200