task="adamw1_ccnnv7_t2_bike"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=3 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "bike baseline on good model"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/mybikedata  \
--lr 1e-3 \
--decay 0.1 \
--loss_fn "MSEL1Mean" \
--batch_size 5 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model/ccnn_v7_t4_shb_fixed/ccnn_v7_t4_shb_fixed_checkpoint_71600.pth \
--datasetname my_bike_non_overlap \
--optim adamw \
--cache \
--epochs 2001 > logs/$task.log  &

echo logs/$task.log  # for convenience