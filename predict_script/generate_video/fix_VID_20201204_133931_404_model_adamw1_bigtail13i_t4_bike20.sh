task="fix_VID_20201204_133931_404_model_adamw1_bigtail13i_t4_bike20"
# HTTPS_PROXY="http://10.60.28.99:86"
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python predict_video_server.py  \
--task_id $task  \
--note "eval bike20 bad model"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/crowd_counting_video/raw_video/VID_20201204_133931_404.mp4  \
--output /data/rnd/thient/thient_data/crowd_counting_video/predict_video/ \
--batch_size 1 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/adamw1_bigtail13i_t4_bike20/adamw1_bigtail13i_t4_bike20_checkpoint_valid_mae=-3.2068.pt \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience