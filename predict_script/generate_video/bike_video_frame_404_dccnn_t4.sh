task="bike_video_frame_404_dccnn_t4"
# HTTPS_PROXY="http://10.60.28.99:86"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python predict_image.py  \
--task_id $task  \
--note "predict image frame 404, q100, model t4 "  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/crowd_counting_video/video_frame/video_bike_q100  \
--output /data/rnd/thient/thient_data/crowd_counting_video/predict_video/ \
--eval_only  \
--batch_size 1 \
--load_model /data/rnd/thient/crowd_counting_framework/saved_model_best/adamw1_bigtail13i_t4_bike20/adamw1_bigtail13i_t4_bike20_checkpoint_valid_mae=-3.2068.pt \
--datasetname shanghaitech_non_overlap_test_with_densitygt \
--eval_density \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience