HTTPS_PROXY="http://10.60.28.99:86" nohup python dataset_script/jhucrowd_density_map.py \
--task_id jhu_train \
--input /data/rnd/thient/thient_data/jhu_crowd_plusplus/train_data_train_split \
 > logs/jhucrowd_density_map_train_t3.log  &

 echo logs/jhucrowd_density_map_train_t3.log