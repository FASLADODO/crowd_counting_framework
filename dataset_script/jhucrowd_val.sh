HTTPS_PROXY="http://10.60.28.99:86" nohup python dataset_script/jhucrowd_density_map.py \
--task_id jhu_val \
--input /data/rnd/thient/thient_data/jhu_crowd_plusplus/train_data_validate_split \
 > logs/jhucrowd_density_map_val_t3_to_full.log  &

 echo logs/jhucrowd_density_map_val_t3_to_full.log