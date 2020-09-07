HTTPS_PROXY="http://10.60.28.99:86" nohup python dataset_script/jhucrowd_density_map.py \
--task_id jhu_test \
--input /data/rnd/thient/thient_data/jhu_crowd_plusplus/test_data \
 > logs/jhucrowd_density_map_test_t2.log  &

 echo logs/jhucrowd_density_map_test_t2.log