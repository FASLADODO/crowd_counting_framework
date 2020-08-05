nohup python debug/verify_sha.py \
--input  /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A/train_data_train_split \
--output viz/verify_sha_3_train_split_shanghaitech_keepfull_r50 \
--dataset  shanghaitech_keepfull_r50  \
--meta_data  logs/verify_sha_3_train_split_shanghaitech_keepfull_r50.csv  \
> logs/verify_sha_3_train_split_shanghaitech_keepfull_r50.log &