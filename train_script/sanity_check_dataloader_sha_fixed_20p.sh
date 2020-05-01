CUDA_VISIBLE_DEVICES=1 nohup python sanity_check_dataloader.py  \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_A   > logs/sanity_check_dataloader_shaA_fixed.log  &

CUDA_VISIBLE_DEVICES=1 nohup python sanity_check_dataloader.py  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A   > logs/sanity_check_dataloader_shaA_geo.log  &