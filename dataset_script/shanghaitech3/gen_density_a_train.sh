OMP_NUM_THREADS=10 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python shanghaitech-adapt.py  \
--root "/data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech"  \
--part "a_train"      \
--output "/data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3"      \
--trunc 3.0   \
> a_train.log  &
