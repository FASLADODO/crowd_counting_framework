OMP_NUM_THREADS=10 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python shanghaitech-fixed.py  \
--root "/data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech"  \
--part "b_test"      \
--output "/data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3"      \
--trunc 3.0   \
> b_test.log  &
