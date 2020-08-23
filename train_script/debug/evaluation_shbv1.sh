task="evaluation_shb_v1"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python debug/evaluate_shb.py  \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B/test_data  \
--output visualize/$task  \
--meta_data logs/$task.txt  \
--datasetname shanghaitech_non_overlap \
 > logs/$task.log  &

echo logs/$task.log




##
#def _parse():
#    parser = argparse.ArgumentParser(description='evaluatiuon SHB')
#    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
#    parser.add_argument('--output', action="store", type=str, default="visualize/verify_dataloader_shanghaitech")
#    parser.add_argument('--load_model', action="store", type=str, default="visualize/verify_dataloader_shanghaitech")
#    parser.add_argument('--model', action="store", type=str, default="visualize/verify_dataloader_shanghaitech")
#    parser.add_argument('--meta_data', action="store", type=str, default="data_info.txt")
#    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull_r50")
#    arg = parser.parse_args()
#    return arg
##
##