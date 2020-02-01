PWD="$(pwd)"
DATAPATH=/data
# if DATAPATH is not exist, please manually make it

 # mv shanghaitech-with-people-density-map.zip data/ ; unzip data/shanghaitech-with-people-density-map.zip

 # mv ucf-cc-50-with-people-density-map.zip  ../data/ ; cd ../data/ ;unzip ucf-cc-50-with-people-density-map.zip

# mv perspective-shanghaitech.zip $DATAPATH
# mv shanghaitech-with-people-density-map.zip $DATAPATH

unzip perspective-shanghaitech.zip $DATAPATH
unzip shanghaitech-with-people-density-map.zip $DATAPATH

# put perspective
mv  $DATAPATH/perspective-ShanghaiTech/A/train_pmap/  $DATAPATH/ShanghaiTech/part_A/train_data/pmap/
mv  $DATAPATH/perspective-ShanghaiTech/A/test_pmap/  $DATAPATH/ShanghaiTech/part_A/test_data/pmap/
mv  $DATAPATH/perspective-ShanghaiTech/B/train_pmap/  $DATAPATH/ShanghaiTech/part_B/train_data/pmap/
mv  $DATAPATH/perspective-ShanghaiTech/B/test_pmap/  $DATAPATH/ShanghaiTech/part_B/test_data/pmap/