 mkdir -p data/

 # mv shanghaitech-with-people-density-map.zip data/ ; unzip data/shanghaitech-with-people-density-map.zip

 # mv ucf-cc-50-with-people-density-map.zip  ../data/ ; cd ../data/ ;unzip ucf-cc-50-with-people-density-map.zip

mv perspective-shanghaitech.zip data/
mv shanghaitech-with-people-density-map.zip data/
cd data/
unzip perspective-shanghaitech.zip
unzip shanghaitech-with-people-density-map.zip data/
cd ..

# put perspective
mv  data/perspective-ShanghaiTech/A/train_pmap/  data/ShanghaiTech/part_A/train_data/pmap/
mv  data/perspective-ShanghaiTech/A/test_pmap/  data/ShanghaiTech/part_A/test_data/pmap/
mv  data/perspective-ShanghaiTech/B/train_pmap/  data/ShanghaiTech/part_B/train_data/pmap/
mv  data/perspective-ShanghaiTech/B/test_pmap/  data/ShanghaiTech/part_B/test_data/pmap/