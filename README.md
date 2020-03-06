### Environment
```
conda create -n env python=3.7 anaconda
conda install pytorch torchvision cpuonly -c pytorch 
conda install -c conda-forge opencv
<!--  pip install comet_ml -->
conda install -c comet_ml -c conda-forge comet_ml
conda install ignite -c pytorch
conda install h5py
conda install scikit-learn
conda install -c anaconda pillow
conda install -c anaconda numpy
```

for CUDA 9.0

```shell script
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

### make data folder
Let make `/data` folder at root
```
cd /
sudo mkdir data
sudo chown -R tt /data/
```

meow
