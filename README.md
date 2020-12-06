# WORK IN PROGRESS

Use as your own risk. 


### Environment
```
conda create -n env python=3.7 anaconda
conda install pytorch torchvision cpuonly -c pytorch 
conda install -c conda-forge opencv
<!--  pip install comet_ml -->
# conda install -c comet_ml -c conda-forge comet_ml  # maybe bug
conda install ignite -c pytorch
pip install pytorch-ignite  # if conda downgrade pytorch, then use pip
conda install -c conda-forge h5py
conda install scikit-learn
conda install -c anaconda pillow  # consider conda-forge
conda install -c anaconda numpy  # 
conda install -c anaconda matplotlib  # 
conda install pandas
# pip install torchsummary 
# pip install kornia # still buggy not recommended 
conda install -c photosynthesis-team piq 
```

for CUDA 9.0

```shell script
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### make data folder
Let make `/data` folder at root
```
cd /
sudo mkdir data
sudo chown -R tt /data/
```

When you use comet.ml
```shell script
conda install -c comet_ml comet_ml 
```

""
TODO: 
https://github.com/kornia/kornia/blob/master/kornia/losses/ssim.py

add ssim from here

I have trouble using ssim, the model does not trained. If you are not sure about ssim, avoid it now. Or find big, well maintained open-source implementation of ssim.
