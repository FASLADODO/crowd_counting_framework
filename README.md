# WORK IN PROGRESS
I am trying to publish a paper for graduating. I really need to publish.

<b> Please do not copy my work and my ideal to use as your own. </b> I really need to publish (a) paper(s), I hope you understand.
<br><br>

However, you can use my framework as starter code for your own work, or use my re-implementation of other works.


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
pip install torchsummary 
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

When you use comet.ml
```shell script
conda install -c comet_ml comet_ml 
```

""
TODO: 
https://github.com/kornia/kornia/blob/master/kornia/losses/ssim.py

add ssim from here