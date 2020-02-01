# All thing in this folder handle data

### Kaggle dataset

First, install https://anaconda.org/conda-forge/kaggle 
```bash
conda install -c conda-forge kaggle 
```

Set Kaggle api to ~/.kaggle/kaggle.json  as [instructed](https://github.com/Kaggle/kaggle-api)

### Set up dataset 

Download dataset from kaggle 

```shell script
sh download_kaggle_crowd_dataset.sh
```

Unzip and set name for dataset

```shell script
sh unzip_file.sh 
```

Default, dataset will be save to `/data`. However, you have to manually create `/data` folder. 

```shell script
sudo mkdir data
sudo chown -R tt /data/
```