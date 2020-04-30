# All thing in this folder handle data

The dataset, which is publicly available on the internet, belong to their original author. I only re-upload and process the dataset to use for my own project. I make it publicly available so I might save some of your time.

Shanghaitech dataset [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://pdfs.semanticscholar.org/7ca4/bcfb186958bafb1bb9512c40a9c54721c9fc.pdf)

UCF-CC-50 dataset [Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images](http://openaccess.thecvf.com/content_cvpr_2013/papers/Idrees_Multi-source_Multi-scale_Counting_2013_CVPR_paper.pdf)

### Kaggle dataset

You will need a Kaggle account, and a working installation of Conda on your machine.

First, install [Kaggle](https://anaconda.org/conda-forge/kaggle) package by conda  
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