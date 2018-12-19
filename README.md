# CAMFace

> This project uses class activation map for to generate heatmaps of face images to tell which part of face have the most effect on a specific attribute such as happy or calm.

## Requirement

* Numpy 1.14.2
* Pandas 0.22.0
* Skimage 0.13.1
* Tensorflow 1.12.0

## Usage

### 1. Prepare Dataset

There're two dataset included in our project, Caltech256 and MITKface. We use class Dataset to create train and test dataset. Since we didn't upload the train and test dataset. First of all you need to create the dataset by executing the following code. For example,

```
python3 src/dataset.py --name caltech
```

After finishing this, we can find the correspoding train.pickle and test.pickle in the directory it shows.

### 2. Train Network

###VGG Pretrained Weight

The file is beyond 100M limitation so I store it into my Google Drive. Download [VGG Pretrained](https://drive.google.com/open?id=12lGgU9XjM4qIYzDJ7GtbjfEmcYir0E15) from here. Then put it into directory "trained_models/VGG".

### 3. Test Network

### 4. Generate Single Heatmap

## Result

**Top 5 face with highest value for happy attribute and Last 5 with the lowest value for happy attribute **

[![happy_unhappy.jpg](https://i.loli.net/2018/12/20/5c1acc5330639.jpg)](https://i.loli.net/2018/12/20/5c1acc5330639.jpg)

**Example to generate the heatmap of happy attribute with the given face**

[![hillary1_caring_4.11.png](https://i.loli.net/2018/12/20/5c1acc5330bc3.png)](https://i.loli.net/2018/12/20/5c1acc5330bc3.png)