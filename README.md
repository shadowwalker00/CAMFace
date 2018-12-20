# CAMFace

> This project uses class activation map for to generate heatmaps of face images to tell which part of face have the most effect on a specific attribute such as happy or calm.

## Requirement

* Numpy 1.14.2
* Pandas 0.22.0
* Skimage 0.13.1
* Tensorflow 1.12.0

## Usage

### 1. Prepare Dataset

We are using **MIT2kFace** as our dataset. We use class Dataset to create train and test dataset. Since we didn't upload the train and test dataset. First of all you need to create the dataset by executing the following code. For example,

```
python3 src/dataset.py --name face
```

After finishing this, we can find the correspoding **train.pickle, test.pickle and label.pickle** in the directory it shows.

### 2. Train Network

#### 1. VGG Pretrained Weight

In the project, we utilize **VGG16** as our network and initialize weights from Conv1_1 to Conv5_3 using the pretrained weight. Download [VGG Pretrained](https://drive.google.com/open?id=12lGgU9XjM4qIYzDJ7GtbjfEmcYir0E15) from here. Put it into directory "trained_models/pretrained_weight/VGG". In the project, we learn the whole network.

#### 2. Train Process

```
python3 src/face_cam_train.py --epoch 15 --model modelname
```

If want train only with CPU, annotate the following line in trainNet function

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

After training, you can save model in "/trained_models/VGG/face" and a loss.pickle in "/out/" which can use **plot.py** script to plot the loss value change.

### 3. Test Network

In this part, executing the following code to test the model. In our project, we use spearman rank value to measure the performance. Therefore, the pameter --file is the filename to save spearman table, you can find the file in path "out/".

```
python3 src/face_cam_test.py --model loss_weight-14 --file weightLoss.txt
```

For example, part of the spearman table shows like this

| attribtue  |    correlation     |        p-value         |
| :--------: | :----------------: | :--------------------: |
|   happy    | 0.7802024542631187 | 1.7364721738541213e-91 |
| attractive | 0.7793666138503434 | 3.6073070608744484e-91 |
|    cold    | 0.7790316528531486 | 4.830991142812777e-91  |
|  friendly  | 0.7581367737446818 | 1.5227409100367313e-83 |
|  unhappy   | 0.7551754597936365 | 1.526485942129894e-82  |

### 4. Generate Compare faces under opposite traits



### 5. Generate Single Heatmap



## Result

**Top 5 face with highest value for happy attribute and Last 5 with the lowest value for happy attribute**

[![happy_unhappy.jpg](https://i.loli.net/2018/12/20/5c1acc5330639.jpg)](https://i.loli.net/2018/12/20/5c1acc5330639.jpg)

**Example to generate the heatmap of happy attribute with the given face**

[![hillary1_caring_4.11.png](https://i.loli.net/2018/12/20/5c1acc5330bc3.png)](https://i.loli.net/2018/12/20/5c1acc5330bc3.png)