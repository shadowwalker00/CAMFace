# CAMFace

> This project uses class activation map for to generate heatmaps of face images to tell which part of face have the most effect on a specific attribute such as happy or calm.

## Requirement



## Usage



### VGG Pretrained Weight

The file is beyond 100M limitation so I store it into my Google Drive. Download [VGG Pretrained](https://drive.google.com/open?id=12lGgU9XjM4qIYzDJ7GtbjfEmcYir0E15) from here. Then put it into directory "trained_models/VGG".

## Result

**Top 5 face with highest value for happy attribute and Last 5 with the lowest value for happy attribute **

[![happy_unhappy.jpg](https://i.loli.net/2018/12/20/5c1acc5330639.jpg)](https://i.loli.net/2018/12/20/5c1acc5330639.jpg)

**Example to generate the heatmap of happy attribute with the given face**

[![hillary1_caring_4.11.png](https://i.loli.net/2018/12/20/5c1acc5330bc3.png)](https://i.loli.net/2018/12/20/5c1acc5330bc3.png)