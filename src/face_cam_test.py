from datetime import datetime
import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import cv2
import skimage.io
import skimage.transform
import tensorflow as tf
import scipy.stats as stats
from Detector import Detector

root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'datasets/face_impression/images')
testset_path = os.path.join(root_path,'datasets/face_impression/test.pickle')
label_path = os.path.join(root_path,'datasets/face_impression/label.pickle')
model_path = os.path.join(root_path,'trained_models/VGG/')
out_path = os.path.join(root_path,'out/')
saved_model_name_testing = 'uncrop_model-14'
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')




def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [224,224] , mode='constant')     #resize the image here
    return resized_img   



testset = pd.read_pickle(testset_path)
labelset = pd.read_pickle(label_path)
n_labels = 40
batch_size = 20
graph = tf.Graph()
with graph.as_default():
    images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
    labels_tf = tf.placeholder( tf.int64, [None,n_labels], name='labels')
    detector = Detector(weight_path,n_labels)
    c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference(images_tf)
    loss_tf = tf.losses.mean_squared_error(labels=labels_tf, predictions=output) 
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    acc_mean = 0 
    acc_list = []
    count = 0
    pred = None
    saver.restore( sess, os.path.join( model_path, saved_model_name_testing))
    for start, end in zip(range( 0, len(testset)+batch_size, batch_size),range(batch_size, len(testset)+batch_size, batch_size)):
        count += 1 
        current_data = testset[start:end]
        current_image_paths = current_data['image_path'].values                     #return image paths
        current_images = np.array(list(map(lambda x: load_image(os.path.join(image_path,x)), current_image_paths))) #load image
        good_index = np.array(list(map(lambda x: x is not None, current_images)))   #screen out those are None
        current_data = current_data[good_index]          
        current_image_paths = current_image_paths[good_index]    
        current_images = np.stack(current_images[good_index])   #image data    
        current_labels = np.array(current_data['label'].values)            
        current_labels_deal = np.zeros((current_labels.shape[0],n_labels))
        for index,row in enumerate(current_labels):
            current_labels_deal[index,:] = row
        acc, output_val = sess.run([loss_tf, output],feed_dict={images_tf: current_images,labels_tf: current_labels_deal})        
        acc_list.append(acc)
        if pred is None:
            pred = output_val
        else:
            pred = np.vstack((pred,output_val))


        print("prediction error for batch({}-{}):{}".format(start,end,sum(acc_list)/len(acc_list)))
        acc_mean += sum(acc_list)/len(acc_list)
    print("Overall Prediction Error for Testset (size:{}):{}".format(testset.shape,acc_mean/count))
    testOutMatrix = np.zeros((testset.shape[0],n_labels))
    for i in range(testset.shape[0]):
        testOutMatrix[i,:] = testset['label'].values[i].flatten()
    


    print("====Starting calculating spearman rank correlation====")
    
    phos = []
    p_values = []
    
    for i in range(n_labels):
        pho, p_value = stats.stats.spearmanr(testOutMatrix[:,i],pred[:,i])
        phos.append(pho)
        p_values.append(p_value)
    combine = list(zip(labelset["labelname"].values,phos,p_values))
    combine = sorted(combine,key=lambda x:x[1],reverse=True)

    with open(os.path.join(out_path,"spearman.txt"),'w+') as f:
        for i in range(n_labels):
            f.write(combine[i][0])
            f.write("\t\t\t")                        
            f.write(str(combine[i][1]))
            f.write("\t")
            f.write(str(combine[i][2]))
            f.write("\n") 
    print("====Write result into spearman.txt")
