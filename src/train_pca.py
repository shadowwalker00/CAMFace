"""
train PCA to reduce
"""

from datetime import datetime
import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import tensorflow as tf
from util import *
from detector import Detector
import argparse
from sklearn.decomposition import PCA


root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'datasets/face_impression/images')
trainset_path = os.path.join(root_path,'datasets/face_impression/train.pickle')
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')
model_path = os.path.join(root_path,'trained_models/VGG/face')
pretrained_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/')


# Hyper Parameters
weight_decay_rate = 0.005
momentum = 0.9
batch_size = 40

def trainPCA(model_name,n_epochs):
    """
    trianNet: train the whole network

    paras: model - name of model want to save
           n_epochs - number of epoch times 
    
    return: no actual output, save trained model and loss.pkl which can used to plot the loss line
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    total_loss = []
    init_learning_rate = 0.001
    #read data   
    trainset = pd.read_pickle(trainset_path)
    print ('Read from disk: trainset')
    now = datetime.now(pytz.timezone('US/Eastern'))

    seconds_since_epoch_start = time.mktime(now.timetuple())
    graph = tf.Graph()
    pcaData = None
    iterations = 0
    with graph.as_default():
        learning_rate = tf.placeholder( tf.float32, [])   #learning rate
        images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")         # image placeholder
        #Modify: placeholder's size
        labels_tf = tf.placeholder( tf.float32, [None,40], name='labels')                   # label placeholder

        detector = Detector(weight_path, 40)
        p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)               # return each conv
        saver = tf.train.Saver()


    with tf.Session(graph=graph) as sess:        
        saver.restore( sess, os.path.join( model_path, model_name))            
        trainset = trainset.loc[np.random.permutation(len(trainset))]
        for start, end in zip(range( 0, len(trainset)+batch_size, batch_size),range(batch_size, len(trainset)+batch_size, batch_size)):
            current_data = trainset[start:end]
            current_image_paths = current_data['image_path'].values    #return batch imagePaths with type of np array
            
            #Modify: image path
            current_images = np.array(list(map(lambda x: load_image(os.path.join(image_path,x)), current_image_paths)))

            good_index = np.array(list(map(lambda x: x is not None, current_images)))

            current_data = current_data[good_index]
            current_images = np.stack(current_images[good_index])

            
            # Obtaining the label of each image
            # transform it into a None*44 2d matrix
            current_labels = np.array(current_data['label'].values)            
            current_labels_deal = np.zeros((current_labels.shape[0],40))
            for index,row in enumerate(current_labels):
                current_labels_deal[index,:] = row
                                
            # Run tensorflow session to start train
            gap_val= sess.run([gap],feed_dict={learning_rate: init_learning_rate,images_tf: current_images,labels_tf: current_labels_deal})                                               
            if pcaData is None:
                pcaData = gap_val[0]
            else:
                pcaData = np.vstack((pcaData,gap_val[0]))
            iterations += 1                            
            if iterations%10 == 0:
                print("====Processed: Batch:{}/{}, Iterations:{},Shape:{}====".format(start,len(trainset)+batch_size,iterations,pcaData.shape))
    
    """train PCA object"""
    #train PCA
    pcaObj=PCA(n_components='mle',svd_solver='full',copy=True)
    newData = pcaObj.fit_transform(pcaData)    
    
    #write to pca.pickle
    with open('./out/pca.pickle', 'wb') as f:
        Pickle.dump(pcaObj, f)

    print("=====================================")
    print("PCA output shape is: {}".format(newData.shape))
    print("Written to file named with /out/pca.pickle")
    print("=====================================")

    """processing time"""
    now = datetime.now(pytz.timezone('US/Eastern'))
    seconds_since_epoch_end = time.mktime(now.timetuple())
    print ('Processing took ' + str( np.around( (seconds_since_epoch_end - seconds_since_epoch_start)/60.0 , decimals=1) ) + ' minutes.')    

if __name__=="__main__":
    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.description='Face Class Activation Map Training script'
    parser.add_argument('--epoch', help="Epoch times for training", default=15)
    parser.add_argument('--model',help="Model name want to save",default= "MyModel")
    allPara = parser.parse_args()
    model_name = allPara.model
    epochs = allPara.epoch
    trainPCA(model_name=model_name,n_epochs=int(epochs))
                





