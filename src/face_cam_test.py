from datetime import datetime
import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import tensorflow as tf
import scipy.stats as stats
from detector import Detector
from util import *
import argparse

root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'datasets/face_impression/images')
testset_path = os.path.join(root_path,'datasets/face_impression/test.pickle')
label_path = os.path.join(root_path,'datasets/face_impression/label.pickle')
model_path = os.path.join(root_path,'trained_models/VGG/face')
out_path = os.path.join(root_path,'out/')
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')


def testNet(model,output_file):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    testset = pd.read_pickle(testset_path)
    labelset = pd.read_pickle(label_path)
    
    baseline = np.array([0.24,0.39,0.41,0.71,0.25,0.55,0.52,0.50,0.27,0.49,0.50,0.72,0.58,0.62,
            0.62,0.56,0.18,0.72,0.75,0.52,0.65,0.72,0.72,0.53,0.33,0.24,0.78,0.84,0.55,0.42,0.55,
            0.69,0.30,0.49,0.74,0.28,0.45,0.27,0.43,0.60])
    
    n_labels = 40
    index_want = np.argsort(baseline)[::-1][0:n_labels]

    batch_size = 40
    
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
        saver.restore( sess, os.path.join( model_path, model))
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
                row = row[index_want]
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
            temp = testset['label'].values[i]
            testOutMatrix[i,:] = temp[index_want].flatten()
    
        print("====Starting calculating spearman rank correlation====")
    
        phos = []
        p_values = []
    
        for i in range(n_labels):
            pho, p_value = stats.stats.spearmanr(testOutMatrix[:,i],pred[:,i])
            phos.append(pho)
            p_values.append(p_value)
        combine = list(zip(labelset["labelname"].values[index_want],phos,p_values))
        combine = sorted(combine,key=lambda x:x[1],reverse=True)
    
        with open(os.path.join(out_path,output_file),'w+') as f:
            for i in range(n_labels):
                f.write(combine[i][0].ljust(20)+" "+str(combine[i][1]).ljust(20)+" "+str(combine[i][2]).ljust(20)+"\n")
                print(combine[i])
        print("====Write result into out/{}====".format(output_file))


if __name__=="__main__":
    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.description='Face Class Activation Map Training script'
    parser.add_argument('--file', help="name of file to output", default="out_spearman.txt")
    parser.add_argument('--model',help="Model used to test",default= None)
    allPara = parser.parse_args()
    filename = allPara.file
    model = allPara.model

    #start test   
    #python3 src/face_cam_test.py --model loss_weight-14 --file pretrianed.txt
    testNet(model = model,output_file=filename)
