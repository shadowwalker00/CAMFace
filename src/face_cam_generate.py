import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import cv2
import skimage.io
import tensorflow as tf

from util import load_image
from detector import Detector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'datasets/face_impression/images/')
testset_path = os.path.join(root_path,'datasets/face_impression/test.pickle')
label_path = os.path.join(root_path,'datasets/face_impression/label.pickle')
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')
pretrained_model = None
model_path = os.path.join(root_path,'trained_models/VGG/face/')
saved_model_name_testing = 'loss_weight-14'
output_image_path = 'out/cam/compare2'


testset = pd.read_pickle(testset_path)
labelset = pd.read_pickle(label_path)
n_labels = 40

#These are the opposite pairs for the labels
attribute_wantvis = [(21,14),(27,18),(34,10),(23,7),(31,11),(29,1),(20,2),(33,19),
 (38,9),(36,5),(22,3),(28,6),(30,12),(39,13),(37,4),(35,0),(25,16),(32,8),(26,17),(24,15)]


#convert the ground truth into a matrix 
gt = np.zeros((testset["label"].values.shape[0],40))
for i in range(testset["label"].values.shape[0]):
    gt[i,:] = testset["label"].values[i]


#combine the attribute pair with the selected top 5 images index for each attribute
newL = []    
for t1,t2 in attribute_wantvis:
    rank_t1 = sorted(zip(range(gt[:,t1].shape[0]),gt[:,t1]),key=lambda x:x[1],reverse=True)
    rank_t2 = sorted(zip(range(gt[:,t2].shape[0]),gt[:,t2]),key=lambda x:x[1],reverse=True)
    top5 = [x[0]+4001 for x in rank_t1[:10:2]]+[x[0]+4001 for x in rank_t2[:10:2]]
    newL.append((t1,t2,top5))

graph = tf.Graph()
with graph.as_default():
    images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
    labels_tf = tf.placeholder( tf.int64, [None], name='labels')
    detector = Detector(weight_path,n_labels)
    c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference(images_tf )
    saver = tf.train.Saver()
    classmap = detector.get_classmap( labels_tf, conv6 )

with tf.Session(graph=graph) as sess:
    saver.restore( sess, os.path.join(model_path, saved_model_name_testing))


    for trait1,trait2,selectionlist in newL:
        print(selectionlist)
        current_data = testset.loc[selectionlist]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(list(map(lambda x: load_image(os.path.join(image_path,x)), current_image_paths)))
        good_index = np.array(list(map(lambda x: x is not None, current_images)))    
    
        current_data = current_data[good_index]          
        current_image_paths = current_image_paths[good_index]    
        current_images = np.stack(current_images[good_index])
    
        current_labels = np.array(current_data['label'].values)            
        current_labels_deal = np.zeros((current_labels.shape[0],n_labels))

        #transform the label into matrix
        for index,row in enumerate(current_labels):
            current_labels_deal[index,:] = row

        conv6_val, output_val = sess.run([conv6, output],feed_dict={images_tf: current_images})
        classmap_answer_1 = sess.run(classmap,feed_dict={labels_tf: [trait1]*current_images.shape[0],conv6: conv6_val})
        classmap_answer_2 = sess.run(classmap,feed_dict={labels_tf: [trait2]*current_images.shape[0],conv6: conv6_val})

        classmap_vis_1 = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer_1)
        classmap_vis_2 = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer_2)

        plt.figure()
        plt.subplots_adjust(wspace=0.3, hspace =0.3)
        for index_image, (vis, ori,ori_path) in enumerate(zip(classmap_vis_1, current_images, current_image_paths)):
            ax = plt.subplot(4,5,index_image+1)
            ax.set_title("GT:{:.2f},PR:{:.2f}".format(current_labels_deal[index_image,trait1],output_val[index_image,trait1]),fontsize=8)
            plt.axis('off')
            plt.imshow(ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')            
        for index_image, (vis, ori,ori_path) in enumerate(zip(classmap_vis_2, current_images, current_image_paths)):                 
            ax = plt.subplot(4,5,index_image+1+10)
            ax.set_title("GT:{:.2f},PR:{:.2f}".format(current_labels_deal[index_image,trait2],output_val[index_image,trait2]),fontsize=8)
            plt.axis('off')
            plt.imshow(ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')        
        newname = "{}_{}.jpg".format(labelset["labelname"].values[trait1],labelset["labelname"].values[trait2])
        print(newname+" has been generated with 10 samples")
        plt.savefig(os.path.join(output_image_path,newname))
