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
from Detector import Detector
import csv

root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'datasets/face_impression/cropped_faces/')
testset_path = os.path.join(root_path,'datasets/face_impression/test.pickle')
label_path = os.path.join(root_path,'datasets/face_impression/label.pickle')
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')
pretrained_model = None
model_path = os.path.join(root_path,'trained_models/VGG/')
saved_model_name_testing = 'uncrop_model-14'
output_image_path = 'out/uncrop_result/'

labelset = pd.read_pickle(label_path)

        
n_labels = 40

#These are the opposite pairs for the labels
#attribute_wantvis = [(21,14),(27,18),(34,10),(23,7),(31,11),(29,1),(20,2),(33,19),
# (38,9),(36,5),(22,3),(28,6),(30,12),(39,13),(37,4),(35,0),(25,16),(32,8),(26,17),(24,15)]
def generate_heat(image_name,image_path,out_path,model='bias_model-14'):	
    graph = tf.Graph()
    res = []
    with graph.as_default():
        images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
        labels_tf = tf.placeholder( tf.int64, [None], name='labels')
        detector = Detector(weight_path,n_labels)
        c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference(images_tf )
        saver = tf.train.Saver()
        classmap = detector.get_classmap( labels_tf, conv6 )
        image_list = [image_name]
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, os.path.join(model_path, model))
        current_images = np.array(list(map(lambda x: load_image(os.path.join(image_path,x)), image_list)))
        print(current_images.shape)
        good_index = np.array(list(map(lambda x: x is not None, current_images)))    
        current_images = np.stack(current_images[good_index])
        conv6_val, output_val = sess.run([conv6, output],feed_dict={images_tf: current_images})
    
        for attr in range(40):
            classmap_answer = sess.run(classmap,feed_dict={labels_tf: [attr]*current_images.shape[0],conv6: conv6_val})
            classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer)    
            for index_image, (vis, ori,name) in enumerate(zip(classmap_vis, current_images, image_list)):        
                plt.figure()
                plt.imshow(ori)
                plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
                newname = "{}_{}_{:.2f}{}".format(os.path.splitext(name)[0],labelset["labelname"].values[attr],output_val[index_image,attr],os.path.splitext(name)[1])
                plt.savefig(os.path.join(root_path,out_path,newname))
                plt.close()
                print(newname+" has been generated.")
                res.append(output_val[index_image,attr])
    return res


if __name__ == "__main__":

    """
    This part of code is to generate the test image
    """
    """
    image_list = [file for file in os.listdir(image_path)]
    image_list = sorted(image_list)        
    with open(root_path+"/out/test_face.csv","w") as csvfile: 
        column_name = ["filename"]
        writer = csv.writer(csvfile)             
        column_name.extend(list(labelset["labelname"].values))
        writer.writerow(column_name)       
        store_pred = []	
        for name in image_list:
            temp = generate_heat(name,image_path,output_image_path)
            temp_l = [name]
            temp_l.extend(temp)
            store_pred.append(temp_l)
        writer.writerows(store_pred)      
    """





