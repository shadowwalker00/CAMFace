import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import tensorflow as tf
from util import load_image
from detector import Detector
import argparse

root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'demo/')
testset_path = os.path.join(root_path,'datasets/face_impression/test.pickle')
label_path = os.path.join(root_path,'datasets/face_impression/label.pickle')
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')
model_path = os.path.join(root_path,'trained_models/VGG/face')
output_image_path = os.path.join(root_path,'demo/')

labelset = pd.read_pickle(label_path)

        
n_labels = 40

#These are the opposite pairs for the labels
#attribute_wantvis = [(21,14),(27,18),(34,10),(23,7),(31,11),(29,1),(20,2),(33,19),
# (38,9),(36,5),(22,3),(28,6),(30,12),(39,13),(37,4),(35,0),(25,16),(32,8),(26,17),(24,15)]
def demo(image_name,traitname,model):
    labellist = list(labelset["labelname"])
    index = labellist.index(traitname)
    
    graph = tf.Graph()
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
        good_index = np.array(list(map(lambda x: x is not None, current_images)))   
        print (good_index) 
        current_images = np.stack(current_images[good_index])
        conv6_val, output_val = sess.run([conv6, output],feed_dict={images_tf: current_images})
        classmap_answer = sess.run(classmap,feed_dict={labels_tf: [index]*current_images.shape[0],conv6: conv6_val})
        classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer)    
        for index_image, (vis, ori,name) in enumerate(zip(classmap_vis, current_images, image_list)):
            plt.figure()
            plt.imshow(ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
            plt.title("{}: {:.2f}".format(labelset["labelname"].values[index],output_val[index_image,index]))
            newname = "{}_cam{}".format(os.path.splitext(name)[0],os.path.splitext(name)[1])
            plt.savefig(os.path.join(output_image_path,newname))
            plt.close()
            print(newname+" has been generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description='demo script to generate CAM of given image'
    parser.add_argument('--img', help="image_path you want to generate")
    parser.add_argument('--trait', help="Name of trait you want show", default="happy")
    parser.add_argument('--model',help="Model want to use",default= "loss_weight-14")
    allPara = parser.parse_args()
    file = allPara.img
    trait_name = allPara.trait
    model = allPara.model
    #python3 src/demo.py --img test.jpeg --trait happy --model loss_weight-14
    demo(image_name=file, traitname=trait_name,model=model)






