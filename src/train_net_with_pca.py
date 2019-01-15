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

def trainNet(model_name,n_epochs):
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
    with graph.as_default():
        learning_rate = tf.placeholder( tf.float32, [])   #learning rate
        images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")         # image placeholder
        #Modify: placeholder's size
        labels_tf = tf.placeholder( tf.float32, [None,40], name='labels')                   # label placeholder

        detector = Detector(weight_path,40)
        p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)               # return each conv
    
        """
        Modify: MSE loss function
        Add Weight into the loss function
        """

        loss_weights = tf.constant([0.24,0.39,0.41,0.71,0.25,0.55,0.52,0.50,0.27,0.49,0.50,0.72,0.58,0.62,
            0.62,0.56,0.18,0.72,0.75,0.52,0.65,0.72,0.72,0.53,0.33,0.24,0.78,0.84,0.55,0.42,0.55,
            0.69,0.30,0.49,0.74,0.28,0.45,0.27,0.43,0.60])

        loss_weights = tf.reshape(loss_weights, [-1, 40])        
        loss_tf = tf.losses.mean_squared_error(labels = labels_tf, predictions=output, weights=loss_weights) 


        #regularization
        weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
    
        loss_tf += weight_decay                                                              # update
        saver = tf.train.Saver( max_to_keep=50 )

        optimizer = tf.train.MomentumOptimizer(init_learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients( loss_tf )
        grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
        train_op = optimizer.apply_gradients( grads_and_vars )

        

    with tf.Session(graph=graph) as sess:    
        tf.global_variables_initializer().run()

        #show the trainable variables
        #variable_names = [v.name for v in tf.trainable_variables()]
        #print(variable_names)        
        iterations = 0
        loss_list = []
        print ('Starting the training ...')
        for epoch in range(n_epochs):
            trainset.index = range(len(trainset))
            #Shuffle the index of all the trainset
            trainset = trainset.loc[np.random.permutation(len(trainset) )]
        
            for start, end in zip(
                range( 0, len(trainset)+batch_size, batch_size),
                range(batch_size, len(trainset)+batch_size, batch_size)):

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
                _, loss_val, output_val = sess.run(
                        [train_op, loss_tf, output],
                        feed_dict={
                            learning_rate: init_learning_rate,
                            images_tf: current_images,
                            labels_tf: current_labels_deal
                            })
            
                print("loss",loss_val)
                loss_list.append(loss_val)   #store the loss value
                total_loss.append(loss_val)  #store loss value to the total list, help to visualize the variation

                iterations += 1            
                #Print out every 10 iterations
                if iterations % 10 == 0:
                    print ("======================================")
                    print ("Epoch", epoch + 1, "Iteration", iterations)
                    print ("Processed", start, '/', len(trainset))
                    print ("Training Loss:", np.mean(loss_list))
                    print ("======================================")
                    loss_list = []
            print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print ("producing model after epoch:{}".format(epoch+1))
            print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            saver.save( sess, os.path.join(model_path,model_name), global_step=epoch)            
            init_learning_rate *= 0.99
    
    now = datetime.now(pytz.timezone('US/Eastern'))
    seconds_since_epoch_end = time.mktime(now.timetuple())
    #save total loss to the file 
    with open(root_path+"/out/loss.pkl","wb") as f:
        Pickle.dump(total_loss, f)

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
    trainNet(model_name=model_name,n_epochs=int(epochs))
                





