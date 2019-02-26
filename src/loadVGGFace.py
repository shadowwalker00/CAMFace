import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import skimage.io
import skimage.transform
import tensorflow as tf



data_url = "./trained_models/pretrained_weight/VGG/vggface16.tfmodel"

# Directory to store the downloaded data.
data_dir = "vgg16/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "vgg16.tfmodel"

########################################################################




########################################################################


class VGG16:
    """
    The VGG16 model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the VGG16 model
    will be loaded and can be used immediately without training.
    """

    # Name of the tensor for feeding the input image.
    tensor_name_input_image = "images:0"

    # Names of the tensors for the dropout random-values..
    tensor_name_dropout = 'dropout/random_uniform:0'
    tensor_name_dropout1 = 'dropout_1/random_uniform:0'

    # Names for the convolutional layers in the model for use in Style Transfer.
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):
        # Now load the model from file. The way TensorFlow
        # does this is confusing and requires several steps.

        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():

            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            path = data_url

            with tf.gfile.GFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

                # Now self.graph holds the VGG16 model from the proto-buf file.

            # Get a reference to the tensor for inputting images to the graph.
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get references to the tensors for the commonly used layers.
            tensor_list = [op.values() for op in self.graph.get_operations()]
            for tensor in tensor_list[:-1]:
                with tf.Session():
                    tensor = tensor[0]
                    print(tensor.eval())
            self.layer_tensors = [self.graph.get_tensor_by_name(name) for name in self.layer_names]
            print(self.layer_tensors)

    def get_layer_tensors(self, layer_ids):
        """
        Return a list of references to the tensors for the layers with the given id's.
        """

        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        """
        Return a list of names for the layers with the given id's.
        """

        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        """
        Return a list of all the layers (operations) in the graph.
        The list can be filtered for names that start with the given string.
        """

        # Get a list of the names for all layers (operations) in the graph.
        names = [op.name for op in self.graph.get_operations()]

        # Filter the list of names so we only get those starting with
        # the given string.
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]

        return names

    def create_feed_dict(self, image):
        """
        Create and return a feed-dict with an image.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the graph in TensorFlow.
        """

        # Expand 3-dim array to 4-dim by prepending an 'empty' dimension.
        # This is because we are only feeding a single image, but the
        # VGG16 model was built to take multiple images as input.
        image = np.expand_dims(image, axis=0)

        if False:
            # In the original code using this VGG16 model, the random values
            # for the dropout are fixed to 1.0.
            # Experiments suggest that it does not seem to matter for
            # Style Transfer, and this causes an error with a GPU.
            dropout_fix = 1.0

            # Create feed-dict for inputting data to TensorFlow.
            feed_dict = {self.tensor_name_input_image: image,
                         self.tensor_name_dropout: [[dropout_fix]],
                         self.tensor_name_dropout1: [[dropout_fix]]}
        else:
            # Create feed-dict for inputting data to TensorFlow.
            feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

if __name__=="__main__":
    obj = VGG16()
