"""Class for creating and managing a convolutional neural network
"""
import os
import time
import json
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from sklearn.metrics import confusion_matrix
from bohemianet_input import get_train_batches, get_valid_batches


class Bohemianet(object):

    def __init__(self, model_name="model1"):
        """convolutional neural network class

        Args:
            model_name (str, optional): name of model architecture
        """
        self.model_name = model_name

        with open("models/{}.json".format(model_name)) as f:
            self.arch = json.load(f)

        self.im_size = self.arch['im_size']
        self.num_channels = self.arch['num_channels']
        self.num_classes = self.arch['num_classes']

        self.conv_layers = []
        self.build_model()

    def eval(self, ims):
        """Predict the class probablities of a batch of images

        Args:
            ims (TYPE): batch of images

        Returns:
            list: probability of each class of each image
        """
        feed_dict = {self.x: ims, self.keep_prob: 1.0}
        probs = self.y_pred.eval(session=self.session, feed_dict=feed_dict)
        return probs

    def eval_classes(self, ims):
        """Predict the class of a batch of images

        Args:
            ims (TYPE): batch of images

        Returns:
            list: predicted class of each image
        """
        feed_dict = {self.x: ims, self.keep_prob: 1.0}
        return self.session.run(self.y_pred_cls, feed_dict=feed_dict)

    def eval_accuracy(self, ims, labels):
        """Evaluate the accuracy of the model

        Args:
            ims (img_array): batch of images
            labels (label_array): batch of labels

        Returns:
            float: prediction accuracy on the batch
        """
        feed_dict = {
            self.x: ims,
            self.y_true: labels,
            self.keep_prob: 1.0
        }
        acc = self.session.run(self.accuracy, feed_dict=feed_dict)
        return acc

    def optimize(self, num_iterations=1000, batch_size=64, learning_rate=1e-4, augment=False):
        """Run the model optimizer

        Args:
            num_iterations (int, optional): number of iterations to run
            batch_size (int, optional): size of batch used at each iteration
            learning_rate (float, optional): learning rate
            augment (bool, optional): whether to augment the images
        """
        start_time = time.time()

        count = 0
        max_score = self.validate(batch_size=batch_size)

        for x_batch, y_true_batch in get_train_batches(
                num_iterations, batch_size, augment=augment):

            feed_dict = {
                self.x: x_batch,
                self.y_true: y_true_batch,
                self.keep_prob: 0.5,
                self.learning_rate: learning_rate
            }

            self.session.run(self.optimizer, feed_dict=feed_dict)

            if count % 10 == 0:
                acc = self.session.run(self.accuracy, feed_dict=feed_dict)
                msg = "Optimization Iteration: {0:>6}, "
                msg += "Training Accuracy: {1:>6.1%}"
                print(msg.format(count + 1, acc))

            if count != 0 and count % 500 == 0:
                score = self.validate(batch_size=batch_size)
                if score > max_score:
                    self.save()
                    max_score = score
                msg = "Validation Accuracy: {0:>6.1%}, "
                msg += "Max Validation Accuracy: {1:>6.1%}"
                print(msg.format(score, max_score))
            count += 1
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def validate(self, batch_size=64, conf_m=False):
        """Evaluate the accuracy of the network on the validation set

        Args:
            batch_size (int, optional): number of images to evaluate at once
            conf_m (bool, optional): whether to create a confusion matrix

        Returns:
            float: accuracy of the network on the validation set
        """
        acc = 0
        count = 0
        if conf_m:
            y_true = []
            y_pred = []
        for x_batch, y_true_batch in get_valid_batches(batch_size=batch_size):
            acc += self.eval_accuracy(x_batch, y_true_batch)
            count += 1
            if conf_m:
                y_pred.append(self.eval_classes(x_batch))
                y_true.append(
                    np.apply_along_axis(np.argmax, axis=1, arr=y_true_batch))

        if conf_m:
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            cm = confusion_matrix(y_true, y_pred)
            cm = cm[:10, :10]

            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

        return acc / count

    def save(self):
        """save the network weights
        """
        saver = tf.train.Saver()
        save_path = "models/{}.ckpt".format(self.model_name)
        save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))
        saver.save(self.session, save_path)
        print("Saved session to {}".format(save_path))

    def load(self):
        """load the network weights
        """
        saver = tf.train.Saver()
        load_path = "models/{}.ckpt".format(self.model_name)
        load_path = os.path.abspath(os.path.join(os.getcwd(), load_path))
        saver.restore(self.session, load_path)
        print("Restored session from {}".format(load_path))

    def build_model(self):
        """Build the model based on the architecture file
        """
        im_size_flat = self.im_size * self.im_size * self.num_channels

        self.keep_prob = tf.placeholder(tf.float32)  # prob for dropout layers

        #  Init NN
        self.x = tf.placeholder(
            tf.float32, shape=[None, im_size_flat], name='x')
        x_image = tf.reshape(
            self.x, [-1, self.im_size, self.im_size, self.num_channels])

        self.y_true = tf.placeholder(
            tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        prev_layer = x_image
        prev_layer_params = {
            "type": "im",
            "num_channels": self.num_channels
        }

        output_layer = self.add_layers(
            prev_layer, prev_layer_params, self.arch['layers'])

        self.y_pred = tf.nn.softmax(output_layer)
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=output_layer,
            labels=self.y_true
        )

        self.learning_rate = tf.placeholder(tf.float32)
        cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(cost)

        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def add_layer(self, prev_layer, prev_layer_params, layer_params):
        """Create a layer based on the given parameters

        Args:
            prev_layer (tensor): previous layer in the network
            prev_layer_params (dict): parameters of previous layer
            layer_params (dict): parameters of current layer

        Returns:
            tensor: created layer
        """
        input_type = prev_layer_params['type']
        output_type = layer_params['type']

        if input_type == "im" and output_type == "conv":
            layer = _new_conv_layer(
                input_layer=prev_layer,
                num_input_channels=prev_layer_params['num_channels'],
                filter_size=layer_params['filter_size'],
                num_filters=layer_params['num_filters'],
                use_pooling=layer_params['max_pool'],
                normalize=layer_params['normalize']
            )
            self.conv_layers.append(layer)


        elif input_type == "im" and output_type == "parallel_conv":
            layer = []
            for layers in layer_params['parallel_layers']:
                layer.append(
                    self.add_layers(prev_layer, prev_layer_params, layers))

        elif input_type == "conv" and output_type == "conv":
            layer = _new_conv_layer(
                input_layer=prev_layer,
                num_input_channels=prev_layer_params['num_filters'],
                filter_size=layer_params['filter_size'],
                num_filters=layer_params['num_filters'],
                use_pooling=layer_params['max_pool'],
                normalize=layer_params['normalize']
            )
            self.conv_layers.append(layer)

        elif input_type == "conv" and output_type == "fully_connected":
            layer_flat, num_features = _flatten_layer(prev_layer)

            layer = _new_fc_layer(
                input_layer=layer_flat,
                num_inputs=num_features,
                num_outputs=layer_params['num_outputs'],
                use_relu=layer_params['use_relu'],
                dropout=layer_params['dropout'],
                keep_prob=self.keep_prob
            )

        elif input_type == "parallel_conv" and output_type == "fully_connected":
            flattened_layers = []
            num_features = 0

            for layer in prev_layer:
                flattened_layer, n = _flatten_layer(layer)
                flattened_layers.append(flattened_layer)
                num_features += n

            layer_flat = tf.concat(1, flattened_layers)

            layer = _new_fc_layer(
                input_layer=layer_flat,
                num_inputs=num_features,
                num_outputs=layer_params['num_outputs'],
                use_relu=layer_params['use_relu'],
                dropout=layer_params['dropout'],
                keep_prob=self.keep_prob
            )

        elif input_type == "fully_connected" and output_type == "fully_connected":
            layer = _new_fc_layer(
                input_layer=prev_layer,
                num_inputs=prev_layer_params['num_outputs'],
                num_outputs=layer_params['num_outputs'],
                use_relu=layer_params['use_relu'],
                dropout=layer_params['dropout'],
                keep_prob=self.keep_prob
            )

        elif input_type == "fully_connected" and output_type == "prediction_layer":
            layer = _new_fc_layer(
                input_layer=prev_layer,
                num_inputs=prev_layer_params['num_outputs'],
                num_outputs=self.num_classes,
                use_relu=False
            )

        else:
            msg = "Transition from {} to {} not implemented"
            raise Error(msg.format(input_type, output_type))

        return layer

    def add_layers(self, prev_layer, prev_layer_params, layers):
        """Create a series of layers

        Args:
            prev_layer (tensor): previous layer
            prev_layer_params (dict): parameters of previous layer
            layers (list): list of parameters of layers to create

        Returns:
            tensor: last layer of sequence
        """
        for layer_params in layers:
            prev_layer = self.add_layer(
                prev_layer,
                prev_layer_params,
                layer_params
            )
            prev_layer_params = layer_params
        return prev_layer


def _new_weights(shape):
    """Create new weight variables

    Args:
        shape (tuple): shape of weights

    Returns:
        tensor: weight variables
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def _new_biases(length):
    """Create new bias variables

    Args:
        length (int): number of biases to create

    Returns:
        tensor: bias variables
    """
    return tf.Variable(tf.constant(0.05, shape=[length]))


def _new_conv_layer(
    input_layer, num_input_channels,
    filter_size, num_filters,
    use_pooling=True, normalize=False):
    """Create convolutional layer

    Args:
        input_layer (tensor): input layer
        num_input_channels (int): number of input channels
        filter_size (int): size of filter window
        num_filters (int): number of filters
        use_pooling (bool, optional): whether to use maxpooling
        normalize (bool, optional): whether to normalize the output

    Returns:
        tensor: created layer
    """
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = _new_weights(shape=shape)
    biases = _new_biases(length=num_filters)

    layer = tf.nn.conv2d(
        input=input_layer,
        filter=weights,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    layer += biases

    layer = tf.nn.relu(layer)

    if use_pooling:
        layer = tf.nn.max_pool(
            value=layer,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

    if normalize:
        layer = tf.nn.local_response_normalization(
            layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        )

    return layer


def _flatten_layer(layer):
    """Create a flatened layer from a multidimensional tensor

    Args:
        layer (tensor): layer to flatten

    Returns:
        (tensor, int): flattened layer, number of nodes
    """
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def _new_fc_layer(
    input_layer, num_inputs,
    num_outputs, use_relu=True,
    dropout=False, keep_prob=1):
    """Create fully  connected layer

    Args:
        input_layer (tensor): input layer
        num_inputs (int): number of input nodes
        num_outputs (int): number of output nodes
        use_relu (bool, optional): whether to linear rectification
        dropout (bool, optional): whether to use dropout
        keep_prob (tensor, optional): keeb probability for dropout

    Returns:
        tensor: created layer
    """
    weights = _new_weights(shape=[num_inputs, num_outputs])
    biases = _new_biases(length=num_outputs)
    layer = tf.matmul(input_layer, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    if dropout:
        layer = tf.nn.dropout(layer, keep_prob)

    return layer
