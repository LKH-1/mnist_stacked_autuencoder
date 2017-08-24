import os

import tensorflow as tf
import numpy as np
import scipy
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

os.mkdir('logs/embeddings')
logs_dir = 'logs/embeddings'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
test_labels = mnist.test.labels
test_label_ints = np.argmax(test_labels,axis=1)
test_images = mnist.test.images.reshape((10000,28,28))
test_label_ints = test_label_ints.astype(np.uint8)

with tf.Session() as sess:
    #the meta file contains all of the information to create the model graph
    saver = tf.train.import_meta_graph('logs/model.ckpt-10000.meta')
    #now restore the state of all vars at the last checkpoint
    saver.restore(sess, './logs/model.ckpt-10000')
    graph = tf.get_default_graph()
    predicted_class = graph.get_tensor_by_name('predicted_class/dense/Softmax:0')
    x = graph.get_tensor_by_name('x_placeholder:0')
    y_labels = graph.get_tensor_by_name('y_labels_placeholder:0')

    predictions = predicted_class.eval(feed_dict={x:mnist.test.images, y_labels:test_labels},session=sess)

# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

sprite = images_to_sprite(test_images)
scipy.misc.imsave(os.path.join(logs_dir, 'sprite.png'), sprite)

tf.reset_default_graph()
with tf.Session() as sess:
    embedding_var = tf.Variable(predictions,  name='prediction_embedding')
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(logs_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(logs_dir,'metadata.tsv')
    embedding.sprite.image_path = os.path.join(logs_dir,'sprite.png')
    image_shape = test_images[0].shape
    embedding.sprite.single_image_dim.extend([image_shape[0], image_shape[1]])
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess,os.path.join(logs_dir,'model_embeddings.ckpt'), 1)

with open(os.path.join(logs_dir, 'metadata.tsv'), 'w') as metadata_file:
    metadata_file.write('image\tclass\n')
    for i in range(len(test_label_ints)):
        metadata_file.write('{}\t{}\n'.format(i, test_label_ints[i]))
