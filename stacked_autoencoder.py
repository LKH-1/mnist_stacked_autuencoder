import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def auto_encoder_layer(inputs,hidden_size,layer_name):
    """Create all tensors necessary for training an autoencoder layer and return a dictionary of the relevant tensors."""
    with tf.variable_scope(layer_name): #using variable scope makes collecting trainable vars for the layer easy
        """Create two tensors.  One for the encoding layer and one for the output layer.
        The goal is to have the output layer match the inputs it recieves."""
        encoding_layer = tf.layers.dense(inputs,hidden_size,activation=tf.nn.relu,name='encoding_layer_{}'.format(hidden_size))
        output_layer = tf.layers.dense(encoding_layer,int(inputs.shape[1]), name='outputs')

        """Use the mean squared error of the difference between the inputs and the output layer to define the loss"""
        layer_loss = tf.losses.mean_squared_error(inputs, output_layer)


        all_vars = tf.trainable_variables() #this gets all trainable variables in the computational graph
        layer_vars = [v for v in all_vars if v.name.startswith(layer_name)] #select only the variables in this layer to train
        """create an op to minimize the MSE"""
        optimizer = tf.train.AdamOptimizer().minimize(layer_loss,var_list=layer_vars, name='{}_opt'.format(layer_name))

        """Create a summary op to monitor the loss of this layer"""
        loss_summ = tf.summary.scalar('{}_loss'.format(layer_name),layer_loss)
    return {'inputs':inputs, 'encoding_layer':encoding_layer, 'output_layer':output_layer, 'layer_loss':layer_loss, 'optimizer':optimizer}

def train_layer(output_layer, layer_loss,optimizer):
    """Train each encoding layer for 1000 steps"""
    layer_name = output_layer.name.split('/')[0]
    print('Pretraining {}'.format(layer_name))
    num_steps = 1000
    step=1
    while step <= num_steps:
        batch = mnist.train.next_batch(batch_size)
        _out_layer, _layer_loss, _ =  sess.run([output_layer, layer_loss, optimizer],feed_dict={x:batch[0],y_labels:batch[1]})
        #print(_layer_loss)
        step += 1
    print('layer finished')


#define the number of layers and batch size to use in the model
batch_size = 100
num_layers = 6

#download and import data from the MNIST data set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""Create placeholder tensors for the input images in the MNIST data set.
Each image in the MNIST data set is a hand written digit from 0-9.  The pictures are 28x28(784 total)
pixels where each pixel intensity has been normalized between 0 and 1.
The images will be processed in minibatches, so the shape for the input tensor is (None, 784).  There are 10 classes in this
dataset that are one-hot encoded."""
x = tf.placeholder(tf.float32, shape=[None, 784])
y_labels = tf.placeholder(tf.float32, shape=[None, 10])

#store each of the layers in a list so we can interate through and train them later
model_layers = []

"""For each encoding layer, make the number of hidden units half the number of input units.
This will force the encoding layer to learn a simpler representation of the data."""
hidden_size = x.shape[1].value/2
next_layer_inputs = x

#create all of the layers for the model
for layer in range(0, num_layers):
    layer_name = 'layer_{}'.format(layer)
    model_layers.append(auto_encoder_layer(next_layer_inputs, hidden_size, layer_name))
    """After training a layer we will use the encoding from that layer as inputs to the next.
    The outputs from that layer are no longer used."""
    next_layer_inputs = model_layers[-1]['encoding_layer']
    hidden_size = int(hidden_size/2)

"""""Now that all of the layers for the stacked auto encoder have been created add one final layer
for classification.  The output can take on one of 10 values, 0-9, and the labels are one-hot encoded, so
use a softmax layer for the prediction."""
last_layer = model_layers[-1]
outputs = last_layer['encoding_layer']
y = tf.layers.dense(outputs,10,activation=tf.nn.softmax)

#For the loss use cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y))

"""create a global step counter to keep track of epochs during training and add this to the
net_op below.  This will increment the step counter each time net_op is run."""
global_step = tf.train.get_or_create_global_step()
net_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy,global_step=global_step)

#create ops to check accuracy
correct_prediction = tf.equal(tf.argmax(y_labels, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#add a summary op for loging
accuracy_summ = tf.summary.scalar('train_accuracy',accuracy)

"""Use a MonitoredTrainingSession for running the computations.  It makes running on distributed systems
possible, handles checkpoints, saving summaries, and restoring from crashes easy."""

#create hooks to pass to the session.  These can be used for adding additional calculations, loggin, etc.
#This hook simply tells the session how many steps to run
hooks=[tf.train.StopAtStepHook(last_step=10000)]

#This command collects all summary ops that have been added to the graph and prepares them to run in the next session
tf.summary.merge_all()

logs_dir = 'logs'



with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=logs_dir,save_summaries_steps=100) as sess:

    start_time = time.time()

    """First train each layer one at a time, freezing weights from previous layers.
    This was accomplished by declaring which variables to update when each layer optimizer was defined."""
    for layer_dict in model_layers:
        output_layer = layer_dict['output_layer']
        layer_loss = layer_dict['layer_loss']
        optimizer = layer_dict['optimizer']
        train_layer(output_layer, layer_loss, optimizer)

    #Now train the whole network for classification allowing all weights to change.
    while not sess.should_stop():
        batch = mnist.train.next_batch(batch_size)
        _y, _cross_entropy, _net_op, _accuracy = sess.run([y, cross_entropy, net_op, accuracy], feed_dict={x:batch[0],y_labels:batch[1]})
        print(_accuracy)
print('Training complete\n')

#examine the final test set accuracy by loading the trained model, along with the last saved checkpoint
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('logs/model.ckpt-10000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./logs'))
    _accuracy_test = accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_labels:mnist.test.labels})
    print('test_set accuracy: {}'.format(_accuracy_test))

duration = (time.time() - start_time)/60
print("Run complete.  Total time was {} min".format(duration))
