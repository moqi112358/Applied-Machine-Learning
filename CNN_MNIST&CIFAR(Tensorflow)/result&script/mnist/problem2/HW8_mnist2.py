from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
FLAGS = None
  
def train():
	f = open('acc.txt', 'w')
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True,fake_data=FLAGS.FakeData)
	sess = tf.InteractiveSession()
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
			
	def conv2d(x, W, strides=[1, 1, 1, 1]):
		return tf.nn.conv2d(x, W, strides, padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

	def feed_dict(train):
		if train or FLAGS.FakeData:
			xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.FakeData)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k}
	
	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

	with tf.name_scope('input_reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])
		tf.summary.image('input', x_image, 10)

	# First convolutional layer - maps one grayscale image to 32 feature maps.
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	# Pooling layer - downsamples by 2X.
	h_pool1 = max_pool_2x2(h_conv1)
	
	# Second convolutional layer -- maps 32 feature maps to 64.
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	# Second pooling layer.
	h_pool2 = max_pool_2x2(h_conv2)

	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	# Dropout - controls the complexity of the model, prevents co-adaptation of
	# features.
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# Map the 1024 features to 10 classes, one for each digit
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	tf.summary.scalar('cross entropy', cross_entropy)
	train_step = tf.train.AdamOptimizer(FLAGS.lambda_).minimize(cross_entropy)
    
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	tf.global_variables_initializer().run()

	train_d = np.zeros((784))
	test_d = np.zeros((10))
	tr_acc_list = []
	for i in range(FLAGS.max_step):
		if i % 100 == 0:
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			if i != 0:
				s,tr_acc = sess.run([merged,accuracy],feed_dict={x:train_d[1:,],y_:test_d[1:,],keep_prob: 1.0})
				f.write('Step: %s Train Accuracy: %s Test Accuracy: %s\n' % (i, tr_acc, acc))
				print('Step: %s Train Accuracy: %s Test Accuracy: %s' % (i, tr_acc, acc))
				tr_acc_list.append(acc)
			else:
				f.write('Step: %s Test Accuracy: %s\n' % (i, acc))
				print('Step: %s Test Accuracy: %s' % (i, acc))
			train_d = np.zeros((784))
			test_d = np.zeros((10))
		else: 
			fd_train = feed_dict(True)
			summary, _ = sess.run([merged, train_step], feed_dict=fd_train)
			train_writer.add_summary(summary, i)
			train_d = np.vstack((train_d,list(fd_train.values())[0]))
			test_d = np.vstack((test_d,list(fd_train.values())[1]))
	f.write('\n Average test accuracy: %s' % (np.mean(tr_acc_list)))
	f.write('\n Max test accuracy: %s' % (np.amax(tr_acc_list)))
	f.close()

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()
  
if __name__ == '__main__':
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_boolean('FakeData', False, 'If true, uses fake data for unit testing.')
	flags.DEFINE_integer('max_step', 5001, 'Number of steps to run trainer.') ##
	flags.DEFINE_float('lambda_', 0.0001, 'Initial learning rate.')
	flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
	flags.DEFINE_string('data_dir', '/home/huaminz2/cs498/hw8/mnist/problem2/input_data', 'Directory for storing data')
	flags.DEFINE_string('log_dir', '/home/huaminz2/cs498/hw8/mnist/problem2/mnist_with_summaries', 'Summaries log directory')
	#parser.add_argument('--fake_data', nargs='?', const=True, type=bool,default=False,help='If true, uses fake data for unit testing.')
	#parser.add_argument('--max_steps', type=int, default=1000,help='Number of steps to run trainer.')
	#parser.add_argument('--learning_rate', type=float, default=0.001,help='Initial learning rate')
	#parser.add_argument('--dropout',type=float, default=0.9,help='Keep probability for training dropout.')
	#parser.add_argument('--data_dir',type=str,default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
	#parser.add_argument('--log_dir',type=str,default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',help='Summaries log directory')
	#FLAGS, unparsed = parser.parse_known_args()
	tf.app.run()
			
	
 
