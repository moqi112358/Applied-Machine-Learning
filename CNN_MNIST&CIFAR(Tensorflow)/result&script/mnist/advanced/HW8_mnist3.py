from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import math
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

	def feed_dict(train,LR):
		if train or FLAGS.FakeData:
			xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.FakeData)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k, lr: LR}
	
	# Input placeholders
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	lr = tf.placeholder(tf.float32)
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	sess.run(tf.initialize_all_variables())
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	x_image = tf.reshape(x, [-1,28,28,1])
	keep_prob = tf.placeholder(tf.float32)
	
	# First convolutional layer - maps one grayscale image to 32 feature maps.
	W1 = tf.Variable(tf.truncated_normal([6, 6, 1, 6], stddev=0.1))
	B1 = tf.Variable(tf.constant(0.1, tf.float32, [6]))
	
	W2 = tf.Variable(tf.truncated_normal([6, 6, 6, 12], stddev=0.1))
	B2 = tf.Variable(tf.constant(0.1, tf.float32, [12]))
	
	W3 = tf.Variable(tf.truncated_normal([4, 4, 12, 24], stddev=0.1))
	B3 = tf.Variable(tf.constant(0.1, tf.float32, [24]))

	W4 = tf.Variable(tf.truncated_normal([7 * 7 * 24, 200], stddev=0.1))
	B4 = tf.Variable(tf.constant(0.1, tf.float32, [200]))
	
	W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
	B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
	
	#stride = 1  # output is 28x28
	Y1 = tf.nn.relu(tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
	#stride = 2  # output is 14x14
	Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2)
	#stride = 2  # output is 7x7
	Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)
	YY = tf.reshape(Y3, shape=[-1, 7 * 7 * 24])
	
	Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
	YY4 = tf.nn.dropout(Y4, keep_prob)
	Ylogits = tf.matmul(YY4, W5) + B5
	y_conv = tf.nn.softmax(Ylogits)
	
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_)
	cross_entropy = tf.reduce_mean(cross_entropy)*100
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('cross entropy', cross_entropy)
	tf.summary.scalar('accuracy', accuracy)
	

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	tf.global_variables_initializer().run()
	tr_acc_list = []
	train_d = np.zeros((784))
	test_d = np.zeros((10))
	for i in range(FLAGS.max_step):
		max_learning_rate = 0.003
		min_learning_rate = 0.0001
		decay_speed = 2000.0
		learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
		
		if i % 100 == 0:
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False,learning_rate))
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
			fd_train = feed_dict(True,learning_rate)
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
	flags.DEFINE_integer('max_step', 100001, 'Number of steps to run trainer.')
	flags.DEFINE_float('lr', 0.0001, 'Initial learning rate.')
	flags.DEFINE_float('dropout', 0.75, 'Keep probability for training dropout.')
	flags.DEFINE_string('data_dir', '/home/huaminz2/cs498/hw8/mnist/advanced/input_data', 'Directory for storing data')
	flags.DEFINE_string('log_dir', '/home/huaminz2/cs498/hw8/mnist/advanced/mnist_with_summaries', 'Summaries log directory')
	#parser.add_argument('--fake_data', nargs='?', const=True, type=bool,default=False,help='If true, uses fake data for unit testing.')
	#parser.add_argument('--max_steps', type=int, default=1000,help='Number of steps to run trainer.')
	#parser.add_argument('--learning_rate', type=float, default=0.001,help='Initial learning rate')
	#parser.add_argument('--dropout',type=float, default=0.9,help='Keep probability for training dropout.')
	#parser.add_argument('--data_dir',type=str,default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
	#parser.add_argument('--log_dir',type=str,default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',help='Summaries log directory')
	#FLAGS, unparsed = parser.parse_known_args()
	tf.app.run()
			
	
 
