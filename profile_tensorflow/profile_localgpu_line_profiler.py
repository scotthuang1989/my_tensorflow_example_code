'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server.

Change the hardcoded host urls below with your own hosts.
Run like this:

pc-01$ python example.py --job_name="ps" --task_index=0
pc-02$ python example.py --job_name="worker" --task_index=0
pc-03$ python example.py --job_name="worker" --task_index=1
pc-04$ python example.py --job_name="worker" --task_index=2

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
from tensorflow.python.client import timeline
import argparse

FLAGS = None

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# config
NUM_CHANNELS = 1
NUM_LABELS = 10
IMAGE_SIZE = 28
learning_rate = 0.0005
training_epochs = 10000
frequency = 300

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EVALUATION = 5
INITIAL_LEARNING_RATE = 0.1


def _variable_on_cpu(name, shape, initializer):
	with tf.device("/cpu:0"):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name,shape, initializer=initializer,dtype=dtype)
	return var

def _variable_with_weight_decay(name,shape,stddev, wd):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(name,shape,
			tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
		tf.add_to_collection('losses',weight_decay)
	return var

def inference(images):
	# conv1
	with tf.variable_scope("conv1") as scope:
		conv1_weights = _variable_with_weight_decay('weights',
			shape=[5,5,NUM_CHANNELS,64],stddev=5e-2,
			wd=0)
		conv1_biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))

		conv = tf.nn.conv2d(images,conv1_weights,[1,1,1,1],padding="SAME")
		pre_activation = tf.nn.bias_add(conv,conv1_biases)
		conv1 = tf.nn.relu(pre_activation,name=scope.name)
		#summary??
	# pool1
	pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],
						strides=[1,2,2,1],
						padding='SAME')

	# conv2d
	with tf.variable_scope('conv2') as scope:
		conv2_weights = _variable_with_weight_decay('weights',
						shape=[5,5,64,64],
						stddev=5e-2,wd=0)
		conv2_biases = _variable_on_cpu('biases',shape=[64],initializer=tf.constant_initializer(0.0))
		pre_activation = tf.nn.conv2d(pool1,conv2_weights,[1,1,1,1],padding='SAME')
		conv2 = tf.nn.relu(pre_activation,name=scope.name)
		#summary??
	# pool2
	pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	with tf.variable_scope('local3') as scope:
		reshape = tf.reshape(pool2,[FLAGS.batch_size,-1])
		dim = reshape.get_shape()[1].value
		local3_weights = _variable_with_weight_decay('weights',shape=[dim,384],
							stddev=0.04,wd=0.004)
		local3_biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape,local3_weights)+local3_biases,name=scope.name)
		#summary??

	with tf.variable_scope('local4') as scope:
		local4_weights = _variable_with_weight_decay('weights',shape=[384,192],
						stddev=0.04,wd=0.004)
		local4_biases = _variable_on_cpu('biases',shape=[192],initializer=tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3,local4_weights)+local4_biases,name=scope.name)
		#summary??

	with tf.variable_scope('softmax_linear') as scope:
		sm_weights = _variable_with_weight_decay('weights',[192,NUM_LABELS],
								stddev=1/192,wd=0)
		sm_biases = _variable_on_cpu('biases',[NUM_LABELS],tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4,sm_weights),sm_biases, name=scope.name)
		#summary??
	return softmax_linear

def model_loss(images,labels):
	"""Calculate the total loss on a single tower running the CIFAR model.

	Args:
	scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
	images: Images. 4D tensor of shape [batch_size, height, width, channel_number].
	labels: Labels. 1D tensor of shape [batch_size].

	Returns:
	 Tensor of shape [] containing the total loss for a batch of data
	"""
	logits = inference(images)
	labels = tf.cast(labels, tf.int64)
	with tf.name_scope("cross_entropy"):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=labels,logits=logits,name='cross_entropy_per_example'
					)
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'),name='total_loss')

def eval(images,labels):
	"""
	logits:
	"""
	with tf.name_scope("evaluation"):
		labels = tf.cast(labels, tf.int64)
		logits = inference(images)
		return tf.nn.in_top_k(logits,labels,1)

# @profile
def train():
	print("test1")
	trace_file = open('timeline.ctf.json', 'w')
	with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
		global_step = tf.get_variable(
			'global_step',
			[],
			initializer = tf.constant_initializer(0),
			trainable = False)
		x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS], name='x-input')
		y = tf.placeholder(tf.float32, shape=[FLAGS.batch_size], name='y-input')
			# input images
		num_batches_per_epoch = int((NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size))
		writer = tf.summary.FileWriter(FLAGS.logs_path, graph=tf.get_default_graph())
		tf.summary.scalar('learning_rate', INITIAL_LEARNING_RATE)
		summary_op = tf.summary.merge_all()
		opt = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)

		sess = tf.Session(graph=graph, config=tf.ConfigProto(
  							allow_soft_placement=True))
		with tf.device('/gpu:0'):
			with tf.name_scope("tower0"):
				total_loss = model_loss(x,y)
				grads = opt.compute_gradients(total_loss)
				# train_op = opt.minimize(total_loss,global_step=global_step)
		train_op = opt.apply_gradients(grads)
		count=0
		sess.run(tf.global_variables_initializer())
		for epoch in range(FLAGS.num_epoch):
			for i in range(num_batches_per_epoch):
				count +=1
				begin_time = time.time()
				batch_x,batch_y = mnist.train.next_batch(FLAGS.batch_size)
				run_metadata = tf.RunMetadata()
				_,loss,summary,step = sess.run([train_op, total_loss,summary_op,global_step],
						feed_dict={x:batch_x.reshape([FLAGS.batch_size,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS]),y:batch_y},
						options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
						run_metadata=run_metadata)
				writer.add_summary(summary,step)
				if count%frequency == 0:
					elapsed_time = time.time()-begin_time

					print("Step: %d," % (step+1),
								" Epoch: %2d," % (epoch+1),
								" Batch: %3d of %3d," % (i+1, num_batches_per_epoch),
								" Loss: %.4f," % loss,
								# "accuracy on test:%0.4f," % (np.sum(all_eval_result)/(NUM_EVALUATION*FLAGS.batch_size)),
								" AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
					count = 0
			writer.add_graph(sess.graph)
	trace = timeline.Timeline(step_stats=run_metadata.step_stats)
	trace_file.write(trace.generate_chrome_trace_format())
	trace_file.close()
def main(argv=None):
	print("test0")
	train()

# if __name__ == '__main__':
# 	tf.app.run()
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',default= 256,type=int,
                            help="""Number of images to process in a batch.""")
parser.add_argument('--data_dir', default='/tmp/mnist_distributed',type=str,
                           help="""Path to the CIFAR-10 data directory.""")
parser.add_argument('--use_fp16', default=False,type=bool,
                            help="""Train the model using fp16.""")
parser.add_argument('--use_gpu',default=False,type=bool,
							help="""wether train with gpu""")
parser.add_argument('--num_epoch', default=5,type=int,
                            help="""Number of batches to run.""")
parser.add_argument('--logs_path', default="/tmp/mnist/1", type=str,
					help='default location for logs')

FLAGS, unparsed = parser.parse_known_args()
main()
