import tensorflow as tf
import data_io
import data_process
import numpy as np
import os
import sys
import random

max_len = 376
filter_no = 10

seq, label = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.ls.positives.fa", nega = "data\\ALKBH5_Baltz2012.ls.negatives.fa", channel = 1, window_size = max_len)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, max_len, 4])
y = tf.placeholder(tf.float32, shape = [None])

def wv(shape):
	init = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(init)

def bv(shape):
	init = tf.constant(0.1, shape = shape)
	return tf.Variable(init)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

W_conv1 = wv([8, 4, 1, filter_no])
b_conv1 = bv([filter_no])
W_conv2 = wv([20, 4, 1, filter_no])
b_conv2 = bv([filter_no])
W_conv3 = wv([38, 4, 1, filter_no])
b_conv3 = bv([filter_no])

x_seq = tf.reshape(x, [-1, max_len, 4, 1])

h_conv1 = tf.nn.relu(conv2d(x_seq, W_conv1) + b_conv1)
h_pool1 = tf.reduce_max(h_conv1, [1, 2])

h_conv2 = tf.nn.relu(conv2d(x_seq, W_conv2) + b_conv2)
h_pool2 = tf.reduce_max(h_conv2, [1, 2])

h_conv3 = tf.nn.relu(conv2d(x_seq, W_conv3) + b_conv3)
h_pool3 = tf.reduce_max(h_conv3, [1, 2])

h_pool_concat = tf.concat([tf.reshape(h_pool1, [-1, filter_no]), tf.reshape(h_pool2, [-1, filter_no]), tf.reshape(h_pool3, [-1, filter_no])], 1)

keep_prob = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h_pool_concat, keep_prob)

W_fc = wv([filter_no * 3, 1])
b_fc = bv([1])

y_conv_high = tf.matmul(h_pool_concat, W_fc) + b_fc
y_conv = tf.reduce_max(y_conv_high, [1])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.cast(tf.round(y_conv), tf.int32), tf.cast(tf.round(y), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

No = []

for i in range(200):
	No.append(i)

for i in range(120):
	random.shuffle(No)
	batch_seq = []
	batch_label = []
	for j in range(50):
		batch_seq.append(seq[No[j]])
		batch_label.append(label[No[j]])
	# print(np.array(batch_seq).shape)
	# print(np.array(batch_label).shape)
	# print(batch_label)
	train_accuracy = accuracy.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
	print("step %d, training accuracy %g" % (i, train_accuracy))
	train_step.run(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict = {x: seq, y: label, keep_prob: 1.0}))