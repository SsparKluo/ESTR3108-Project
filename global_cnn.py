import tensorflow as tf
import data_io
import data_process
import numpy as np
import os
import sys
import random

max_len = 376
filter_no = 30

seq, label = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.ls.positives.fa", nega = "data\\ALKBH5_Baltz2012.ls.negatives.fa", channel = 1, window_size = max_len)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, max_len, 4])
y = tf.placeholder(tf.float32, shape = [None, 2])

def wv(shape):
	init = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(init)

def bv(shape):
	init = tf.constant(0.1, shape = shape)
	return tf.Variable(init)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

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


W_fc1 = wv([filter_no * 3, filter_no])
b_fc1 = bv([filter_no])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_concat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = wv([filter_no, 2])
b_fc2 = bv([2])

y_conv = tf.matmul(h_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


for i in range(500):
	No = []
	for j in range (216):
		No.append(j)
	random.shuffle(No)
	# print(No)
	batch_seq = []
	batch_label = []
	for j in range(50):
		batch_seq.append(seq[No[j]])
		if (label[No[j]]):
			batch_label.append([0, 1])
		else:
			batch_label.append([1, 0])
	# print(np.array(batch_seq).shape)
	# print(np.array(batch_label).shape)
	# print(batch_label)
	train_accuracy = accuracy.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
	y_out = y_conv.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
	# h_pool1_out = h_pool1.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
	# h_p_c = h_pool_concat.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
	# print(y_out)
	# print(h_pool1_out)
	# print(h_p_c)
	print("step %d, training accuracy %g" % (i, train_accuracy))
	train_step.run(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 0.5})

print(np.array(seq).shape)

oh_label = []
for i, one_label in enumerate(label):
	if (one_label):
		oh_label.append([0, 1])
	else:
		oh_label.append([1, 0])

y_out_final = y_conv.eval(feed_dict = {x: seq, y: oh_label, keep_prob: 1.0})
correct_prediction_out = correct_prediction.eval(feed_dict = {x: seq, y: oh_label, keep_prob: 1.0})
# print(label)
# print(oh_label)
# print(y_out_final)
# print(correct_prediction_out)
print("test accuracy %g" % accuracy.eval(feed_dict = {x: seq, y: oh_label, keep_prob: 1.0}))