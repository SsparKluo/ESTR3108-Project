import tensorflow as tf
import data_io
import data_process
import numpy as np
import os
import sys
import random

data_name = sys.argv[1]

window_size = 101
filter_no = 30
max_len = window_size + 6
channel = 7

def next_batch(size, X, y):
	index_neg = []
	index_pos = []
	for i in range(0, len(y)):
		if y[i][0] == 0:
			index_neg.append(i)
		else:
			index_pos.append(i)

	np.random.shuffle(index_pos)
	np.random.shuffle(index_neg)

	index = index_neg[:int(size * 0.6)] + index_pos[:int(size * 0.4)]
	np.random.shuffle(index)

	batch_X = np.array([X[i] for i in index])
	batch_y = np.array([y[i] for i in index])
	
	return batch_X, batch_y

def wv(shape):
	init = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(init)

def bv(shape):
	init = tf.constant(0.1, shape = shape)
	return tf.Variable(init)

def conv2dv(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 1, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')


train_X, train_y, max_len0 = data_process.get_data(posi = "data\\" + data_name + ".train.positives.fa", nega = "data\\" + data_name + ".train.negatives.fa", channel = 7, window_size = window_size)
test_X, test_y, max_len1 = data_process.get_data(posi = "data\\" + data_name + ".ls.positives.fa", nega = "data\\" + data_name + ".ls.negatives.fa", channel = 7, window_size = window_size)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 7, 4, max_len])
y = tf.placeholder(tf.float32, shape = [None, 2])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = wv([10, 4, 7, 28])
b_conv1 = bv([28])
W_conv2 = wv([12, 1, 28, 112])
b_conv2 = bv([112])
W_conv3 = wv([38, 1, 112, 224])
b_conv3 = bv([224])

# [-1, 104, 4, 7]
x_seq = tf.reshape(x, [-1, max_len, 4, channel])

# [-1, 104, 1, 28]
h_conv1 = tf.nn.relu(conv2d(x_seq, W_conv1) + b_conv1)
#h_pool1 = tf.reduce_max(h_conv1, [1, 2])
# [-1, 52, 1, 28]
h_pool1 = max_pool_2x2(h_conv1)

# [-1, 52, 1, 112]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = tf.reduce_max(h_conv2, [1, 2])
# [-1, 26, 1, 112]
h_pool2 = max_pool_2x2(h_conv2)

# [-1, 26, 1, 224]
#h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3 = tf.reduce_max(h_conv3, [1, 2])
# [-1, 13, 1, 224]
#h_pool3 = max_pool_2x2(h_conv3)

#h_pool_concat = tf.concat([tf.reshape(h_pool1, [-1, filter_no]), tf.reshape(h_pool2, [-1, filter_no]), tf.reshape(h_pool3, [-1, filter_no])], 1)
h_pool = tf.reshape(h_pool2, [-1, 4 * max_len * 112])
h_drop0 = tf.nn.dropout(h_pool, keep_prob)

W_fc1 = wv([4 * max_len * 112, 256])
b_fc1 = bv([256])
h_fc1 = tf.nn.relu(tf.matmul(h_drop0, W_fc1) + b_fc1)

h_drop1 = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = wv([256, 2])
b_fc2 = bv([2])

y_conv = tf.matmul(h_drop1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

print(train_y.size)

train_time = train_y.size // 4

stop_count = 0

for i in range(train_time):
	batch_seq, batch_label = next_batch(256, train_X, train_y)
	# print(np.array(batch_seq).shape)
	# print(np.array(batch_label).shape)
	# print(batch_label)
	if i % 10 == 0:
		train_accuracy = accuracy.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		if (train_accuracy > 0.999):
			stop_count = stop_count + 1
		#y_out = y_conv.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		# h_pool1_out = h_pool1.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		# h_p_c = h_pool_concat.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		# print(y_out)
		# print(h_pool1_out)
		# print(h_p_c)
		print("step %d / %d, training accuracy %g" % (i, train_time, train_accuracy))
		print("test accuracy %g" % accuracy.eval(feed_dict = {x: test_X, y: test_y, keep_prob: 1.0}))
	train_step.run(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 0.5})
	if (i % 500 == 0):
		saver = tf.train.Saver()
		saver.save(sess, './model_multi_' + data_name + '/my_model_multi', global_step = i // 100)
	if (stop_count > (train_time // 300)):
		saver = tf.train.Saver()
		saver.save(sess, './model_multi_' + data_name + '/my_model_multi', global_step = i // 100)
		break


print(np.array(train_X).shape)

'''
oh_label = []
for i, one_label in enumerate(train_y):
	if (one_label):
		oh_label.append([0, 1])
	else:
		oh_label.append([1, 0])
'''

y_out_final = y_conv.eval(feed_dict = {x: train_X, y: train_y, keep_prob: 1.0})
correct_prediction_out = correct_prediction.eval(feed_dict = {x: train_X, y: train_y, keep_prob: 1.0})
# print(label)
# print(oh_label)
# print(y_out_final)
# print(correct_prediction_out)
print("test accuracy %g" % accuracy.eval(feed_dict = {x: test_X, y: test_y, keep_prob: 1.0}))