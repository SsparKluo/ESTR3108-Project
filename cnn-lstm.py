import tensorflow as tf
import data_io
import data_process
import numpy as np
import os
import sys
import random
from tensorflow.contrib import rnn

max_len = 494
filter_no = 30

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

    index = index_neg[:int(size * 0.45)] + index_pos[:int(size * 0.55)]
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

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 1, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')


train_X, train_y, max_len0 = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.train.positives.fa", nega = "data\\ALKBH5_Baltz2012.train.negatives.fa", channel = 1, max_len = max_len)
test_X, test_y, max_len1 = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.ls.positives.fa", nega = "data\\ALKBH5_Baltz2012.ls.negatives.fa", channel = 1, max_len = max_len)

max_len = 500

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, max_len, 4])
y = tf.placeholder(tf.float32, shape = [None, 2])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = wv([10, 4, 1, 8])
b_conv1 = bv([8])
W_conv2 = wv([12, 1, 8, 16])
b_conv2 = bv([16])
W_conv3 = wv([38, 4, 16, 32])
b_conv3 = bv([32])

x_seq = tf.reshape(x, [-1, max_len, 4, 1])

h_conv1 = tf.nn.relu(conv2d(x_seq, W_conv1) + b_conv1)
#h_pool1 = tf.reduce_max(h_conv1, [1, 2])
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = tf.reduce_max(h_conv2, [1, 2])
h_pool2 = max_pool_2x2(h_conv2)

#h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3 = tf.reduce_max(h_conv3, [1, 2])
#h_pool3 = max_pool_2x2(h_conv3)

hidden_size = 200
layer_num = 2

lstm_input = tf.reshape(h_pool2, [-1, 200, 200])

lstm_cell = rnn.BasicLSTMCell(num_units=200, forget_bias=1.0, state_is_tuple=True)
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

init_state = mlstm_cell.zero_state(200, dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=lstm_input, initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]

#h_pool_concat = tf.concat([tf.reshape(h_pool1, [-1, filter_no]), tf.reshape(h_pool2, [-1, filter_no]), tf.reshape(h_pool3, [-1, filter_no])], 1)
h_pool = tf.reshape(h_state, [-1, 20 * 200])
h_drop0 = tf.nn.dropout(h_pool, keep_prob)

W_fc1 = wv([20 * 200, 512])
b_fc1 = bv([512])
h_fc1 = tf.nn.relu(tf.matmul(h_drop0, W_fc1) + b_fc1)

h_drop1 = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = wv([512, 2])
b_fc2 = bv([2])

y_conv = tf.matmul(h_drop1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


for i in range(1200):
	batch_seq, batch_label = next_batch(256, train_X, train_y)
	# print(np.array(batch_seq).shape)
	# print(np.array(batch_label).shape)
	# print(batch_label)
	if i % 10 == 0:
		train_accuracy = accuracy.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		#y_out = y_conv.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		# h_pool1_out = h_pool1.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		# h_p_c = h_pool_concat.eval(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 1.0})
		# print(y_out)
		# print(h_pool1_out)
		# print(h_p_c)
		print("step %d, training accuracy %g" % (i, train_accuracy))
		print("test accuracy %g" % accuracy.eval(feed_dict = {x: test_X, y: test_y, keep_prob: 1.0}))
	train_step.run(feed_dict = {x: batch_seq, y: batch_label, keep_prob: 0.75})

saver = tf.train.Saver()
saver.save(sess, './model/my_model', global_step = 3)


"""
print(np.array(train_X).shape)

oh_label = []
for i, one_label in enumerate(train_y):
	if (one_label):
		oh_label.append([0, 1])
	else:
		oh_label.append([1, 0])

y_out_final = y_conv.eval(feed_dict = {x: train_X, y: oh_label, keep_prob: 1.0})
correct_prediction_out = correct_prediction.eval(feed_dict = {x: train_X, y: oh_label, keep_prob: 1.0})
# print(label)
# print(oh_label)
# print(y_out_final)
# print(correct_prediction_out)
print("test accuracy %g" % accuracy.eval(feed_dict = {x: test_X, y: test_y, keep_prob: 1.0}))
"""