from types import MappingProxyType
import tensorflow as tf
import data_io
import data_process
import numpy as np
import os
import sys
import random

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

gg=tf.Graph()
gl=tf.Graph()

"""
GLOBAL
"""   

train_X, train_y, max_len0 = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.train.positives.fa", nega = "data\\ALKBH5_Baltz2012.train.negatives.fa", channel = 1 )
test_X_g, test_y_g, max_len_g = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.ls.positives.fa", nega = "data\\ALKBH5_Baltz2012.ls.negatives.fa", channel = 1)

if max_len0 > max_len_g:
    test_X_g, test_y_g, max_len_g = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.ls.positives.fa", nega = "data\\ALKBH5_Baltz2012.ls.negatives.fa", channel = 1, max_len = max_len0)
    max_len_g = max_len0

max_len_g += 6

with gg.as_default():

    x_g = tf.placeholder(tf.float32, shape = [None, max_len_g, 4])
    y_g = tf.placeholder(tf.float32, shape = [None, 2])
    keep_prob_g = tf.placeholder(tf.float32)

    W_conv1_g = wv([10, 4, 1, 8])
    b_conv1_g = bv([8])
    W_conv2_g = wv([12, 1, 8, 16])
    b_conv2_g = bv([16])
    W_conv3_g = wv([38, 4, 16, 32])
    b_conv3_g = bv([32])

    x_seq_g = tf.reshape(x_g, [-1, max_len_g, 4, 1])

    h_conv1_g = tf.nn.relu(conv2d(x_seq_g, W_conv1_g) + b_conv1_g)
    #h_pool1 = tf.reduce_max(h_conv1, [1, 2])
    h_pool1_g = max_pool_2x2(h_conv1_g)

    h_conv2_g = tf.nn.relu(conv2d(h_pool1_g, W_conv2_g) + b_conv2_g)
    #h_pool2 = tf.reduce_max(h_conv2, [1, 2])
    h_pool2_g = max_pool_2x2(h_conv2_g)

    #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    #h_pool3 = tf.reduce_max(h_conv3, [1, 2])
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool_concat = tf.concat([tf.reshape(h_pool1, [-1, filter_no]), tf.reshape(h_pool2, [-1, filter_no]), tf.reshape(h_pool3, [-1, filter_no])], 1)
    h_pool_g = tf.reshape(h_pool2_g, [-1, 4 * max_len_g * 16])
    h_drop0_g = tf.nn.dropout(h_pool_g, keep_prob_g)

    W_fc1_g = wv([4 * max_len_g * 16, 512])
    b_fc1_g = bv([512])
    h_fc1_g = tf.nn.relu(tf.matmul(h_drop0_g, W_fc1_g) + b_fc1_g)

    h_drop1_g = tf.nn.dropout(h_fc1_g, keep_prob_g)

    W_fc2_g = wv([512, 2])
    b_fc2_g = bv([2])

    y_conv_g = tf.matmul(h_drop1_g, W_fc2_g) + b_fc2_g

    correct_prediction = tf.equal(tf.argmax(y_conv_g, 1), tf.argmax(y_g, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

"""
LOCAL
"""
max_len_l = 107

with gl.as_default():

    x_l = tf.placeholder(tf.float32, shape = [None, 7, 4, max_len_l])
    y_l = tf.placeholder(tf.float32, shape = [None, 2])
    keep_prob_l = tf.placeholder(tf.float32)

    W_conv1_l = wv([10, 4, 7, 28])
    b_conv1_l = bv([28])
    W_conv2_l = wv([10, 1, 28, 112])
    b_conv2_l = bv([112])
    W_conv3_l = wv([38, 4, 112, 224])
    b_conv3_l = bv([32])

    x_seq_l = tf.reshape(x_l, [-1, max_len_l, 4, 7])

    h_conv1_l = tf.nn.relu(conv2d(x_seq_l, W_conv1_l) + b_conv1_l)
    #h_pool1 = tf.reduce_max(h_conv1, [1, 2])
    h_pool1_l = max_pool_2x2(h_conv1_l)

    h_conv2_l = tf.nn.relu(conv2d(h_pool1_l, W_conv2_l) + b_conv2_l)
    #h_pool2 = tf.reduce_max(h_conv2, [1, 2])
    h_pool2_l = max_pool_2x2(h_conv2_l)

    #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    #h_pool3 = tf.reduce_max(h_conv3, [1, 2])
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool_concat = tf.concat([tf.reshape(h_pool1, [-1, filter_no]), tf.reshape(h_pool2, [-1, filter_no]), tf.reshape(h_pool3, [-1, filter_no])], 1)
    h_pool_l = tf.reshape(h_pool2_l, [-1, 4 * max_len_l * 112])
    h_drop0_l = tf.nn.dropout(h_pool_l, keep_prob_l)

    W_fc1_l = wv([4 * max_len_l * 112, 256])
    b_fc1_l = bv([256])
    h_fc1_l = tf.nn.relu(tf.matmul(h_drop0_l, W_fc1_l) + b_fc1_l)

    h_drop1_l = tf.nn.dropout(h_fc1_l, keep_prob_l)

    W_fc2_l = wv([256, 2])
    b_fc2_l = bv([2])

    y_conv_l = tf.matmul(h_drop1_l, W_fc2_l) + b_fc2_l

    correct_prediction_l = tf.equal(tf.argmax(y_conv_l, 1), tf.argmax(y_l, 1))
    accuracy_l = tf.reduce_mean(tf.cast(correct_prediction_l, tf.float32))

    

with tf.Session(graph=gg) as sess:
    saver_g = tf.train.Saver()
    ckpt_g = tf.train.get_checkpoint_state('./gmodel/')
    if ckpt_g and ckpt_g.model_checkpoint_path:
        saver_g.restore(sess, ckpt_g.model_checkpoint_path)
    #sess.run(tf.initialize_all_variables())
    print(sess.run(accuracy, feed_dict = {x_g: test_X_g, y_g: test_y_g, keep_prob_g: 1.0}))
    out1 = sess.run(y_conv_g, feed_dict = {x_g: test_X_g, y_g: test_y_g, keep_prob_g: 1.0})

    
with tf.Session(graph=gl) as sess:
    saver_l = tf.train.Saver()
    ckpt_l = tf.train.get_checkpoint_state('./lmodel/')
    if ckpt_l and ckpt_l.model_checkpoint_path:
        saver_l.restore(sess, ckpt_l.model_checkpoint_path)
    #sess.run(tf.initialize_all_variables())
    test_X_l, test_y_l, temp = data_process.get_data(posi = "data\\ALKBH5_Baltz2012.ls.positives.fa", nega = "data\\ALKBH5_Baltz2012.ls.negatives.fa", channel = 7, window_size = 101)
    print(sess.run(accuracy_l, feed_dict = {x_l: test_X_l, y_l: test_y_l, keep_prob_l: 1.0}))
    out2 = sess.run(y_conv_l, feed_dict = {x_l: test_X_l, y_l: test_y_l, keep_prob_l: 1.0})
    
out = (out1 + out2) / 2
print(np.mean(np.equal(np.argmax(out1, 1), np.argmax(test_y_g, 1))))
print(np.mean(np.equal(np.argmax(out2, 1), np.argmax(test_y_g, 1))))
print(np.mean(np.equal(np.argmax(out, 1), np.argmax(test_y_g, 1))))

#correct_prediction = tf.equal(tf.argmax(y_conv_l + y_conv_g, 1), tf.argmax(y_l, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print("test accuracy %g" % accuracy.eval(feed_dict = {x_g: test_X_g, y_g: test_y_g, keep_prob_g: 1.0, x_l: test_X_l, y_l: test_y_l, keep_prob_l: 1.0}))

#results = sess_g.run(y_conv_g, feed_dict = { x_g: test_X_g, y_g: test_y_g, keep_prob_g: 1.0 })
#results = sess_l.run(y_conv_l, feed_dict = { x_l: test_X_l, y_l: test_y_l, keep_prob_l: 1.0 })
