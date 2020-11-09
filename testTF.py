import os
import tensorflow as tf

hello = tf.constant('hello')
se = tf.Session()
print(se.run(hello))