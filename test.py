
import tensorflow as tf
 
# 在系统默认计算图上创建张量和操作
a=tf.constant([1.0,2.0])
b=tf.constant([2.0,1.0])
result = a+b
 
# 定义两个计算图
g1=tf.Graph()
g2=tf.Graph()
 
# 在计算图g1中定义张量和操作
with g1.as_default():
    a = tf.constant([1.0, 1.0])
    b = tf.constant([1.0, 1.0])
    result1 = a + b
 
with g2.as_default():
    a = tf.constant([2.0, 2.0])
    b = tf.constant([2.0, 2.0])
    result2 = a + b
 
 
# 在g1计算图上创建会话
with tf.Session(graph=g1) as sess:
    out1 = sess.run(result1)
    print ("g1:", format(out1))
 
with tf.Session(graph=g2) as sess:
    out2 = sess.run(result2)
    print ("g2:", format(out2))
 
# 在默认计算图上创建会话
with tf.Session(graph=tf.get_default_graph()) as sess:
    out3 = sess.run(result)
    print ("g:", format(out3))
 
print(format(out1))
  # 返回计算图中操作的个数