# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:49:21 2018

@author: yao
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import init
from numpy import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def compute_accuracy(v_xs, v_ys):
    global y_
    y_pre = sess.run(y_, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

data,label = init.loadData()

train_data,test_data,train_label,test_label = train_test_split(data,label,test_size = 0.2)

xs = tf.placeholder(tf.float32,[None,4])
ys = tf.placeholder(tf.float32,[None,3])

y_ = add_layer(xs,4,3, activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_),reduction_indices=[1])) 

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(50):
    sess.run(train_step,feed_dict={xs:train_data,ys:train_label})
    if i % 5 ==0:
         print(compute_accuracy(test_data, test_label))
        
pre_ = sess.run(y_,feed_dict={xs:test_data})

for i in range(30):
    pre_label = argmax(pre_[i])
    real_label = argmax(test_label[i])
    print("prediction is: %d,real is: %d"%(pre_label,real_label))