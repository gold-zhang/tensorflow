# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:35:22 2018

@author: yao
"""
import tensorflow as tf
import numpy as np

x = np.linspace(-1,1,100)  
b = 0.5
y = 3 * x + b


xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

Weights = tf.Variable(tf.random_normal([1]))
biases = tf.Variable(tf.zeros([1]) + 0.1)

pre = tf.multiply(Weights,xs) + biases

loss = tf.reduce_mean(tf.square(pre - ys))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(50):
    sess.run(optimizer,feed_dict={xs:x,ys:y})
#    if i % 5 == 0:
#        print(sess.run(Weights),sess.run(biases))
        
print(sess.run(pre,feed_dict={xs:0.5}))