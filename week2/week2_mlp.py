
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# In[ ]:

#  how to install ipynb kernel: https://ipython.readthedocs.io/en/latest/install/kernel_install.html

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


# In[ ]:

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)


# <font color='black'>Exercise 1: Optimization</font>
# 
# Instead of the RMSprop optimizer try to experiment with other ones.
# 
# 1. MomentumOptimizer: If gradient descent is navigating down a valley with steep sides, it tends to madly oscillate from one valley wall to the other without making much progress down the valley. This is because the largest gradients point up and down the valley walls whereas the gradient along the floor of the valley is quite small. Momentum Optimization attempts to remedy this by keeping track of the prior gradients and if they keep changing direction then damp them, and if the gradients stay in the same direction then reward them. This way the valley wall gradients get reduced and the valley floor gradient enhanced.
# 2. AdagradOptimizer: Adagrad is optimized to finding needles in haystacks and for dealing with large sparse matrices. It keeps track of the previous changes and will amplify the changes for weights that change infrequently and suppress the changes for weights that change frequently. 
# 3. AdadeltaOptimizer: Adadelta is an extension of Adagrad that only remembers a fixed size window of previous changes. This tends to make the algorithm less aggressive than pure Adagrad. 
# 4. AdamOptimizer: Adaptive Moment Estimation (Adam) keeps separate learning rates for each weight as well as an exponentially decaying average of previous gradients. This combines elements of Momentum and Adagrad together and is fairly memory efficient since it doesn’t keep a history of anything (just the rolling averages). It is reputed to work well for both sparse matrices and noisy data. 
# 

# In[ ]:

# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# optimizer = tf.train.AdadeltaOptimizer(starter_learning_rate).minimize(loss)
# optimizer = tf.train.AdagradOptimizer(starter_learning_rate).minimize(loss)    
# optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(loss)      
# optimizer = tf.train.MomentumOptimizer(starter_learning_rate, 0.001).minimize(loss) 
# optimizer = tf.train.FtrlOptimizer(starter_learning_rate).minimize(loss)    
# optimizer = tf.train.RMSPropOptimizer(starter_learning_rate).minimize(loss)   


# In[ ]:

# Launch the graph in a session
# based on https://github.com/nlintz/TensorFlow-Tutorials/blob/master/04_modern_net.ipynb
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))


# Exercise 2: Try to add some of the following normalizations (l2, batch normalization) and see whether it increases the performance of the network.

# In[ ]:

#tf.nn.l2_normalize  
#tf.nn.moments(x,axes=[0]) Calculate the mean and variance of x
#tf.nn.batch_normalization(x,mean,variance,offset,scale,variance_epsilon, name=None),
# mean and variance of the batch can be found by using moments function.


# Usually mean, variance are calculated based on the batch of data and offset and scale can be optimized. Using moments function mean and variance can be calculated from the batch.

# In[ ]:



