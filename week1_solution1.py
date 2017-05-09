import tensorflow as tf

a = tf.Variable(0.0, name='a')
b = tf.Variable(0.0, name='b')

f = tf.add((a + 2.0 * b - 7.0)**2, (2.0 * a + b - 5.0)**2, name='f')

session = tf.Session()
writer = tf.summary.FileWriter('./graphs', session.graph)

init = tf.global_variables_initializer()
session.run(init)

writer.close()
