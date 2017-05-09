import tensorflow as tf

a = tf.Variable(0.0, name='a')
b = tf.Variable(0.0, name='b')

f = tf.add((a + 2.0 * b - 7.0)**2, (2.0 * a + b - 5.0)**2, name='f')

optimizer = tf.train.GradientDescentOptimizer(0.1)
optimization_step = optimizer.minimize(f)

session = tf.Session()
writer = tf.summary.FileWriter('./graphs', session.graph)

init = tf.global_variables_initializer()
session.run(init)

for i in range(100):
	a_, b_, f_, _ = session.run([a, b, f, optimization_step])
	print('i = {}  a = {}  b = {}  f(a,b) = {}'.format(i, a_, b_, f_))

writer.close()
