import tensorflow as tf

x = tf.Variable([0., 0.], name='x')
y = tf.constant([1.0, 1.0], name='y')
c = tf.placeholder(tf.float32, shape=[1])

g = (c * x - y)**2

optimizer = tf.train.GradientDescentOptimizer(0.1)
optimization_step = optimizer.minimize(g)

session = tf.Session()
session.run(tf.initialize_all_variables())

for i in range(100):
	x_, y_, g_, _ = session.run([x, y, g, optimization_step], feed_dict={c: [1.]})
	print('i = {}  x = {}  y = {}  g(x, y) = {}'.format(i, x_, y_, g_))
