import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = w * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))
learning_rate = tf.Variable(0.1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
	sess.run(train)
	print ('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(w), sess.run(b)))

