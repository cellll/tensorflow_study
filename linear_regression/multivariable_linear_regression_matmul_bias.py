import tensorflow as tf

x_data = [[1,1,1,1,1], [1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]
y_data = [1,2,3,4,5]

w = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

hypothesis = tf.matmul(w, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))
learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2001):
	sess.run(train)
	print (step, sess.run(cost), sess.run(w))

sess.close()
