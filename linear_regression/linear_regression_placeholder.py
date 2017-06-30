import tensorflow as tf

x_data = [1.,2.,3.,4]
y_data = [2.,4.,6.,8]

w = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = w * x + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
	sess.run(train, feed_dict={x:x_data, y:y_data})
	print (step, sess.run(cost, feed_dict={x: x_data, y:y_data}), sess.run(w), sess.run(b))


print (sess.run(hypothesis, feed_dict={x:5}))


