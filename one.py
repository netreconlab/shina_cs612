import tensorflow as tf

node1 = tf.constant(3., dtype=tf.float32)
node2 = tf.constant(4.)
node3 = tf.add(node1, node2)

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
adder = a + b
add_and_triple = adder * 3.

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x +b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

fixW = tf.assign(W,[-1.])
fixb = tf.assign(b, [1.])


print("a:",a)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
sess.run([fixW,fixb])
#print("sess.run:", sess.run(node3))
#print("sess.run:", sess.run(adder, {a: 3., b: 7.}))
#print("sess.run:", sess.run(add_and_triple, {a: 3., b: 7.}))
#print("sess.run:", sess.run(linear_model, {x: [1.,2.,3.,4.]}))
print("sess.run:", sess.run(loss, {x: [1.,2.,3.,4.], y: [0.,-1.,-2.,-3.]}))
