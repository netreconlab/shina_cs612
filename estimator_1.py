import tensorflow as tf
import numpy as np

temp = tf.feature_column.numeric_column("x", shape=[1], dtype=tf.float32)
feature_column = [temp]

print(tf.train.get_global_step())

#print(x)
#print(feature_column)

#data_Sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

estimator = tf.estimator.LinearRegressor(feature_columns=feature_column)


input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y = y_train, batch_size=4, shuffle= True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y = y_train, num_epochs= 1000, batch_size=4, shuffle= False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y = y_eval, num_epochs= 1000, batch_size=4, shuffle= False)


estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
evaluation_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("evaluation metrics: %r"% evaluation_metrics)



