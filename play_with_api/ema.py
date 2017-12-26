import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(tf.constant(0))

ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_average = ema.apply([v1])

with tf.Session().as_default() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # check intial value
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1,5))
    sess.run(maintain_average)

    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 1000))
    sess.run(maintain_average)

    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_average)

    print(sess.run([v1, ema.average(v1)]))
