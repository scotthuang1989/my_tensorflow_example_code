"""
use nightly image: pip install tf-nightly
"""

import tensorflow as tf
import time

def py_function():
    while True:
        time.sleep(1)
        yield [time.time()-1506000000]

for i in range(5):
    print(next(py_function()))


data = py_function()
import pdb; pdb.set_trace()
dataset = tf.contrib.data.Dataset.from_generator(lambda: data, tf.float64)

value = dataset.make_one_shot_iterator().get_next()

queue = tf.FIFOQueue(capacity=2, dtypes=[tf.float64], shapes=[1])

enqueue_op = queue.enqueue([value])

data_sample = queue.dequeue()

with tf.Session() as sess:

    sess.run(enqueue_op)

    for step in range(5):
        sess.run(enqueue_op)
        one_data = sess.run(data_sample)
        print(one_data)
