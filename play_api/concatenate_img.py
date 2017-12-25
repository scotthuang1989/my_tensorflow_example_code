"""
Profile np.concatenate and tf.concat.

Plan.
--------------------

read 1000 frame from video and concatenate them.
"""
import argparse
import time

import cv2
import tensorflow as tf
import numpy as np
from memory_profiler import profile

ap = argparse.ArgumentParser()

ap.add_argument('-v', '--video', type=str, help='video file', default='/home/scott/Videos/S11E07.mp4')


FLAGS = ap.parse_args()

TEST_NUM = 200
BATCH_SIZE = 4


def numpy_concate():
  """Use numpy concate images."""
  cap = cv2.VideoCapture(FLAGS.video)
  start = time.time()
  for i in range(TEST_NUM):
    concate_list = []
    for j in range(BATCH_SIZE):
       ret, frame = cap.read()
       frame = np.expand_dims(frame, 0)
       concate_list.append(frame)
    con_result = np.concatenate(concate_list)

  end = time.time()
  cap.release()
  return end - start


def numpy_concate_slow():
  """Use numpy concate images."""
  cap = cv2.VideoCapture(FLAGS.video)
  start = time.time()
  for i in range(TEST_NUM):
    multi_images = None
    for j in range(BATCH_SIZE):
       ret, frame = cap.read()
       frame = np.expand_dims(frame, 0)
       if not isinstance(multi_images, np.ndarray):
         multi_images = frame
       else:
         multi_images = np.concatenate([multi_images, frame])
  end = time.time()
  cap.release()
  return end - start

# @profile
# def tf_concate():
#   cap = cv2.VideoCapture(FLAGS.video)
#   with tf.Session().as_default() as sess:
#     print("Graph loaded")
#     start = time.time()
#     for i in range(TEST_NUM):
#       concate_list = []
#       for j in range(BATCH_SIZE):
#          ret, frame = cap.read()
#          frame = tf.expand_dims(frame, 0)
#          concate_list.append(frame)
#       con_result = tf.concat(concate_list, axis=0)
#       r_con_result = sess.run(con_result)
#       del r_con_result
#     end = time.time()
#   cap.release()
#   return end - start

@profile
def tf_concate():
  cap = cv2.VideoCapture(FLAGS.video)
  sess = tf.Session()
  print("Graph loaded")
  start = time.time()
  for i in range(TEST_NUM):
    concate_list = []
    for j in range(BATCH_SIZE):
       ret, frame = cap.read()
       concate_list.append(frame)
    con_result = tf.stack(concate_list, axis=0)
    r_con_result = sess.run(con_result)
    import pdb; pdb.set_trace()
    

  end = time.time()
  cap.release()
  return end - start

def main():
  # np_time = numpy_concate()
  # print("numpy concate: %d" % (np_time,))
  # np_slow_time = numpy_concate_slow()
  # print("numpy concate slow: %d" % (np_slow_time,))
  np_slow_time = tf_concate()
  print("tf concate: %d" % (np_slow_time,))


if __name__ == "__main__":
  main()
