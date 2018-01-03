import tensorflow as tf
import glob
import gc
# from memory_profiler import profile


@profile
def function_mark():
  pass

@profile
def stack_images():
  image_file_list = glob.glob("car_images/*.jpg")
  sess = tf.Session()

  for _ in range(300):
    # read image
    image1 = tf.gfile.FastGFile(image_file_list[0], 'rb').read()
    image2 = tf.gfile.FastGFile(image_file_list[1], 'rb').read()
    # decode image
    image1_decode = tf.image.decode_image(image1, channels=3)
    image2_decode = tf.image.decode_image(image2, channels=3)
    # stack image
    image_stack = tf.stack([image1_decode, image2_decode])
    # run session
    r_image_stack = sess.run(image_stack)
    # mark function. so I can check the memory-usage of every loop.
    function_mark()
    # force garbage collection, so all the un-reference variable will be freed.
    del r_image_stack
    gc.collect()




if __name__ == "__main__":
  stack_images()


def simple_example():
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  stack_1 = tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
  stack_2 = tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]

  sess = tf.Session()

  r_stack_1, r_stack_2 = sess.run([stack_1, stack_2])

  print(r_stack_1)
  print("=======================")
  print(r_stack_2)
