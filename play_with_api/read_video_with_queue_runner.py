"""Use queue runner read images."""
import tensorflow as tf
import cv2
import numpy as np


video_path = "1.mp4"
coord = tf.train.Coordinator()


def queue_input():
    cap = cv2.VideoCapture(video_path)
    with tf.device("/cpu:0"):
        while True:
            (grabbed, frame) = cap.read()

            if not grabbed:
                print("reach video ending.")
                coord.request_stop()
                break
            img = np.asarray(frame)
            yield img


with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tensorboard_logs', sess.graph)
    tf.global_variables_initializer()

    queue = tf.FIFOQueue(capacity=5, dtypes=tf.uint8, shapes=[720, 1280, 3])

    test_dataset = tf.data.Dataset.from_generator(queue_input, output_types=tf.uint8)
    value = test_dataset.make_one_shot_iterator().get_next()

    enqueue_op = queue.enqueue(value)

    # Create a queue runner that will run 1 threads in parallel to enqueue
    # examples. In general, the queue runner class is used to create a number of threads cooperating to enqueue
    # tensors in the same queue.
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 1)

    # Create a coordinator, launch the queue runner threads.
    # Note that the coordinator class helps multiple threads stop together and report exceptions to programs that wait
    # for them to stop.frame_num
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    # tf.train.add_queue_runner(qr)

    # Run the training loop, controlling termination with the coordinator.
    frames_tensor = queue.dequeue(name='dequeue')
    while True:
        if coord.should_stop():
            break
        frame = sess.run(frames_tensor)
        cv2.imshow("windows", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    coord.join(enqueue_threads)

train_writer.close()
cv2.destroyAllWindows()
print("quit program")
