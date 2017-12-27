# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def main(_):
  with tf.Graph().as_default() as graph:
    with tf.device('/device:CPU:0'):
      global_steps = tf.train.create_global_step()
      hooks = [tf.train.StopAtStepHook(num_steps=5)]
      # StopAtStepHook need global step.
      # Generally, optimizer will increase globa step,but  int this example,
      # we need increase it manually.
      increase_global_step = tf.assign_add(global_steps, 1)
      with tf.train.MonitoredTrainingSession(
                checkpoint_dir="/tmp/test23432xx",
                hooks=hooks,
                log_step_count_steps=10,
                save_summaries_steps=10) as sess:
          count = 0
          while not sess.should_stop():
            count += 1
            print(sess.run(increase_global_step))
      print("Actual runing steps: %d" % (count,))


if __name__ == '__main__':
  tf.app.run()
