[REFERENCE OF THIS GUIDE](https://github.com/tensorflow/ecosystem)

This guide setup  a kubernets cluster and run [this example](https://github.com/tensorflow/ecosystem/tree/master/kubernetes) distributly.

## Build docker image
```shell
cd ~/github
git clone https://github.com/tensorflow/ecosystem.git
cd ecosystem/docker
```
**
before build the image, I made some modificationes:
add a stop hook. so the training process will stop after 100000 steps.
I choose this step because, the running process will long enough to show the process and
at the same time will not run forever.
**
```python
hooks=[tf.train.StopAtStepHook(last_step=100000)]
with tf.train.MonitoredTrainingSession(
    master=target,
    is_chief=(FLAGS.task_index == 0),
    checkpoint_dir=FLAGS.train_dir,hooks=hooks) as sess:
  while not sess.should_stop():
    print("training...")
    sess.run(train_op)
```
build the image
```shell
docker build -t your_docker_id/my_tf_dist_example:v10 -f Dockerfile .
```
output
```
Sending build context to Docker daemon   25.6kB
Step 1/3 : FROM tensorflow/tensorflow:nightly
 ---> 0a25191914d4
Step 2/3 : COPY mnist.py /
 ---> Using cache
 ---> 512e4eb0e62e
Step 3/3 : ENTRYPOINT python /mnist.py
 ---> Using cache

 ---> 2615ec12af61
Successfully built 2615ec12af61
Successfully tagged my_tf_dist_example:v1
```
make sure image is built succesfully
```shell
docker image ls
```
output:
```shell
REPOSITORY                     TAG                 IMAGE ID            CREATED             SIZE
my_tf_dist_example             v1                  2615ec12af61        3 days ago          1.24GB
tfeco                          v1                  2615ec12af61        3 days ago          1.24GB
tensorflow/tensorflow          nightly             0a25191914d4        3 days ago          1.24GB
friendlyhello                  latest              f2bd3efbe85b        2 weeks ago         696MB
<none>                         <none>              a5160d548da1        2 weeks ago         696MB
```
push the image to docker-hub
```shell
docker push your_docker_id/my_tf_dist_example:v1
```

## Build tf_records
command:
```python
python  tensorflow/tensorflow/examples/how_tos/reading_data/convert_to_records.py\
```
generated tfrecords located at `/tmp/data/`
Here are the file lists:
```
test.tfrecords
train.tfrecords
validation.tfrecords
```

## Test the docker image locally, before go distribution

### Crate a local docker volume to store input data and training data
```shell
# create volumne
$ docker volume create mnist_data_vol
```
after copy input data:(train.tfrecords, test.tfrecords, validation.tfrecords) into it and create train directory.
we should have following directory structure.
```
root@scott-z230:/var/lib/docker/volumes/mnist_data_vol/_data# tree .
.
├── data
│   ├── test.tfrecords
│   ├── train.tfrecords
│   └── validation.tfrecords
└── train_dir
```
run the container locally.
```shell
docker run --mount source=mnist_data_vol,target=/data ng7711/my_tf_dist_example:v7 --train_dir="/data/train_dir" --data_dir="/data/data"
```

## Go distribution

### setup kubernets cluster with kubeadm
* here are the [official guide](https://kubernetes.io/docs/setup/independent/install-kubeadm/)
* [my notes](https://github.com/scotthuang1989/tools_study/tree/master/kubernetes)
* I setup a cluster with 2 nodes.

### Setup a PersistentVolume for input data and train data.

* choose Volume type: NFS
* setup up a NFS server [reference doc](http://blog.csdn.net/scotthuang1989/article/details/77839772)
* [setup a kubernetes NFS Volume](https://github.com/scotthuang1989/tools_study/tree/master/kubernetes/nfspv)
* copy tfrecords to this corresponding location on the host.


### Customize template file

set following parameter in myjob.template.jinjia

* name
* image
* worker_replicas
* ps_replicas
* script
* data_dir
* train_dir
**note**: data_dir and train_dir should both be set to your PV(not a local path), because it need to be accessed from all pods.

#### Generate template
```shell
python render_template.py myjob.template.jinjia > myjob.template
```

### Run training

```shell
kubectl create -f myjob.template
```

### Check and Debug

* you can print log from every pods with

```
kubectl logs pod_names
```

Here is the content in my train_dir after tranning complete:
```shell
-rw-r--r-- 1 nobody nogroup    373 Sep  5 16:54 checkpoint
-rw-r--r-- 1 nobody nogroup 392818 Sep  5 15:58 events.out.tfevents.1504597661.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 15:58 events.out.tfevents.1504598291.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 15:58 events.out.tfevents.1504598309.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 15:59 events.out.tfevents.1504598338.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:00 events.out.tfevents.1504598398.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:02 events.out.tfevents.1504598531.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:07 events.out.tfevents.1504598752.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:16 events.out.tfevents.1504599177.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:23 events.out.tfevents.1504599679.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:31 events.out.tfevents.1504600103.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:38 events.out.tfevents.1504600617.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:47 events.out.tfevents.1504601031.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 195244 Sep  5 16:54 events.out.tfevents.1504601558.mymnistdist-worker-0-746hm
-rw-r--r-- 1 nobody nogroup 152019 Sep  5 16:54 graph.pbtxt
-rw-r--r-- 1 nobody nogroup 473132 Sep  5 16:31 model.ckpt-100016.data-00000-of-00001
-rw-r--r-- 1 nobody nogroup    347 Sep  5 16:31 model.ckpt-100016.index
-rw-r--r-- 1 nobody nogroup  63878 Sep  5 16:31 model.ckpt-100016.meta
-rw-r--r-- 1 nobody nogroup 473132 Sep  5 16:38 model.ckpt-100017.data-00000-of-00001
-rw-r--r-- 1 nobody nogroup    347 Sep  5 16:38 model.ckpt-100017.index
-rw-r--r-- 1 nobody nogroup  63878 Sep  5 16:38 model.ckpt-100017.meta
-rw-r--r-- 1 nobody nogroup 473132 Sep  5 16:47 model.ckpt-100018.data-00000-of-00001
-rw-r--r-- 1 nobody nogroup    347 Sep  5 16:47 model.ckpt-100018.index
-rw-r--r-- 1 nobody nogroup  63878 Sep  5 16:47 model.ckpt-100018.meta
-rw-r--r-- 1 nobody nogroup 473132 Sep  5 16:54 model.ckpt-100019.data-00000-of-00001
-rw-r--r-- 1 nobody nogroup    347 Sep  5 16:54 model.ckpt-100019.index
-rw-r--r-- 1 nobody nogroup  63878 Sep  5 16:54 model.ckpt-100019.meta
-rw-r--r-- 1 nobody nogroup 473132 Sep  5 16:54 model.ckpt-100020.data-00000-of-00001
-rw-r--r-- 1 nobody nogroup    347 Sep  5 16:54 model.ckpt-100020.index
-rw-r--r-- 1 nobody nogroup  63878 Sep  5 16:54 model.ckpt-100020.meta
```
