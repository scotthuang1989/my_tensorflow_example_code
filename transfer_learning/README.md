# Target:

* Take a look at existing models
  * [pre-trained model](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)

* Apply this on cifar10 dataset.


## Concept

### Bottlenecks

An informal term we often use for the layer just before the final output layer that actually does the classification.

* This penultimate layer has been trained to output a set of values that's good enough for the classifier to user to distinguish between all the classes it's been asked to recognize.



## Setup and Running
[reference](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)

### Clone the git repo

```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2

cd tensorflow-for-poets-2
```

### Download the training images

get some example images

```
curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files
```

### Re-training the network

The retrain script can retrain either:
* Inception V3 model
* MobileNet

In this example, we will choosse Inception V3

#### Configuration of Inception V3

* Input image resolution: 128,160, 192 or 224

* The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.5 or 0.25.

we can set 2 shell variables as follows :
```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
```
### Run the training

```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos
```

## Using the Retrained model

The retraining script writes data to the following two files:

* tf_files/retrained_graph.pb, which contains a version of the selected network with a final layer retrained on your categories.

* tf_files/retrained_labels.txt, which is a text file containing labels.

Here are command to use the model.
```
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```
