Target of this post is try to implement early stopping and periodiclly report accuracy with tensorflow `Estimator`

Tensorflow: 1.3

## Implement with tf.estimator.Estimator and tf.estimator.EstimatorSpec

if we check interface of Estimator and EstimatorSpec, the only place we can implement this function is `training_hooks` of `EstimatorSpec`.

Estimator
```
__init__(
    model_fn,
    model_dir=None,
    config=None,
    params=None
)

train(
    input_fn,
    hooks=None,
    steps=None,
    max_steps=None
)
```
EstimatorSpec

```
__new__(
    cls,
    mode,
    predictions=None,
    loss=None,
    train_op=None,
    eval_metric_ops=None,
    export_outputs=None,
    training_chief_hooks=None,
    training_hooks=None,
    scaffold=None
)
```
I find a hook that seems can do the job(google "early stopping tensorflow" ):

tensorflow.contrib.learn.monitors.ValidationMonitor:

```
  def __init__(self, x=None, y=None, input_fn=None, batch_size=None,
               eval_steps=None,
               every_n_steps=100, metrics=None, hooks=None,
               early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, name=None):

early_stopping_rounds: `int`. If the metric indicated by
    `early_stopping_metric` does not change according to
    `early_stopping_metric_minimize` for this many steps, then training
    will be stopped.
early_stopping_metric: `string`, name of the metric to check for early
    stopping.
early_stopping_metric_minimize: `bool`, True if `early_stopping_metric` is
    expected to decrease (thus early stopping occurs when this metric
    stops decreasing), False if `early_stopping_metric` is expected to
    increase. Typically, `early_stopping_metric_minimize` is True for
    loss metrics like mean squared error, and False for performance
    metrics like accuracy.
```

So I try to add add a hook with following code:

```
validation_monitor = monitors.ValidationMonitor(
    input_fn=functools.partial(input_fn, subset="evaluation"),
    eval_steps=128,
    every_n_steps=88,
    early_stopping_metric="accuracy",
    early_stopping_rounds = 1000
)
hooks = [per_example_hook,validation_monitor]
classifier = tf.estimator.Estimator(
    model_fn=model_fn_cnn,
    model_dir= FLAGS.train_dir,
    config=config
)
classifier.train(input_fn=functools.partial(input_fn,subset="training"),
                 steps=FLAGS.train_steps,
                 hooks=hooks
                 )
```
I got this error:

```
TypeError: Hooks must be a SessionRunHook, given: <tensorflow.contrib.learn.python.learn.monitors.ValidationMonitor object at 0x7fc726d68
```
This error is pretty clear: `ValidationMonitor` is not Subclass of 'SessionRunHook'.

## Experiment

To get ride of this error, we need use Experiment: `tf.contrib.learn.Experiment`

Here is the code:

```
validation_monitor = monitors.ValidationMonitor(
    input_fn=functools.partial(input_fn, subset="evaluation"),
    eval_steps=128,
    every_n_steps=88,
    early_stopping_metric="accuracy",
    early_stopping_rounds = 1000
)
hooks = [ validation_monitor]
# you can use both core Estimator or contrib Estimator
# contrib_classifier = contrib_estimator.Estimator(
#                 model_fn=model_fn_cnn_experiment,
#                 model_dir=FLAGS.train_dir,
#                 config=config
#                 )
classifier = tf.estimator.Estimator(
    model_fn=model_fn_cnn,
    model_dir= FLAGS.train_dir,
    config=config
)
experiment = tf.contrib.learn.Experiment(
    classifier,
    train_input_fn=functools.partial(input_fn, subset="training"),
    eval_input_fn=functools.partial(input_fn, subset="evaluation"),
    train_steps=FLAGS.train_steps,
    eval_steps=100,
    min_eval_frequency=80,
    train_monitors=hooks
    # eval_metrics="accuracy"
)
experiment.train_and_evaluate()
```

If we dive deeper, we can find the 'root cause of this error'.

Here is a fragment of file: `tensorflow/contrib/learn/python/learn/experiment.py

```
# Estimator in core cannot work with monitors. We need to convert them
# to hooks. For Estimator in contrib, it is converted internally. So, it is
# safe to convert for both cases.
hooks = monitors.replace_monitors_with_hooks(hooks, self._estimator)
if self._core_estimator_used:
  return self._estimator.train(input_fn=input_fn,
                               steps=steps,
                               max_steps=max_steps,
                               hooks=hooks)
else:
  return self._estimator.fit(input_fn=input_fn,
                             steps=steps,
                             max_steps=max_steps,
                             monitors=hooks)

```

the comment state it very clearly. The both `core Estimator` and `contrib Estimator` can not handle minotors, but conbrib Estimator can convert it internally, for core Eestimator we need do it ourself. By use Experiment, The Experiment do it for us.
In fact I find that monitors are deprecated. you can see this thread on [github](https://github.com/tensorflow/tensorflow/pull/12651#issuecomment-329069846).
So here is no buit-in support for early stopping now.
but the implementation of `ValidationMonitor` can certainly give us some clue of how to implement it itself.

## Tricky part

If you run this code. you will find out that the evaluate will only run twice:
* 1st step
* last step.

The solution is that: you need add a config, then pass the config to it to `Eestimator`

```
config = tf.contrib.learn.RunConfig(model_dir=FLAGS.train_dir,save_checkpoints_steps=100)
```
The reason of configuration is that, the evaluation only run when new checkpoint file have been saved. but the default config will only save checkpoint file at first and last step.
so we need specify the frequency of saving checkpoint file.

The realted code of this post is on [here](https://github.com/scotthuang1989/udcity-deeplearning/blob/master/get_highest_score/model/notmnist_train.py)
