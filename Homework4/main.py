import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

train_ds, test_ds = tfds.load('mnist', split=['train','test'], as_supervised=True)

#tfds.show_examples(train_ds, ds_info)
def prepare_mnist_data(mnist):
    #turning the images into vectors
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #converting data into float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    #normalization (values  [0,255] --> [-1,1])
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
    #creating one-hot targets (vector: one-hot vectors being 0-10. )
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    #cache this in memory
    mnist = mnist.cache()
    mnist = mnist.shuffle(1000) #simply shuffling the data
    mnist = mnist.batch(64) #batching the data in stacks of size x, concatinated along the first axes -> 28*28*x
    mnist = mnist.prefetch(20) #prepares 20 datapoints at a time so they can be used by the network (ann=gpu, prefetch=cpu)
    return mnist

train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)