# Introduction

This repository is archived. Originally it was meant to hold rewritten
Tensorflow 1.x AutoAugment ops to Tensorflow 2.x until I found
out it was already done.

To use existing AutoAugment / RandAugment one can run the following snippet:

```python
!wget https://raw.githubusercontent.com/tensorflow/models/master/official/vision/image_classification/augment.py

import tensorflow as tf
from augment import AutoAugment, RandAugment

aa = AutoAugment()  # ra = RandAugment()

# Run AutoAugment on batch of images in a for loop:

@tf.function
def apply_autoaugment(images, labels):

    # Create placeholder for stacking results:
    batch_size = tf.shape(images)[0]
    original_dtype = images.dtype

    results = tf.TensorArray(
        dtype=original_dtype,
        size=batch_size,
        element_shape=tf.TensorShape(dims=images[0].shape),
    )

    for batch_id in tf.range(batch_size):
        single_tensor = tf.gather(images, batch_id)
        transformed_frame = aa.distort(single_tensor)
        results = results.write(batch_id, transformed_frame)

    transformed_images = results.stack()
    return transformed_images, labels

dataset = ... # Load image data
dataset = dataset.map(apply_autoaugment)
```
