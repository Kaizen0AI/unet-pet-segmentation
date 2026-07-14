import tensorflow_datasets as tfds
import tensorflow as tf
from src.config import BATCH_SIZE, IMAGE_SIZE

def load_dataset(validation_split=0.2):
    dataset_name = "oxford_iiit_pet"
    data_dir = "E:/datasets"
    dataset, info = tfds.load(
        dataset_name,
        data_dir=data_dir,
        with_info=True,
        as_supervised=False,
    )

    train_dataset = dataset["train"]

    train_size = int(0.8 * info.splits["train"].num_examples)
    val_size = info.splits["train"].num_examples - train_size

    train_dataset = dataset["train"].shuffle(
    info.splits["train"].num_examples,
    seed=42,
    reshuffle_each_iteration=False
    )
    
    train_ds = train_dataset.take(train_size)
    val_ds = train_dataset.skip(train_size)
    test_ds = dataset['test']
    return train_ds, val_ds, test_ds, info

def preprocess(dataset, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, augmentation=False, training=False):
    """
    Preprocess a tf.data.Dataset.

    Training datasets should use:
      augmentation=True
      training=True

    Validation/Test datasets should use:
      augmentation=False
      training=False
    """
    def preprocess_sample(sample):
        image = tf.image.resize(sample['image'], image_size)
        mask = tf.image.resize(sample['segmentation_mask'], image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.int32)
        mask = mask - 1  # Adjust mask values to start from 0
        if augmentation:
            flip = tf.random.uniform([]) > 0.5
            image, mask = tf.cond(
                flip, lambda: (
                    tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
                    ), 
                    lambda: (image, mask)
                    )
        return image, mask
    if training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset