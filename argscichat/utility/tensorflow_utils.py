import tensorflow as tf
import numpy as np


def add_gradient_noise(t, stddev=1e-3, name="add_gradient_noise"):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """

    with tf.name_scope(name) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random.normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def decode_record(record, name_to_features):
    """
    TPU does not support int64
    """

    example = tf.io.parse_single_example(record, name_to_features)

    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def load_single_dataset(filepath, name_to_features):
    data = tf.data.TFRecordDataset(filepath)
    data = data.map(lambda record: decode_record(record, name_to_features))
    return data


def create_dataset(filepath, batch_size, name_to_features, selector, is_training=True,
                   input_pipeline_context=None, shuffle_amount=10000, prefetch_amount=1024,
                   reshuffle_each_iteration=True, sampling=False, sampler=None):
    dataset = load_single_dataset(filepath=filepath, name_to_features=name_to_features)

    # Dataset is sharded by the number of hosts (num_input_pipelines == num_hosts)
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)

    dataset = dataset.map(selector, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_amount, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.repeat()
        if sampling:
            dataset = dataset.map(lambda x, y: sampler.sampling((x, y)),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(prefetch_amount)
    return dataset


def get_dataset_fn(filepath, batch_size, name_to_features, selector, is_training=True,
                   shuffle_amount=10000, prefetch_amount=1024,
                   reshuffle_each_iteration=True,
                   sampling=False, sampler=None):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset"""
        bs = ctx.get_per_replica_batch_size(batch_size) if ctx else batch_size
        dataset = create_dataset(filepath=filepath, batch_size=bs,
                                 name_to_features=name_to_features,
                                 selector=selector,
                                 is_training=is_training,
                                 input_pipeline_context=ctx,
                                 shuffle_amount=shuffle_amount,
                                 reshuffle_each_iteration=reshuffle_each_iteration,
                                 prefetch_amount=prefetch_amount,
                                 sampling=sampling,
                                 sampler=sampler)
        return dataset

    return _dataset_fn


def retrieve_numpy_labels(data_fn, steps):
    numpy_data = list(data_fn().map(lambda x, y: y).take(steps).as_numpy_iterator())
    if type(numpy_data[0]) == dict:
        numpy_data = {key: np.concatenate([item[key] for item in numpy_data]) for key in numpy_data[0].keys()}
    else:
        numpy_data = np.concatenate([item for item in data_fn().map(lambda x, y: y).take(steps)])

    return numpy_data
