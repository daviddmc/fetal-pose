import tensorflow as tf
from math import ceil

def get_dataset_from_indexable(map_fn, output_types, output_shapes, N, batch_size, is_shuffle=True, n_parallel=4):
  
    ds = tf.data.Dataset.range(N)
    if is_shuffle:
        ds = ds.shuffle(buffer_size=N)

    def map_func(i):
        outputs = tf.py_func(map_fn, [i], output_types, stateful=False)
        for out, shape in zip(outputs, output_shapes):
            out.set_shape(shape)
        return outputs
    
    ds = ds.map(map_func, num_parallel_calls=min(batch_size, n_parallel)).batch(batch_size)
    ds = ds.prefetch(1)
    ds.length = int(ceil(float(N)/batch_size))

    return ds