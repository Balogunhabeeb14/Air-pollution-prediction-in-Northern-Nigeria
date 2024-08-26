import tensorflow as tf

# Verify that TensorFlow sees the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
