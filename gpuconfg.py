import pyopencl as cl
import tensorflow as tf
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

platforms = cl.get_platforms()
for platform in platforms:
    print("Platform:", platform.name)
    devices = platform.get_devices()
    for device in devices:
        print("Device:", device.name, "| Type:", device.type)

print(tf.config.list_physical_devices('GPU'))
print("TensorFlow Version:", tf.__version__)
print("Is oneDNN enabled?", tf.config.optimizer.get_experimental_options())