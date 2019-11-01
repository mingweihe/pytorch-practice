# tensorflow 2.0
import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image

class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
    
    def image_summary(self, tag, vals, step):
        with self.writer.as_default():
            for img, pred in zip(*vals):
                img = np.reshape(img, (-1, 28, 28, 1))
                tf.summary.image('%s-pred:%d' % (tag, pred), img, step=step)

    def histogram_summary(self, tag, values, step, bins=1000):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
