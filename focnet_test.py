import tensorflow as tf

from focnet import FocNet


def test_focnet():
    model = FocNet(n_filters=4)
    model.build(tf.TensorShape([None, None, None, 1]))
