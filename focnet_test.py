import numpy as np
import tensorflow as tf

from focnet import FocNet


tf.random.set_seed(0)

def test_focnet():
    model = FocNet(n_filters=4)
    model.build(tf.TensorShape([None, None, None, 1]))

def test_focnet_change():
    model = FocNet(n_filters=4)
    x = tf.random.normal((1, 64, 64, 1))
    y = x + tf.random.normal((1, 64, 64, 1))
    model(x)
    before = [v.numpy() for v in model.trainable_variables]
    model.compile(optimizer='sgd', loss='mse')
    model.train_on_batch(x, y)
    after = [v.numpy() for v in model.trainable_variables]
    for b, a in zip(before, after):
        assert np.any(np.not_equal(b, a))
