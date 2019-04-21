import matplotlib.pyplot as plt
import tensorflow as tf


def density2image(model, size, extent):
    grid_x = tf.linspace(start=extent[0], stop=extent[1], num=size)
    grid_y = tf.linspace(start=extent[3], stop=extent[2], num=size)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid_x, grid_y = tf.reshape(grid_x, shape=(size, size, 1)), tf.reshape(grid_y, shape=(size, size, 1))
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.reshape(grid, shape=(size ** 2, 2))

    p = model(grid)
    return tf.reshape(p, shape=(size, size)).numpy()

