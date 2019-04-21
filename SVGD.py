import tensorflow as tf
import tensorflow_probability as tfp

def pairwise_distance(x):
    # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    norm = tf.reduce_sum(x * x, 1)
    norm = tf.reshape(norm, [-1, 1])
    return norm - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(norm)


def kernel_fn(x, h='median'):
    pdist = pairwise_distance(x)

    if h == 'median':
        # https://stackoverflow.com/questions/43824665/tensorflow-median-value
        lower = tfp.stats.percentile(pdist, 50.0, interpolation='lower')
        higher = tfp.stats.percentile(pdist, 50.0, interpolation='higher')

        median = (lower + higher) / 2.
        median = tf.cast(median, tf.float32)

        h = tf.sqrt(0.5 * median / tf.math.log(x.shape[0] + 1.))
        h = tf.stop_gradient(h)

    return tf.exp(-pdist / h ** 2 / 2)


@tf.function
def svgd(samples, log_prob_fn, negative_log_prob=False):
    """
    Sein Variational Gradient Descent
    https://arxiv.org/pdf/1608.04471
    :param samples: random samples, called particles in the paper.
    :param log_prob_fn: unnormalized log probability function. Can be an energy function f.E.
    :param negative_log_prob: wether to scale log probability by negative coefficient. (for energy functions for example)
    :return: gradient to samples
    """
    num_particles, dim = samples.shape

    kernel = kernel_fn(samples)
    kernel_grad = tf.gradients(kernel, samples)[0]

    log_prob = log_prob_fn(samples)
    log_prob = -log_prob if negative_log_prob else log_prob
    log_prob_grad = tf.gradients(log_prob, samples)[0]

    return (-1) * (tf.matmul(kernel, log_prob_grad) + kernel_grad) / num_particles

