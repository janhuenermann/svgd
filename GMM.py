import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from SVGD import svgd, kernel_fn
from _helper import density2image

tfd = tfp.distributions

"""This toy example shows that using SVGD we can learn
   a sampler network that samples from a Gaussian mixture 
   density function! """

toy_dist = tfd.Mixture(
    cat=tfd.Categorical(probs=[0.5, 0.5]),
    components=[
        tfd.MultivariateNormalDiag(loc=tf.constant([-1., +1]), scale_diag=tf.constant([0.2, 0.2])),
        tfd.MultivariateNormalDiag(loc=tf.constant([+1., -1]), scale_diag=tf.constant([0.2, 0.2]))
    ])


def target_density(x):
    return toy_dist.prob(x)


def target_density_log(x):
    return toy_dist.log_prob(x)


def block(x):
    y = K.layers.Dense(8)(x)
    y = K.layers.BatchNormalization()(y)
    y = K.layers.Activation('tanh')(y)

    return y


def build_sampler(x):
    """Just some random network architecture"""
    input = x

    x = K.layers.concatenate([block(x), x])
    x = K.layers.concatenate([block(x), x])
    x = K.layers.concatenate([block(x), x])

    x = K.layers.Dense(2, kernel_initializer=K.initializers.VarianceScaling(scale=1))(x)

    return K.models.Model(inputs=input, outputs=x)


K.backend.set_learning_phase(1)

latent_size = 12
latent_distribution = tfd.MultivariateNormalDiag(
    loc=tf.zeros(shape=[latent_size]),
    scale_diag=2 * tf.ones(shape=[latent_size]))

sampler = build_sampler(K.layers.Input(shape=(latent_size,)))
batch_size = 256

fig, ax = plt.subplots()
fig.set_tight_layout(True)

extent = [-5.0, 5.0, -5.0, 5.0]

ax.imshow(density2image(target_density, size=100, extent=extent), extent=extent)
ax.invert_yaxis()
ax.grid(zorder=10, color='#cccccc', alpha=0.5, linestyle='-.', linewidth=0.7)

points, = ax.plot([], [], '.', c='w', alpha=0.75, markersize=3)

optim = tf.optimizers.Adagrad(learning_rate=0.05)
eps = latent_distribution.sample(sample_shape=batch_size)


def step(i):
    global eps

    label = 'timestep {0}'.format(i)
    print(label)

    eps = latent_distribution.sample(sample_shape=batch_size)

    with tf.GradientTape() as tape:
        tape.watch(eps)
        samples = sampler(eps)

    grads = tape.gradient(target=samples, sources=sampler.trainable_variables,
                          output_gradients=svgd(samples, target_density_log))
    grads, norm = tf.clip_by_global_norm(grads, clip_norm=1.0)

    optim.apply_gradients(zip(grads, sampler.trainable_variables))

    points.set_data(samples[..., 0].numpy(), samples[..., 1].numpy())
    ax.set_xlabel(label)

    return points, ax


ani = FuncAnimation(fig, step, frames=500, interval=20)
# ani.save('anim.mp4', writer='ffmpeg')
plt.show()