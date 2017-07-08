import tensorflow as tf

# Define a single scalar Normal distribution.
dist = tf.contrib.distributions.Normal(loc=0., scale=3.)

# Evaluate the cdf at 1, returning a scalar.
dist.cdf(1.)

# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = tf.contrib.distributions.Normal(loc=[0.], scale=[1.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
# dist.prob([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
print(dist.sample([10, 20]))
