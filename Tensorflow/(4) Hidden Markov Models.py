import tensorflow as tf

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time


#%%

# Cold days are encoded by a 0 and hot days are encoded by a 1.
# The first day in our sequence has an 80% chance of being cold.
# A cold day has a 30% chance of being followed by a hot day.
# A hot day has a 20% chance of being followed by a cold day.
# On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
# %%


tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # For Day 1
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])  # For days after Day 1 (transition henceforth)
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above
# range for cold day is from -5°C - 5°C, whereas range for hot day is from 5°C - 25°C
# the loc argument represents the mean and the scale is the standard deviation


# %%

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)  # steps refers to the number of steps represents the number of days that we would like to predict information for.

# %%

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
