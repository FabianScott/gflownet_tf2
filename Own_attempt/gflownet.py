import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

tfd = tfp.distributions

# HUGE ASSUMPTION IN THIS CODE:
# We start in a corner and can reach all states by going
# only in one direction on each axis
class GFlowNet:
    def __init__(self,
                 env,
                 n_layers=2,
                 n_hidden=32,
                 gamma=0.5,
                 epochs=100,
                 lr=0.005,
                 decay_steps=10000,
                 decay_rate=0.8
                 ):
        self.env = env
        self.dim = env.dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.max_trajectory_length = self.dim * self.env.size
        self.action_space_size = self.dim + 1    # Assuming we need not move backwards

        lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        self.get_model()

    def get_model(self):
        # GO BACK AND REVIEW
        # Dimension are due to one-hot encoding of coordinates
        # columns represent 0-env.size and each row is a dimension
        # of the environment, see bottom of demo file.
        input_ = Input(shape=(self.dim, self.env.size), name='Input')
        layers = [Dense(units=self.n_hidden, activation='relu')(input_)]
        for _ in range(self.n_layers):
            layers.append(Dense(units=self.n_hidden, activation='relu')(layers[-1]))
        fpm = Dense(units=self.dim + 1, activation='log_softmax', name='forward_policy')(layers[-1])
        bpm = Dense(units=self.dim, activation='log_softmax', name='backward_policy')(layers[-1])
        self.z0 = tf.Variable(0., name='z0', trainable=True)
        self.model = Model(input_, [fpm, bpm])
        self.unif = tfd.Uniform(low=[0] * (self.dim + 1), high=[1] * (self.dim + 1))

    def mask_forward_actions(self, position_batch):
        """
        This function returns a mask for a batch of actions,
        checking if any of them are out of bounds.
        """
        batch_size = position_batch.shape[0]
        action_mask = 0 < (position_batch < (self.env.size - 1))
        column = np.ones((batch_size, 1))  # Choosing '1' (stop) is always allowed
        return np.append(action_mask, column, axis=1)

    def mask_and_norm_forward_actions(self, position_batch, forward_prob_batch):
        """
        Weight actions by their probabilities after applying
        the mask, then normalise, so they sum to 1 and represent
        probabilities for each action
        """
        mask = self.mask_forward_actions(position_batch)
        mask_prob = mask * forward_prob_batch.numpy()
        return mask_prob / np.sum(mask_prob, axis=1, keepdims=True)

    def sample_trajectories(self, batch_size=5, explore=False):
        """
        Using the current policy, sample (batch_size) trajectories from
        the environment, starting in the origin.

        """
        continue_sampling = np.ones(batch_size, dtype=bool)
        positions = np.zeros((batch_size, self.dim))    # notice dimensions, one position per batch
        trajectories = [positions.copy()]
        one_hot_actions = []
        batch_rewards = np.zeros(batch_size)

        for step in range(self.max_trajectory_length - 1):
            # Convert to one-hot, then use the model to predict probabilities
            # of each potential action (It seems we assume there are only
            # 3 directions, no moving backwards)
            positions_one_hot = tf.one_hot(positions, self.env.size, axis=-1)
            # Using exp(output) to turn into a probability
            model_forward_probs = tf.math.exp(self.model.predict(positions_one_hot)[0])
            actions_normalised = self.mask_and_norm_forward_actions(positions, model_forward_probs)
            # Given the normalised actions, sample them based on their probabilities
            # The output here is in the range [0-self.action_space[ since
            # an action can only be moving up or to the right, as this
            # allows us to reach all possible states

            actions = tfd.Categorical(probs=actions_normalised).sample()
            one_hot_action = tf.one_hot(actions, self.action_space_size).numpy()
            # Iterate through the sampled actions, one for each batch
            for i, action in enumerate(actions):
                # The action not in one-hot-encoding means 'stop' and has
                # the value self.dim, thus this is the and of the loop
                if action == self.dim and continue_sampling[i]:
                    continue_sampling[i] = False
                    batch_rewards[i] = self.env.get_reward(positions[i])
                elif not continue_sampling[i]:
                    positions[i] = -1
                    one_hot_action[i] = 0
                else:
                    positions[i, action] += 1
            # For each position traversed and the actions used to get there,
            # append them to the lists
            trajectories.append(positions.copy())
            one_hot_actions.append(one_hot_action)
        return np.stack(trajectories, axis=1), np.stack(one_hot_actions, axis=1), batch_rewards



