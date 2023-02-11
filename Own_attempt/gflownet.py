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
                 decay_rate=0.8,
                 n_samples=1000
                 ):
        self.env = env
        self.dim = env.dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gamma = gamma  # weighting of random sampling
        self.epochs = epochs
        self.lr = lr
        self.max_trajectory_length = self.dim * self.env.size
        self.action_space_size = self.dim + 1  # Assuming we need not move backwards
        self.n_samples = n_samples

        lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.get_model()
        self.data = {'positions': None, 'actions': None, 'rewards': None}

    def get_model(self):
        # GO BACK AND REVIEW
        # Dimension are due to one-hot encoding of coordinates
        # columns represent 0-env.size and each row is a dimension
        # of the environment, see bottom of demo file.
        input_ = Input(shape=(self.dim, self.env.size), name='Input')
        flatten = Flatten()(input_)
        layers = [Dense(units=self.n_hidden, activation='relu')(flatten)]
        for i in range(self.n_layers - 1):
            layers.append(Dense(units=self.n_hidden, activation='relu')(layers[-1]))
        fpm = Dense(units=self.dim + 1, activation='log_softmax', name='forward_policy')(layers[-1])
        bpm = Dense(units=self.dim, activation='log_softmax', name='backward_policy')(layers[-1])
        self.z0 = tf.Variable(0., name='z0', trainable=True)
        self.model = Model(input_, [fpm, bpm])
        self.unif = tfd.Uniform(low=[0] * (self.dim + 1), high=[1] * (self.dim + 1))

    def sample_trajectories(self, batch_size=5, explore=False):
        """
        Using the current policy, sample (batch_size) trajectories from
        the environment, starting in the origin.

        """
        continue_sampling = np.ones(batch_size, dtype=bool)
        # dtype must be int for use in indexing reward_space
        positions = np.zeros((batch_size, self.dim), dtype=int)  # notice dimensions, one position per batch
        trajectories = [positions.copy()]
        one_hot_actions = []
        batch_rewards = np.zeros(batch_size)

        for step in range(self.max_trajectory_length - 1):
            # Convert to one-hot, then use the model to predict probabilities
            # of each potential action (It seems we assume there are only
            # 3 directions, no moving backwards)
            positions_one_hot = tf.one_hot(positions, self.env.size, axis=-1)
            # Using exp(output) to turn into a probability
            model_forward_logits = self.model.predict(positions_one_hot)[0]
            model_forward_probs = tf.math.exp(model_forward_logits)

            if explore:  # Sample from a uniform distribution for random actions
                uniform_sample = self.unif.sample(sample_shape=model_forward_probs.shape[0])
                model_forward_probs = uniform_sample * self.gamma + model_forward_probs

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

    def sample_trajectories_bakwards(self, position, from_grad=False):
        position_one_hot = tf.one_hot(np.expand_dims(position, 0), self.env.size, axis=-1)
        positions = [position_one_hot]
        # Start by storing the terminating action
        actions = [tf.one_hot(self.action_space_size - 1, self.action_space_size).numpy()]
        # Check if in origin already,
        keep_going = not np.all(position == np.zeros(len(position)))

        current_position = position.copy()
        while keep_going:

            if from_grad:
                model_backward_logits = self.model(position_one_hot)[1]
            else:
                model_backward_logits = self.model.predict(position_one_hot)[1]
            model_backward_probs = tf.math.exp(model_backward_logits)
            normalised_actions = self.mask_and_norm_backward_actions(current_position, model_backward_probs)
            action = np.argmax(normalised_actions)
            action_one_hot = tf.one_hot(action, self.action_space_size).numpy()

            # Again assuming only one direction of movement per axis
            current_position[action] -= 1
            position_one_hot = tf.one_hot(np.expand_dims(current_position, 0), self.env.size, axis=-1)
            # Check if in origin
            keep_going = not np.all(current_position == np.zeros(len(position)))
            positions.append(position_one_hot.numpy().copy())
            actions.append(action_one_hot.copy())

        # Return in reverse order, so starting from origin
        return np.flip(np.concatenate(positions, axis=0), axis=0), \
            np.flip(np.stack(actions, axis=0), axis=0)

    def sample(self, n_samples=None, explore=True, only_unique=True):
        n_samples = self.n_samples if n_samples is None else n_samples
        trajectories, actions, rewards = self.sample_trajectories(n_samples, explore)
        # extract only the final positions, since we have the actions to get the
        # entire trajectory that is all we need
        positions = np.stack([self.get_last_position(trajectory) for trajectory in trajectories], axis=0)

        if self.data['positions'] is None:
            self.data['trajectories'] = trajectories
            self.data['positions'] = positions
            self.data['actions'] = actions
            self.data['rewards'] = rewards
        else:
            # (batch, len_trajectory, env dimensions)
            self.data['trajectories'] = np.append(self.data['trajectories'], trajectories, axis=0)
            # (batch, env dimensions)
            self.data['positions'] = np.append(self.data['positions'], positions, axis=0)
            # (batch, len_trajectory-1, action dimensions)
            self.data['actions'] = np.append(self.data['actions'], actions, axis=0)
            # (batch,)
            self.data['rewards'] = np.append(self.data['rewards'], rewards, axis=0)

        if only_unique:
            u_positions, u_indices = np.unique(positions, axis=0, return_index=True)
            self.data['trajectories'] = self.data['trajectories'][u_indices]
            self.data['positions'] = u_positions
            self.data['actions'] = self.data['actions'][u_indices]
            self.data['rewards'] = self.data['rewards'][u_indices]

    def train_sampler(self, batch_size=10):
        """
        Returns a list of tuples (final_position, reward)
        for all data in self.data in a randomised order

        """
        out = []
        data_len = self.data['rewards'].shape[0]
        num_iterations = int(data_len // batch_size) + 1
        shuffled_indicies = np.random.choice(data_len, size=data_len, replace=False)
        for i in range(num_iterations):
            sample_indicies = shuffled_indicies[i * batch_size:(i + 1) * batch_size]
            out.append((
                self.data['positions'][sample_indicies],
                tf.convert_to_tensor(self.data['rewards'][sample_indicies], dtype='float32')
            ))
        return out

    def trajectory_balance_loss(self, batch, from_grad=False):

        positions, rewards = batch
        losses = []
        for reward, position in zip(rewards, positions):
            trajectory, backward_actions = self.sample_trajectories_bakwards(position, from_grad=from_grad)
            tf_trajectory = tf.convert_to_tensor(trajectory, dtype='float32')
            # For each position in the trajectory, get the forward and backward
            # policies:
            if from_grad:
                forward_policy, backward_policy = self.model(tf_trajectory)
            else:
                forward_policy, backward_policy = self.model.predict(tf_trajectory)

            # Now find the probabilities ascribed to each action by the forward
            # and backward policies respectively
            forward_probs = tf.reduce_sum(
                tf.multiply(forward_policy, backward_actions),
                axis=1
            )
            # Ignore origin, but append a 0 for stopping
            backward_probs = tf.reduce_sum(
                tf.multiply(backward_policy[1:, :], backward_actions[:-1, :self.dim]),
                axis=1
            )
            backward_probs = tf.concat([backward_probs, [0]], axis=0)
            # Calculate the log sum of the probabilities using fancy
            sum_forward = tf.reduce_sum(forward_probs)
            sum_backward = tf.reduce_sum(backward_probs)
            # Calculate trajectory balance loss function and add to batch loss
            numerator = self.z0 + sum_forward
            denominator = tf.math.log(reward) + sum_backward
            tb_loss = tf.math.pow(numerator - denominator, 2)
            losses.append(tb_loss)

        return losses

    def train(self, weight_path='data/weights', batch_size=10, verbose=True):
        loss_results = []
        min_loss = np.inf
        self.sample()
        for epoch in range(self.epochs):
            epoch_loss_list = []
            for batch in self.train_sampler(batch_size):
                loss_values, gradients = self.grad(batch)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables + [self.z0])
                )
                losses_batch = [sample for sample in loss_values]
                epoch_loss_list.append(np.mean(losses_batch))
            epoch_loss = np.mean(epoch_loss_list)
            if epoch_loss < min_loss:
                self.model.save_weights(weight_path)
                min_loss = epoch_loss
            loss_results.append(epoch_loss)
            if verbose and epoch % 9 == 0:
                print(f'Epoch: {epoch} Loss: {epoch_loss}')

        self.model.load_weights(weight_path)
        return loss_results

    # The following are helper functions

    def mask_forward_actions(self, position_batch):
        """
        This function returns a mask for a batch of actions,
        checking if any of them are out of bounds.
        """
        batch_size = position_batch.shape[0]
        # In the following, the zero in not necessary given the assumption
        # of only moving in one direction per axis:
        action_mask = 0 < (position_batch < (self.env.size - 1))
        column = np.ones((batch_size, 1))  # Choosing 'stop' is always allowed
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

    def mask_and_norm_backward_actions(self, position, backward_probs):
        masked_actions = position * backward_probs.numpy()
        return masked_actions / np.sum(masked_actions, axis=1, keepdims=True)

    def grad(self, batch):
        # Directly copied
        """Calculate gradients based on loss function values. Notice the z0 value is
        also considered during training.
        :param batch: (tuple of ndarrays) Output from self.train_gen() (positions, rewards)
        :return: (tuple) (loss, gradients)
        """
        with tf.GradientTape() as tape:
            loss = self.trajectory_balance_loss(batch, from_grad=True)
            grads = tape.gradient(loss, self.model.trainable_variables + [self.z0])
        return loss, grads

    def get_last_position(self, trajectory):
        mask = trajectory != -1
        trajectory_no_pad = trajectory[mask[:, 0]]
        return trajectory_no_pad[-1]

    def clear_eval_data(self):
        """Refresh self.eval_data dictionary."""
        self.eval_data = {'positions': None, 'actions': None, 'rewards': None}


if __name__ == '__main__':
    from cube_env import CubeEnv

    env = CubeEnv()
    agent = GFlowNet(env)
    agent.train()
