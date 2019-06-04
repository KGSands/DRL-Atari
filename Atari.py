"""
###############################################################
########Deep Q-Learning Neural Network for Atari Games#########
###############################################################
A mixture of:
    - My Q-learning chatbot:
    https://github.com/KGSands/DRL-Chatbot/commits/master
    - Hvass Labs Tensorflow Series:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials
###############################################################
Main Classes:
    MotionTracer            Processes images from game into states
    ReplayMemory            The memory which holds the states, action taken and Q-values
    NeuralNetwork           The neural network for estimating Q-values
    Agent                   The agent to learn the environment
###############################################################
"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import scipy
import sys
import time
from timeit import default_timer as timer
import argparse
import sys
import gym
###############################################################
# Checkpoint directory
checkpoint_directory = '/home/keelan/ai/Atari/Checkpoints'
## TODO: Create logs and log directory
###############################################################

# State processing
# From https://github.com/Hvass-Labs/TensorFlow-Tutorials

# Height and width
state_height = 105
state_width = 80

# Size of the images, number of images in the state and shape
state_image_size = np.array([state_height, state_width])
state_channels = 2
state_shape = [state_height, state_width, state_channels]

def _rgb_to_grayscale(image):
    # RBG to grayscale

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return img_gray

def pre_process_image(image):
    # Change frame from game to state

    # Convert to gray-scale
    img = _rgb_to_grayscale(image)

    # Resize
    img = scipy.misc.imresize(img, size=state_image_size, interp='bicubic')

    return img
###############################################################


class MotionTracer:
    """
    Description:
        Processes raw images from the game.
        Uses last 2 images from the game environment (DeepMind = 4) to detect motion
        https://github.com/Hvass-Labs/TensorFlow-Tutorials

    Parameters:
        image: First image from game environment (reset)
        decay: Tail-length of motion trace
    """

    def __init__(self, image, decay = 0.97):

        # Pre-process image
        img = pre_process_image(image)
        self.last_input = img.astype(np.float)

        # Set last output to 0
        self.last_output = np.zeros_like(img)

        self.decay = decay

    def process(self, image):
        """
        Description:
            Process an image from the game

        Parameters:
            image: Image from the game
        """

        # Pre-process
        img = pre_process_image(image)

        # Subtract previous input so only changed pixels remain
        img_dif = img - self.last_input

        # Copy input to last_input
        self.last_input = img[:]

        # Set to white / black based on threshold
        img_motion = np.where(np.abs(img_dif) > 20, 255.0, 0.0)

        # Add the (previous output * decay) to give the tail and clip within bounds
        output = img_motion + self.decay * self.last_output
        output = np.clip(output, 0.0, 255.0)
        self.last_output = output

        return output

    def get_state(self):
        """
        Description:
            Return a state that can be input into the NN.
            This is the last input and last output.
        Parameters:
            N/A
        """

        # Stack last input and output images
        state = np.dstack([self.last_input, self.last_output])

        # Convert to 8-bit to save space
        state = state.astype(np.uint8)

        return state

class ReplayMemory:
    """
    Description:
        This class holds many previous states of the environment

    Parameters:
        size: size of the replay memory (states).
        num_actions: Number of possible actions in the environment.
        discount_rate: The discount factor for updating Q-values.
    """

    def __init__(self, size, num_actions, discount_rate = 0.97):

        # Previous states array
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Q-values corresponding to the states
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Old Q-values for comparing
        self.old_q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions corresponding to the states
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Rewards corresponding to the states
        self.rewards = np.zeros(shape=size, dtype=np.float)

        # Whether the life was lost (died)
        self.end_life = np.zeros(shape=size, dtype=np.bool)

        # Whether the game has ended (game over)
        self.end_episode = np.zeros(shape=size, dtype=np.bool)

        # Number of states
        self.size = size

        # Discount factor per step
        self.discount_rate = discount_rate

        # Reset the size of the replay memory
        self.current_size = 0

        # Split between high and low
        self.error_threshold = 0.1

        # For a balance of high and low values
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

    def is_full(self):
        # Used to check if the replay memory is full

        return self.current_size == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.current_size / self.size

    def reset(self):
        # Empty the replay memory

        self.current_size = 0

    def add_memory(self, state, q_values, action, reward, end_life, end_episode):
        # Add a state into the replay memory

        # Move to current index and increment size
        curr = self.current_size
        self.current_size += 1

        # Store
        self.states[curr] = state
        self.q_values[curr] = q_values
        self.actions[curr] = action
        self.end_life[curr] = end_life
        self.end_episode[curr] = end_episode
        self.rewards[curr] = reward

        # Clip reward between -1.0 and 1.0
        self.rewards[curr] = np.clip(reward, -1.0, 1.0)

    def update_q_values(self):
        # Update the Q-values in the replay memory

        # Keep old q-values
        self.old_q_values[:] = self.q_values[:]

        # Update the Q-values in a backwards loop
        for curr in np.flip(range(self.current_size-1),0):

            # Get data from curr
            action = self.actions[curr]
            reward = self.rewards[curr]
            end_episode = self.end_episode[curr]

            # Calculate Q-Value
            if end_episode:
                # No future steps therefore it is just the observed reward
                value = reward
            else:
                # Discounted future rewards
                value = reward + self.discount_rate * np.max(self.q_values[curr + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[curr] = abs(value - self.q_values[curr, action])

            # Update the Q-value with the better estimate.
            self.q_values[curr, action] = value

    def get_batch_indices(self, batch_size):
        # Get random indices from the replay memory (number = batch_size)

        self.indices = np.random.choice(self.current_size, size=batch_size, replace=False)

    def get_batch_values(self):
        # Get the states and Q-values for these indices

        batch_states = self.states[self.indices]
        batch_q_values = self.q_values[self.indices]

        return batch_states, batch_q_values

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.

        https://github.com/Hvass-Labs/TensorFlow-Tutorials
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.current_size]

        # Create an index of the estimation errors that are low.
        idx = err<self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.current_size
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        https://github.com/Hvass-Labs/TensorFlow-Tutorials
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch


class NeuralNetwork:
    """
    Description:
        This implements the neural network for Q-learning.
        The neural network estimates Q-values for a given state so the agent can decide which action to take.
    Parameters:
        num_actions: The number of actions (number of Q-values needed to estimate)
        replay_memory: This is used to optimize the neural network and produce better Q-values    """

    def __init__(self, num_actions, replay_memory):

        # Reset default graph
        #tf.reset_default_graph()

        # Path for saving/loading checkpoints
        self.checkpoint_dir = os.path.join(checkpoint_directory, "checkpoint")

        # Sample random batches
        self.replay_memory = replay_memory

        # Inputting states into the neural network
        #with tf.name_scope("inputs"):
        self.states = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name='state')

        # Learning rate placeholder
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # Input the new Q-values placeholder that we want the states to map to
        self.q_values_new = tf.placeholder(dtype=tf.float32, shape=[None, num_actions], name='new_q_values')

        # Initialise weights close to 0
        weights = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

        # CNN Padding
        padding = 'SAME'

        # Activation function for the layers
        activation = tf.nn.relu

        # CNN Layer 1
        # Note: input = states
        c_layer1 = tf.layers.conv2d(inputs=self.states, name='CNN_layer1', filters=16,
                                  kernel_size=3, strides=2, padding=padding,
                                  kernel_initializer=weights, activation=activation)

        # CNN Layer 2
        c_layer2 = tf.layers.conv2d(inputs=c_layer1, name='CNN_layer2', filters=32,
                                  kernel_size=3, strides=2, padding=padding,
                                  kernel_initializer=weights, activation=activation)

        # CNN Layer 3
        c_layer3 = tf.layers.conv2d(inputs=c_layer2, name='CNN_layer3', filters=64,
                                  kernel_size=3, strides=1, padding=padding,
                                  kernel_initializer=weights, activation=activation)

        # Flatten output to input into fully-connected network
        flatten_output = tf.contrib.layers.flatten(c_layer3)

        # 1st FC-layer
        fc_layer1 = tf.layers.dense(inputs=flatten_output, name='fc_layer1', units=1024,
                                     kernel_initializer=weights, activation=activation)

        # 2nd
        fc_layer2 = tf.layers.dense(inputs=fc_layer1, name='fc_layer2', units=1024,
                                     kernel_initializer=weights, activation=activation)

        # 3rd
        fc_layer3 = tf.layers.dense(inputs=fc_layer2, name='fc_layer3', units=1024,
                                     kernel_initializer=weights, activation=activation)

        #4th
        fc_layer4 = tf.layers.dense(inputs=fc_layer3, name='fc_layer4', units=1024,
                                     kernel_initializer=weights, activation=activation)

        #5th
        output_layer = tf.layers.dense(inputs=fc_layer4, name='fc_layer5', units=num_actions,
                                     kernel_initializer=weights, activation=None)
        # Set the Q-values equal to the output from the output layer
        #with tf.name_scope('Q-values'):
        self.q_values = output_layer
        #tf.summary.histogram("Q-values", self.q_values)

        # Get the loss
        # Note: mean-squared error between old and new Q-values (L2-Regression)
        #with tf.name_scope('loss'):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.q_values - self.q_values_new), axis = 1))
        #tf.summary.scalar("loss", self.loss)

        # Optimiser for minimising the loss (learn better Q-values)
        #with tf.name_scope('optimizer'):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Create TF session for running NN
        self.session = tf.Session()

        # Merge all summaries for Tensorboard
        #self.merged = tf.summary.merge_all()

        # Create Tensorboard session
        #self.writer = tf.summary.FileWriter(log_directory, self.session.graph)

        # Initialise all variables and run
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving the NN at the end of training
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def close(self):
        # Close TF session
        self.session.close()

    def get_q_values(self, states):
        # Calculate and return the estimated Q-values for the given states

        # Get estimated Q-values from the neural network
        q_values = self.session.run(self.q_values, feed_dict={self.states: states})

        return q_values

    def optimize(self, min_epochs=1.0, max_epochs=10,
                 batch_size=128, loss_limit=0.015,
                 learning_rate=1e-3):
        """
        Additions from https://github.com/Hvass-Labs/TensorFlow-Tutorials
        Description:
            This is the optimization function for the neural network. This updates the Q-values
            from a random batch using the learning rate.
        Parameters:
            current_state: (for graph) The current state being processed when the optimize function is called
            learning_rate: The learning rate of the neural network
            batch_size: The size of the batch taken from the replay memory
        """

        print("Optimization of Neural Network in progress with learning rate {0}".format(learning_rate))
        print("\tLoss-limit: {0:.3f}".format(loss_limit))
        print("\tMax epochs: {0:.1f}".format(max_epochs))

        # Prepare the probability distribution for sampling the replay-memory.
        self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Number of optimization iterations corresponding to one epoch.
        iterations_per_epoch = self.replay_memory.current_size / batch_size

        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs)

        # Maximum number of iterations to perform.
        max_iterations = int(iterations_per_epoch * max_epochs)

        print(max_iterations)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)
        for i in range(max_iterations):
            # Randomly sample a batch of states and target Q-values
            # from the replay-memory. These are the Q-values that we
            # want the Neural Network to be able to estimate.
            state_batch, q_values_batch = self.replay_memory.random_batch()

            # Create a feed-dict for inputting the data to the TensorFlow graph.
            # Note that the learning-rate is also in this feed-dict.
            feed_dict = {self.states: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate}

            # Perform one optimization step and get the loss-value.
            loss_val, _ = self.session.run([self.loss, self.optimizer],
                                           feed_dict=feed_dict)

            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Stop the optimization if we have performed the required number
            # of iterations and the loss-value is sufficiently low.
            if i > min_iterations and loss_mean < loss_limit:
                break

    def save(self, count_states):
        # Save the completed trained network
        self.saver.save(self.session, save_path=self.checkpoint_dir, global_step=count_states)
        print("Checkpoint saved")

    def load(self):
        # Load the network for testing
        try:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_directory)
            self.saver.restore(self.session, save_path=latest_checkpoint)
            print("\n\nLoad of neural network successful. Please wait.")
        except:
            print("Could not find checkpoint.")

class LinearControlSignal:
    """
    From https://github.com/Hvass-Labs/TensorFlow-Tutorials
    A control signal that changes linearly over time.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value

class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.

    From https://github.com/Hvass-Labs/TensorFlow-Tutorials
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e6,
                 start_value=1.0, end_value=0.1,
                 repeat=False):
        """

        :param num_actions:
            Number of possible actions in the game-environment.

        :param epsilon_testing:
            Epsilon-value when testing.

        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.

        :param start_value:
            Starting value for linearly decreasing epsilon.

        :param end_value:
            Ending value for linearly decreasing epsilon.

        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        """
        Use the epsilon-greedy policy to select an action.

        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.

        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.

        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.

        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)
        print(epsilon)
        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon

class Agent:
    """
    Description:
        Agent that replies to the user simulator. Receives S,R -> Action output
    Parameters:
        training: training or testing. This is for run()
    """

    def __init__(self, training):
        """
        Parameters:
            env_name: Name of the env in OpenAI
            training: Training or testing. This is for env_narun()
        """

        # Create the game-env using OpenAI Gym
        self.env = gym.make('Breakout-v0')

        # Get number of actions from environment
        self.num_actions = self.env.action_space.n

        # Training or testing
        self.training = training

        # Set the initial training epsilon
        # self.epsilon = 0.10

        # Get the number of actions for storing memories and Q-values etc.
        total_actions = self.env.action_space.n

        if self.training:
            # Training: Set the parameters
            self.learning_rate_control = LinearControlSignal(start_value=1e-3,
                                                             end_value=1e-5,
                                                             num_iterations=5e6)

            self.loss_limit_control = LinearControlSignal(start_value=0.1,
                                                          end_value=0.015,
                                                          num_iterations=5e6)

            self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                          end_value=10.0,
                                                          num_iterations=5e6)

            self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                       end_value=1.0,
                                                       num_iterations=5e6)

            # Training: Set up the replay memory
            self.replay_memory = ReplayMemory(size=200000, num_actions=total_actions)
        else:
            # Testing: These are not needed
            self.learning_rate = None
            self.replay_memory = None

        # List of string names for the actions in the env
        self.action_names = self.env.unwrapped.get_action_meanings()

        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=1e6,
                                            num_actions=self.num_actions,
                                            epsilon_testing=0.01)

        # Create the neural network
        self.neural_network = NeuralNetwork(num_actions=total_actions, replay_memory=self.replay_memory)

        # This stores the rewards for each episode
        self.rewards = []

    def get_action_name(self, action):
        # Return the name of an action
        return self.action_names[action]

    def get_lives(self):
        # Get the number of lives the agent has
        return self.env.unwrapped.ale.lives()

    def get_action(self, q_values, count_states):
        """
        Description:
            Use the epsilion greedy to select an action
        Parameters:
            q_values: Q-values at the current state
            count_states: count of processed states
        Return:
            action: the selected action to take in the game
        """
        curr_epsilon = self.epsilon_greedy.get_epsilon(count_states, self.training)

        if np.random.random() < curr_epsilon:
            # Random action
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Select highest Q-value
            action = np.argmax(q_values)

        return action

    def get_testing_action(self, q_values):
        # During testing, always select the max Q-value
        action = np.argmax(q_values)

        return action

    def run(self, num_episodes=None):
        """
        Description:
            Run the agent in either training or testing mode
        Parameters:
            num_episodes: The number of episodes the agent will run in training mode
        """

        if self.training:

            # Don't render
            #self.env.render()

            # Reset following loop
            end_episode = True

            # Counter for states
            count_states = 0

            # Counter for episodes
            count_episodes = 0

            if num_episodes is None:
                # Loop forever by comparing the episode-counter to infinity.
                num_episodes = float('inf')

            while count_episodes <= num_episodes:
                if end_episode:
                    # Reset the game and get first image frame
                    img = self.env.reset()

                    # Create new motion tracer
                    motion_tracer = MotionTracer(img)

                    # Reset episode reward
                    reward_episode = 0

                    # Increment episode counter
                    count_episodes += 1

                    # Number of lives left in this episode
                    num_lives = self.get_lives()

                    if count_episodes > num_episodes:
                        self.neural_network.save(count_states)

                # Get the state
                state = motion_tracer.get_state()

                # Get the Q-values for the state
                q_values = self.neural_network.get_q_values(states=[state])[0]

                # Determine the action
                action = self.get_action(q_values=q_values, count_states=count_states)

                # Take a step using the action
                img, reward, end_episode, info = self.env.step(action=action)

                # Process image from game
                motion_tracer.process(image=img)

                # Add to the reward for this episode
                reward_episode += reward

                # Check if a life was lost
                num_lives_new = self.get_lives()
                end_life = (num_lives_new < num_lives)
                num_lives = num_lives_new

                # Increment the counter for states
                count_states += 1

                # Add to replay memory
                self.replay_memory.add_memory(state=state,q_values=q_values,action=action,reward=reward,end_life=end_life,end_episode=end_episode)

                use_fraction = self.replay_fraction.get_value(iteration=count_states)

                if self.replay_memory.is_full() \
                    or self.replay_memory.used_fraction() > use_fraction:
                    # If the replay memory is full, update all the Q-values in a backwards sweep
                    self.replay_memory.update_q_values()

                    learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                    loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                    max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                    # Improve the policy with random batches from the replay memory
                    self.neural_network.optimize(learning_rate=learning_rate, loss_limit=loss_limit, max_epochs=max_epochs)

                    self.neural_network.save(count_states)

                    # Reset the replay memory
                    self.replay_memory.reset()

                if end_episode:
                # Add the reward of the episode to the rewards array
                    self.rewards.append(reward_episode)

                # Reward from previous episodes (mean of last 30)
                if len(self.rewards) == 0:
                    # No previous rewards
                    reward_mean = 0.0
                else:
                    # Get the mean of the last 30
                    reward_mean = np.mean(self.rewards[-30:])

                if end_episode:
                    # Print statistics
                    statistics = "{0:4}:{1}\tReward: {2:.1f}\tMean Reward (last 30): {3:.1f}\tQ-min: {4:5.7f}\tQ-max: {5:5.7f}"
                    print(statistics.format(count_episodes, count_states, reward_episode, reward_mean, np.min(q_values), np.max(q_values)))

        # TESTING
        else:
            # Load the pre-trained NN
            self.neural_network.load()
            img = self.env.reset()
            motion_tracer = MotionTracer(img)

            while True:
                # Render game environment to screen
                self.env.render()

                # Pause to slow down the game to make the game easier to see
                time.sleep(0.01)

                # Get state
                state = motion_tracer.get_state()

                # Get Q-values
                q_values = self.neural_network.get_q_values(states=[state])[0]

                # Get testing action (max value)
                action=self.get_testing_action(q_values=q_values)

                # Observe output of action
                img, reward, end_episode, info = self.env.step(action=action)

                # Process next state
                motion_tracer.process(image=img)

                # Close if the agent dies
                if(end_episode == True):
                    print("The agent has died. Closing in 10...")
                    time.sleep(10)
                    break

if __name__ == '__main__':
    # Running the system from command line

    # Parsing for command line
    description = "Q-Learning chatbot"

    # Create parser and arguments
    parser = argparse.ArgumentParser(description=description)

    # Training argument: add "-training" to run training
    parser.add_argument("-training", required=False,
                        dest='training', action='store_true',
                        help="train or test agent")

    # Prase the args
    args = parser.parse_args()
    training = args.training

    # Time taken to train system
    start = timer()

    # Create and run agent
    agent = Agent(training=training)
    agent.run()

    # Calculate time taken
    end = timer()
    time_taken = ((end - start)/60)

    # Get the rewards
    rewards = agent.rewards

    # Print statistics about the rewards
    if training:
        print("################################################")
        # Number of episodes
        print("Statistics for {0} episodes".format(len(rewards)))
        # Maximum reward observed
        print("Max:\t\t\t",                 np.max(rewards))
        # Maximum reward instance
        print("Max occurrence:\t\t",          rewards.index(np.max(rewards)))
        # Mean reward for all occurrences
        print("Mean Reward:\t\t",           np.mean(rewards))
        # Minimum reward observed
        print("Min:\t\t\t",                 np.min(rewards))
        # Time taken
        print("Time taken(m):\t\t",          time_taken)

"""        # Plot the total average reward over time
        rewards_plot = []
        for x in range(len(rewards)):
            if (x>0 and (x % 10000 == 0)):
                # This gets the average reward of the last 10000 results
                values = np.mean(rewards[(x-10000):x])
                rewards_plot.append(values)


        # Plot the reward over time
        plt.title("Graph of Total Average Reward Over Time")
        plt.ylabel("Average Reward")
        plt.xlabel("Number of episodes /1000 episodes")
        plt.plot(rewards_plot, 'r')
        plt.show()"""
