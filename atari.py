## Keelan Atari NN
###############################################################
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
# Checkpoint dir
checkpoint_directory = '/home/keelan/ai/Atari/Checkpoints'
###############################################################
"""
105 x 80 input: pre-processed images from environment gray-scaled and resized
"""

# Height and width
state_height = 105
state_width = 80

# Size of the images, number of images in the state and shape
state_image_size = np.array([state_height, state_width])
state_channels = 2
state_shape = [state_height, state_width, state_channels]

def _rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return img_gray

def pre_process_image(image):
    """Pre-process raw image from game environment into a state"""

    # Convert to gray-scale
    img = _rgb_to_grayscale(image)

    # Resize
    img = scipy.misc.imresize(img, size=state_image_size, interp='bicubic')

    return img

###############################################################


#### FINISHED ####
class MotionTracer:
    """
    Processes raw images from the game.

    Uses last 2 images from the game environment (DeepMind = 4) to detect motion
    https://github.com/Hvass-Labs/TensorFlow-Tutorials
    """

    def __init__(self, image, decay = 0.75):
        """
        Parameters:
            image: First image from game environment (reset)
            decay: Tail-length of motion trace
        """

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

#### FINISHED ####
#### Can change discount rate ####
class ReplayMemory:
    """
    This class holds many previous states of the environment.
    """

    def __init__(self, size, num_actions, discount_rate = 0.8):
        """
        Parameters:
            size: size of the replay memory (states).
            num_actions: Number of possible actions in the environment.
            discount_rate: The discount factor for updating Q-values.
        """

        # Previous states array
        ### float??
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

    def is_full(self):
        # Used to check if the replay memory is full

        return self.current_size == self.size

    def reset_size(self):
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
        #self.rewards[curr] = np.clip(reward, -1.0, 1.0)

    def update_q_values(self):
        # Update the Q-values in the replay memory

        # Keep old Q-values
        self.old_q_values[:] = self.q_values[:]

        # Update Q-values in a backwards loop
        for curr in np.flip(range(self.current_size-1),0):

            # Get data from curr
            action = self.actions[curr]
            reward = self.rewards[curr]
            end_life = self.end_life[curr]
            end_episode = self.end_episode[curr]

            # Calculate Q-value
            if end_life or end_episode:
                # No future steps therefore it is just the observed reward
                value = reward
            else:
                # Discounted future rewards
                value = reward + self.discount_rate * np.max(self.q_values[curr+1])

            # Update Q-values with better estimate
            self.q_values[curr, action] = value

    def get_batch_indices(self, batch_size):
        # Get random indices from the replay memory (number = batch_size)

        self.indices = np.random.choice(self.current_size, size=batch_size, replace=False)

    def get_batch_values(self):
        # Get the states and Q-values for these indices

        batch_states = self.states[self.indices]
        batch_q_values = self.q_values[self.indices]

        return batch_states, batch_q_values


#### NOT FINISHED ####?
class NeuralNetwork:
    """
    This implements the neural network for Q-learning.
    The neural network estimates Q-values for a given state so the agent an decide which action to take.
    """

    def __init__(self, num_actions, replay_memory):

        """
        Parameters:
            num_actions: The number of actions (number of Q-values needed to estimate)
            replay_memory: This is used to optimize the neural network and produce better Q-values
        """

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

        # Initialise weights close to 0 ********* 1e-2 -> 2e-2 -> 1e-3
        weights = tf.truncated_normal_initializer(mean=0.0, stddev=1e-1)

        # CNN Padding
        padding = 'SAME'

        # Activation function for the layers
        activation = tf.nn.relu

        # CNN Layer 1
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

        # Get the less
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

    def optimize(self, current_state, learning_rate, batch_size=128):
        """
        Description:
            This is the optimization function for the neural network. This updates the Q-values
            from a random batch using the learning rate.
        Parameters:
            current_state: The current state being processed when the optimization function is called
            learning_rate: The learning rate of the neural network
            batch_size: The size of the batch taken from the replay memory
        """
        print("Optimization of Neural Network in progress with learning rate {0}".format(learning_rate))

        # Get random indices from the replay memory
        self.replay_memory.get_batch_indices(batch_size)

        # Get the corresponding states and Q-values for the indices
        batch_states, batch_q_values = self.replay_memory.get_batch_values()

        # Feed these values into the neural network and run one optimization and get the loss value
        current_loss, _ = self.session.run([self.loss, self.optimizer], feed_dict = {self.states: batch_states, self.q_values_new: batch_q_values, self.learning_rate: learning_rate})

        # Send the results to tensorboard
        #result = self.session.run(self.merged, feed_dict={self.states: batch_states, self.q_values_new: batch_q_values, self.learning_rate: learning_rate})
        #print("Current loss: ", current_loss)
        #self.writer.add_summary(result, current_state)

    def save(self, count_states):
        # Save the completed trained network
        self.saver.save(self.session, save_path=self.checkpoint_dir, global_step=count_states)
        print("Checkpoint saved")

    def load(self):
        # Load the network for testing
        try:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_directory)
            self.saver.restore(self.session, save_path=latest_checkpoint)
            print("\n\nLoad of neural network successful. Please begin to talk to the chatbot!:")
        except:
            print("Could not find checkpoint")

class Agent:
    """
    Agent that replies to the user simulator
    """

    def __init__(self, training, render=False):
        """
        Parameters:
            env_name: Name of the env in OpenAI
            training: Training or testing. This is for env_narun()
            render: Whether to render the game to the screen or not
        """

        # Create the game-env using OpenAI Gym
        self.env = gym.make('Breakout-v0')

        self.num_actions = self.env.action_space.n

        # Training or testing
        self.training = training

        # Set the initial training epsilon
        self.epsilon = 0.10

        # Render to screen or not
        self.render = render

        # Get the number of actions for storing memories and Q-values etc.
        total_actions = self.env.action_space.n

        if self.training:
            # Training: Set a learning rate
            self.learning_rate = 1e-2

            # Training: Set up the replay memory
            self.replay_memory = ReplayMemory(size=1000, num_actions=total_actions)
        else:
            # Testing: These are not needed
            self.learning_rate = None
            self.replay_memory = None

        # List of string names for the actions in the env
        self.action_names = self.env.unwrapped.get_action_meanings()

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

    def get_action(self, q_values):
        """
        Description:
            Use the epsilion greedy to select an action
        Parameters:
            q_values: Q-values at the current state
        Return:
            action: the selected action to take in the game
        """

        if np.random.random() < self.epsilon:
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

    def run(self, num_episodes=100):
        """
        Description:
            Run the agent in either training or testing mode
        Parameters:
            num_episodes: The number of episodes the agent will run in training mode
        """

        if self.training:

            # Reset following loop
            end_episode = True

            # Counter for states
            count_states = 0

            # Counter for episodes
            count_episodes = 0

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
                action = self.get_action(q_values=q_values)

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

                if self.replay_memory.is_full():
                    # If the replay memory is full, update all the Q-values in a backwards sweep
                    self.replay_memory.update_q_values()

                    # Improve the policy with random batches from the replay memory
                    self.neural_network.optimize(learning_rate=self.learning_rate, current_state=count_states)

                    # Reset the replay memory
                    self.replay_memory.reset_size()

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

            if self.render:
                # Render game env to the screen
                self.env.render()

                # Pause to slow down the game to make the game easier to see
                time.sleep(0.01)



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
        print("Max occurrence:\t",          rewards.index(np.max(rewards)))
        # Mean reward for all occurrences
        print("Mean Reward:\t\t",           np.mean(rewards))
        # Minimum reward observed
        print("Min:\t\t\t",                 np.min(rewards))
        # Time taken
        print("Time taken(m):\t\t",          time_taken)
