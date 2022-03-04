import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# create an environment, pass the string of the name of the Frozen Lake environment
environment = gym.make("FrozenLake-v1")

# we can query for info about the environment, sample states & actions, retrieve rewards, and have
# our agent navigate the frozen lake using the variable environment
environment.reset()

# construct q-table, initialize all the key values to 0 for each state-action pair
# number of rows = size of the state space in the environment
# number of columns = size of the action space

# retrieve the size of the state space
state_spaceSize = environment.observation_space.n

# retrieve the size of the action space
action_spaceSize = environment.action_space.n

# build a q-table using state_spaceSize & action_spaceSize, fill it with 0's
q_tabl = np.zeros((state_spaceSize, action_spaceSize))

# create and initialize all necessary parameters for q-learning implementation

# the total number of episodes we want our agent to play during training 10000
no_episodes = 10000

# the maximum number of steps the agent's allowed to take within a single episode
# If by the nth step, the agent has not reached the frisbee or fallen through a hole,
# then the running episode will terminate with the agent receiving 0 points.
max_no_steps_per_episode = 100

# learning rate 0.1
alpha = 0.1
# discount rate 0.99
gamma = 0.999

# set of alpha-gamma value pairs to test :(0.1, 0.99), (0.3, 0.98), (), (), (), ().

# exploration rate (try 0) 1
epsilon = 0.95
# maximum exploration rate, upper bound value of epsilon: 1
max_epsilon = 1
# minimum exploration rate, lower bound value of epsilon
min_epsilon = 0.01
# exploration decay rate to determine the rate at which the exploration rate will decay
# changed from 0.01 to 0.001
epsilon_decay_rate = 0.001

# list to hold all the rewards we'll obtain from running each episode
# it helps us see how our game score changes over time and evaluate our parameter assignments
all_episodes_rewards = []

# for each single episode
for episode in range(no_episodes):
    init_state = environment.reset()
    # reset the state of the environment back to the starting, initial state #
    env_state = init_state

    # keep track of whether the current episode is finished
    episode_finished = False
    # keep track of the rewards within a currently running episode
    current_episode_rewards = 0
    # runs for each step within a single episode
    for time_step in range(max_no_steps_per_episode):
        # set exploration rate threshold to a random number between 0 and 1 to determine whether
        # the agent should explore or exploit the environment in this step of the episode
        epsilon_rate_threshold = random.uniform(0, 1)
        # if the threshold is greater than the exploration rate, agent exploits the environment
        if epsilon_rate_threshold > epsilon:
            # agent will exploit the environment, and choose the action that has the highest q_value
            # in the q_table for the current state
            agent_action = np.argmax(q_tabl[env_state, :])
        else:  # threshold smaller than or equal to the exploration rate
            # the agent explores the environment and sample a random action
            agent_action = environment.action_space.sample()
        # Once we're done choosing our action, we pass that action object as an argument for the .step() method of
        # the environment, which returns a tuple containing the new state resulted from the action being taken, the
        # reward for the action the agent took, whether or not the action ended the episode, & some diagnostic info.
        new_env_state, taken_action_reward, episode_finished, diagnostic_information = environment.step(agent_action)

        # Using the reward obtained from taking the action from the previous state, we update the Q-value for the
        # associated state-action pair in the Q-table using the Q-learning formula.
        # q_tabl[env_state, agent_action] = old q-value
        # (taken_action_reward + gamma * np.max(q_tabl[new_env_state, :]) = learned value
        q_tabl[env_state, agent_action] = q_tabl[env_state, agent_action] * (1 - alpha) + \
                                alpha * (taken_action_reward + gamma * np.max(q_tabl[new_env_state, :]))

        # update the current state to the new state that was returned once the agent took its last action
        env_state = new_env_state
        # update the rewards from the current episode by adding the reward received from the agent's previous action
        current_episode_rewards = current_episode_rewards + taken_action_reward

        # check if the current episode was ended by the agent's last action, meaning did the agent falls into a hole or
        # reach the goal. If so, the current episode is over, and we move onto the next episode.
        if episode_finished:
            break
    # update exploration rate using exponential decay (the exploration rate decreases or decays at a rate proportional
    # to its current value). Decay the exploration rate using the following formula.
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)

    # Append the rewards for the current episode to the list of rewards from all episodes
    all_episodes_rewards.append(current_episode_rewards)

# Calculate the average reward per a thousand episodes
avg_rewards_per_thousand_episodes = np.split(np.array(all_episodes_rewards), no_episodes/1000)
count = 1000
print("********Average reward per one thousand episodes********\n")
for reward in avg_rewards_per_thousand_episodes:
    print(count, ": ", str(sum(reward/1000)))
    count += 1000

print("\n\n*****Q-table*****\n")
print(q_tabl)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
