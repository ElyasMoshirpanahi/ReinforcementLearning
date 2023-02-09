import gym
import numpy as np
from gym import wrappers
import pybullet_envs
from tqdm import tqdm
# Hyperparameters class
class hyper_parameters():
    def __init__(self):
        self.number_of_steps = 1000
        self.episode_lenght = 1000
        self.learning_rate = 0.02
        self.number_of_directions = 16
        self.number_of_best_directions = 16
        assert self.number_of_best_directions <= self.number_of_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'

# Normalizer class
class Normalizer():
    def __init__(self, number_of_inputs):
        self.n = np.zeros(number_of_inputs)
        self.mean = np.zeros(number_of_inputs)
        self.mean_diff = np.zeros(number_of_inputs)
        self.variance = np.zeros(number_of_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        observed_mean = self.mean
        observed_std = np.sqrt(self.var)
        return (inputs - observed_mean) / observed_std

# Policy class
class Policy:
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(input)
        else:
            return (self.theta - hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for i in range(hp.number_of_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_positive, r_negative, d in rollouts:
            step += (r_positive - r_negative) * d

        self.theta += hp.learning_rate / (hp.number_of_best_directions * sigma_r) * step


#Exploring the policy on one sepcific direction and over one episode
def explore(env,normalizer, policy, direction = None , delta =None):
    state = env.reset()
    done  = False
    num_plays = 0.
    sum_rewards = 0.
    
    while not done and num_plays < hp.episode_lenght:
        normalizer.observe(state)
        state  = normalizer.normalize(state)
        action = policy.evaluate(state , delta ,direction)
        state , reward , done, _ =env.step(action)
        reward  = max(min(reward,1), -1 )#Important to get the outliers
        sum_rewards  += reward
        num_plays  +=1
    return sum_rewards

#Training AI
def train(env, policy, normalizer, hp):
    for step in tqdm(range(hp.number_of_steps)):
        # Initializing the perturbation deltas and positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.number_of_directions
        negative_rewards = [0] * hp.number_of_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.number_of_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction="positive", delta=deltas[k])
          
        # Getting the negative rewards in the negative directions
        for k in range(hp.number_of_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction="negative", delta=deltas[k])
        
        # Gathering all rewards to compute the standard deviation of rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Scoring the rollouts by max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x])[:hp.number_of_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating the policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print(f"Step: {step} Reward: {reward_evaluation}")
         
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

working_dir = mkdir('./exp', 'brs')
monitor_dir = mkdir(working_dir, 'monitor')

hp = hyper_parameters()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force=True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(input_size=nb_inputs, output_size=nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)