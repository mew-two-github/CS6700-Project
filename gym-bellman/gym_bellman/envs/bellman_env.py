import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random

class BellmansDPBase(gym.Env):


  """
  Assume that you are a contestant at a quiz program, where you have an opportunity to win substantial amount of money by answering a series of questions. 
  These are the rules of the constant

  1. At each stage of the contest, you are allowed to chose between 2 actions:
      - Pull out and receive reward
      - Answer a question
  2. A reward r_i is received at the end of the contest. "i" is the max question reached before termination. The reward at all preceeding stages is 0.
  3. The outcome of a question "i" is determined by a probability "p_i". Note that "p_i > p_{i+1}". The probabilty p_i is not visible to the agent.
  4. There are a maximum of N questions(N=16). 
  6. There are a maximum of 16 questions. You receive full reward if the 16th question is reached. 
  7. If you are terminated after answering a question wrong, your reward is 0. 


  Actions:
    0 - Pull out and receive reward
    1 - Answer the next question
  
  Observations:
    A list of outcomes of all previous questions. 1 indicates answered correctly, 0 indicates answered wrong, and "" indicates unanswered
    [1,1,1,1,1,0,"","","","","","","","","",""]

  Rewards:
    - Default reward of 0 at each step
    - Reward at last step is 0 if terminated on an easy question
    - Reward at last step is reward_i if pulled out after ith question
    - Reward at last step is reward_N if Nth question is reached

  Termination criteria:
    - Nth question is reached (N=16)
    - Last action is 0
    - Last question is wrongly answered

  KBC A:
  Assume an environment with the above rules, model an agent that maximises the outcome. 

  KBC B:
  Assume there is a checkpoint question K, after which you are guarenteed a reward of r_k, irrespective of what happens later. Model an agent under this condition. 
  
  KBC C:
  Additional rules:

  - At each stage of the question you now have an opportunity to choose between an easy and a hard question. 
  - Answering an easy question wrong terminates with a 0 reward, like KBC A. 
  - Answer a hard question wrong terminates with a reward of r_i/2. 
  - Note that p_i_{easy} > p_i_{hard}
  - Model an RL agent under these conditions. 


  Actions:
    0 - Pull out and receive reward
    1 - Answer an easy question
    2 - Answer a hard question
  
  """

  def __init__(self):
    
    self.N = 16
    self.k = 0

    # Prob of answering the next question is a GP (depending on hard or easy). p_i = p_{easy/hard} * (gamma{easy/hard} ^ i)
    self.easy_prob_init = 0.99
    self.hard_prob_init = 0.99
    self.easy_gamma = 0.95
    self.hard_gamma = 0.85

    # Net reward at the end of i'th question = 1000 * (2^i)
    self.rewards = [1000 * (2**i) for i in range(self.N)]

    self.rewards.insert(0,0)
    self.action_space = spaces.Discrete(2)

  def reset(self):
    self.wrong_count = 0 
    self.wrong_tolerance = 0
    self.step_count = 0
    self.num_correct_answer = 0
    self.history = ["" for i in range(self.N)]
    self.action_history = ["" for i in range(self.N)]
    return self.history

  def compute_total_reward(self):

    reward = self.rewards[self.step_count]

    # Return 0 if the participant terminates in the beginning
    if self.step_count == 0:
      return 0

    # Return full reward if the participant reaches the last question 
    if self.step_count == self.N-1 and self.wrong_count <= self.wrong_tolerance:
      return reward

    # Return 0 reward if last question was easy
    if self.action_history[self.step_count] == 1:
      
      if self.num_correct_answer >= self.k:
        reward = self.rewards[self.k]
        return reward
      
      return 0

    # Return reward/2 if last question was hard
    if self.action_history[self.step_count] == 2:
      return reward/2

    # Return full reward if the participant choses to quit
    else:
      return reward
  
  def step(self, action):

    obs = self.history
    if action not in self.action_space:
      action = 0

    self.action_history[self.step_count] = action
    if action !=1 and action != 2:
      reward = self.compute_total_reward()
      done = True
      self.history[self.step_count] = -1
      return obs, reward, done, {}

    correct_prob = self.easy_prob_init * (self.easy_gamma ** self.step_count) if action == 1 else self.hard_prob_init * (self.hard_gamma ** self.step_count)
    answer_status = random.random() < correct_prob
    self.wrong_count = self.wrong_count + 1 if not answer_status else self.wrong_count
    self.num_correct_answer = self.num_correct_answer + 1 if answer_status else self.num_correct_answer
    done = False if self.wrong_count <= self.wrong_tolerance and self.step_count+1 < self.N  else True
    self.history[self.step_count] = 1 if answer_status else 0
    reward = self.compute_total_reward() if done else 0
    self.step_count += 1
    return obs, reward, done, {}


class BellmanDpA(BellmansDPBase):

  def __init__(self):
    super().__init__()


class BellmanDpB(BellmansDPBase):

  def __init__(self):
    super().__init__()
    self.k = 6


class BellmanDpC(BellmansDPBase):

  def __init__(self):
    super().__init__()
    self.action_space = spaces.Discrete(3)
