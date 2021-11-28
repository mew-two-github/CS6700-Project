from config import *
import time
import numpy as np
import math

"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""

class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]

        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':
            self.Q = {}
            if self.env_name == 'kbca':
              self.alpha = 0.05
              self.beta = 0.7
              self.epsilon = 0.25
            elif self.env_name == 'kbcb':
              self.alpha = 0.01
              self.beta = 0.7
              self.epsilon = 0.3
            elif self.env_name == 'kbcc':
              self.alpha = 0.1
              self.beta = 0.5
              self.epsilon = 0.05
              self.previous_action = 0
            if self.env_name == 'kbca' or self.env_name == 'kbcb':
              self.no_actions = 2
            else:
              self.no_actions = 3
            for i in range(16):
                for a in range(self.no_actions):
                  self.Q.update({str((i, a)) : 0})
        elif self.env_name == 'acrobot':
          self.alpha = 1
          self.beta = 0.2
          self.epsilon = 0
          self.Q = {}
          # 3 actions: -1, 0, 1
          self.no_actions = 3
          # Number of discrete states theta and thetadot are split into
          self.buckets = 5
          for i in range(self.buckets):
            for j in range(self.buckets):
              for k in range(self.buckets):
                for l in range(self.buckets):
                  for a in range(self.no_actions):
                    state = str(i) + str(j) + str(k) + str(l)
                    self.Q.update({str((state, a)) : 0})
          #print(self.Q.keys())
          self.previous_state = state
          self.previous_action = a
         

    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = 1
        #Generating state from observation
        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':        
          count = 0
          for i in range(16):
            if obs[i] != "":
              count = count+1
          state = count
          temp = -np.inf
          if np.random.uniform(low=0.0, high=1.0, size=None) > (1-self.epsilon):
            for a in range(self.no_actions):
              if self.Q[str((state,a))] > temp:
                  action = a
                  temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(self.no_actions)
          if self.env_name == 'kbcc':
            self.previous_action = action
        elif self.env_name == 'acrobot':
          # Finding state from given observation
          theta1 = np.arctan(obs[1]/obs[0])
          theta2 = np.arctan(obs[3]/obs[2])
          thetadot1 = obs[4]
          thetadot2 = obs[5]
          bucket1 = min(math.floor((theta1 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket2 = min(math.floor((theta2 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket3 = min(math.floor((thetadot1 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          bucket4 = min(math.floor((thetadot2 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          state = str(bucket1) + str(bucket2) + str(bucket3) + str(bucket4)
          temp = -np.inf
          if np.random.uniform(low=0.0, high=1.0, size=None) > (1-self.epsilon):
            for a in range(self.no_actions):
              if self.Q[str((state,a))] > temp:
                  action = a - 1
                  temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(self.no_actions) - 1
          self.previous_state = state
          self.previous_action = action + 1



        #raise NotImplementedError
        return action

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = 1

        if self.env_name == 'kbca' or self.env_name == 'kbcb':
          count = 0
          for i in range(16):
            if obs[i] != "":
              count = count+1
          next_state = count

          if done == "True" and obs[count-1]==1:
            state = next_state
            action = 0
          elif done == "True" and obs[count-1]==0:
            state = next_state-1
            action = 1
          else:
            state = next_state-1  
            action = 1

          temp = -np.inf

          for b in range(2):
            if self.Q[str((next_state,b))] > temp:
              temp = self.Q[str((next_state, b))]
          self.Q[str((state, action))] = (1 - self.beta)*self.Q[str((state,action))] + self.beta*(reward+self.alpha*temp)

          state = next_state
          temp = -np.inf
          if np.random.uniform(low=0.0, high=1.0, size=None) > (1-self.epsilon):
            for a in range(2):
              if self.Q[str((state,a))] > temp:
                  action = a
                  temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(2)  

        elif self.env_name == 'kbcc':
          count = 0
          for i in range(16):
            if obs[i] != "":
              count = count+1
          next_state = count

          action = self.previous_action
          if done == "True" and action == 0:
            state = next_state
          elif done == "True" and action == 1:
            state = next_state-1
          elif done == "True" and action == 2:
            state = next_state-1
          else:
            state = next_state - 1  
          

          temp = -np.inf

          for b in range(3):
            if self.Q[str((next_state,b))] > temp:
              temp = self.Q[str((next_state, b))]
          self.Q[str((state,action))] = (1 - self.beta)*self.Q[str((state,action))] + self.beta*(reward+self.alpha*temp)

          state = next_state
          temp = -np.inf
          if np.random.uniform(low=0.0, high=1.0, size=None) > (1-self.epsilon):
            for a in range(3):
              if self.Q[str((state,a))] > temp:
                  action = a
                  temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(3)  
          self.previous_action = action
        elif self.env_name == 'acrobot':

          # Finding state from given observation
          theta1 = np.arctan(obs[1]/obs[0])
          theta2 = np.arctan(obs[3]/obs[2])
          thetadot1 = obs[4]
          thetadot2 = obs[5]
          bucket1 = min(math.floor((theta1 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket2 = min(math.floor((theta2 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket3 = min(math.floor((thetadot1 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          bucket4 = min(math.floor((thetadot2 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          next_state = str(bucket1) + str(bucket2) + str(bucket3) + str(bucket4)
          # Storing previous state and action for Q value update
          state = self.previous_state
          a = self.previous_action
          # Find best action in current state and update Q value
          temp = -np.inf
          for b in range(self.no_actions):
            if self.Q[str((next_state,b))] > temp:
              temp = self.Q[str((next_state, b))]
          self.Q[str((state, a))] = (1 - self.beta)*self.Q[str((state,a))] + self.beta*(reward+self.alpha*temp)

          # Finding new action using epsilon-greedy
          state = next_state
          temp = -np.inf
          if np.random.uniform(low=0.0, high=1.0, size=None) > (1-self.epsilon):
            for a in range(self.no_actions):
              if self.Q[str((state,a))] > temp:
                  action = a - 1
                  temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(self.no_actions)  - 1
          self.previous_action = action + 1 # since allowed actions are -1,0,1 and key of dictionary is of the the form 0,1,2
          self.previous_state = next_state


        #raise NotImplementedError
        return action

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = 0
        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':       
          count = 0
          for i in range(16):
            if obs[i] != "":
              count = count+1
          state = count
          temp = -np.inf
          for a in range(self.no_actions):
            if self.Q[str((state,a))] > temp:
              action = a
              temp = self.Q[str((state,a))]
        elif self.env_name == 'acrobot':
          # Finding state from given observation
          theta1 = np.arctan(obs[1]/obs[0])
          theta2 = np.arctan(obs[3]/obs[2])
          thetadot1 = obs[4]
          thetadot2 = obs[5]
          bucket1 = min(math.floor((theta1 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket2 = min(math.floor((theta2 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket3 = min(math.floor((thetadot1 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          bucket4 = min(math.floor((thetadot2 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          state = str(bucket1) + str(bucket2) + str(bucket3) + str(bucket4)
          # Pick the best action
          temp = -np.inf
          for a in range(self.no_actions):
            if self.Q[str((state,a))] > temp:
                action = a - 1
                temp = self.Q[str((state,a))]
          self.previous_action = action + 1 # since allowed actions are -1,0,1 and key of dictionary is of the the form 0,1,2


        #raise NotImplementedError
        return action

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = 0
        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':
          count = 0
          for i in range(16):
            if obs[i] != "":
              count = count+1
          state = count
          temp = -np.inf
          for a in range(self.no_actions):
            if self.Q[str((state,a))] > temp:
              action = a
              temp = self.Q[str((state,a))]
        elif self.env_name == 'acrobot':
          # Finding state from given observation
          theta1 = np.arctan(obs[1]/obs[0])
          theta2 = np.arctan(obs[3]/obs[2])
          thetadot1 = obs[4]
          thetadot2 = obs[5]
          bucket1 = min(math.floor((theta1 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket2 = min(math.floor((theta2 + math.pi/2)/(math.pi/self.buckets)),self.buckets-1)
          bucket3 = min(math.floor((thetadot1 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          bucket4 = min(math.floor((thetadot2 + 12.57)/(2*12.57/self.buckets)),self.buckets-1)
          state = str(bucket1) + str(bucket2) + str(bucket3) + str(bucket4)
          # Finding best action
          temp = -np.inf
          for a in range(self.no_actions):
            if self.Q[str((state,a))] > temp:
                action = a - 1
                temp = self.Q[str((state,a))]
        #raise NotImplementedError
        return action