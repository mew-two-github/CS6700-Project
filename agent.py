from config import *
import time
import numpy as np

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
            if self.env_name == 'kbca' or self.env_name == 'kbcb':
              self.no_actions = 2
            else:
              self.no_actions = 3
            for i in range(16):
                for a in range(self.no_actions):
                  self.Q.update({str((i, a)) : 0})
         

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
        action = 0
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

          action = info
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


        #raise NotImplementedError
        return action