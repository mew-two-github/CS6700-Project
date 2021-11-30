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

        # Loading hyperparameters
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.epsilon = self.config["epsilon"]
        self.no_actions = self.config["no_actions"]
        self.previous_action = 0

        # Initialising empty Q table
        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':
            self.Q = {}
            for i in range(16):
                for a in range(self.no_actions):
                  self.Q.update({str((i, a)) : 0})
            self.previous_state = 0

            

        elif self.env_name == 'acrobot':

          self.theta = np.random.randn(4,3)
          self.previous_state = np.zeros(shape=(4))
          self.no_actions = 3
          self.X = np.zeros(shape=(4))
          self.A = np.zeros(shape=(1))
          self.episode = 0
          self.value = np.zeros(shape=(0))
          self.beta = 0.0001
          self.alpha = 0.99
          self.grads = []

        else:
          self.Q = {}
          for i in range(501):
            for a in range(self.no_actions):
              self.Q.update({str((i,a)) : 0})
          self.prev = 1
    
    def p_acro(self,state):
      e = np.exp(state.dot(self.theta))
      p = e/e.sum()
      return p
    
    def pg_acro(self,X,A):
      p = self.p_acro(X)
      mat = p.reshape(-1,1)
      dgibbs = np.diagflat(mat) - np.dot(mat, mat.T)
      dgibbs = dgibbs[A+1,:]
      dlog = dgibbs/p[A+1]
      pg = X.reshape(1,-1).T.dot(dlog[None,:])
      return pg

    def state_from_obs(self,obs):

      theta1 = np.arctan2(obs[1],obs[0])
      theta2 = np.arctan2(obs[3],obs[2])

      thetadot1 = obs[4]
      thetadot2 = obs[5]

      state = np.array([theta1,theta2,thetadot1,thetadot2])
      return state

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
          state = 0
          action = 1
          self.previous_state = state
          self.previous_action = action
        
        elif self.env_name == 'acrobot':
          for i in range(len(self.value)): 
            self.theta += self.beta*self.value[i]*self.grads[i]
          
          #if self.episode%10 == 0:
            #print(self.theta)
            #print(self.value)
            #print(self.A)
            # print(self.value[i]*self.pg_acro(self.X[i,:],self.A[i]))
            # print(self.A[i])
            # print(self.p_acro(self.X[i,:]))
            #print((self.X[i,:],self.A[i]))
              
          self.X = self.state_from_obs(obs)
          p = self.p_acro(self.X)
          action = np.random.choice([-1,0,1],1,p=p)[0]
          self.A = np.array([action])
          self.value = np.zeros(shape=(0))
          self.grads = []

          self.episode += 1
          

        else:
          state = obs
          if np.random.uniform(low=0.0, high=1.0, size=None) > (self.epsilon):
            temp = -np.inf
            for a in range(self.no_actions):
              if self.Q[str((state,a))] >= temp:
                action = a
                temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(self.no_actions)
          self.previous_state = state
          self.previous_action = action
        

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

        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':
          if done == False:
              count = 0
              for i in range(16):
                if obs[i] != '':
                  count = count+1
              next_state = count
              temp = -np.inf  
              for b in range(self.no_actions):
                if self.Q[str((next_state,b))] >= temp:
                  temp = self.Q[str((next_state,b))]
              self.Q[str((self.previous_state,self.previous_action))] = (1 - self.beta)*self.Q[str((self.previous_state,self.previous_action))] + self.beta*(reward+self.alpha*temp)

              state = next_state
              if np.random.uniform(low=0.0, high=1.0, size=None) > (self.epsilon):
                temp = -np.inf
                for a in range(self.no_actions):
                  if self.Q[str((state,a))] >= temp:
                      action = a
                      temp = self.Q[str((state,a))]
              else:
                action = np.random.randint(self.no_actions)

          else:
              self.Q[str((self.previous_state,self.previous_action))] = (1 - self.beta)*self.Q[str((self.previous_state,self.previous_action))] + self.beta*(reward)
              state = 0
              action = 0
          self.previous_state = state
          self.previous_action = action

        elif self.env_name == 'acrobot':
          state = self.state_from_obs(obs)
          self.X = np.vstack([self.X,state])

          p = self.p_acro(state)
          #print(p,state)
          action = np.random.choice([-1,0,1],1,p=p)[0]
          self.A = np.hstack([self.A,action])
          grad = self.pg_acro(state,action)
          self.grads.append(grad)
          self.value += self.alpha*reward
          self.value = np.hstack([self.value,[reward]])

        else:
          next_state = obs
          temp = -np.inf
          for b in range(self.no_actions):
            if self.Q[str((next_state,b))] >= temp:
              temp = self.Q[str((next_state,b))]
          self.Q[str((self.previous_state,self.previous_action))] = (1 - self.beta)*self.Q[str((self.previous_state,self.previous_action))] + self.beta*(reward+self.alpha*temp)

          state = next_state
          if np.random.uniform(low=0.0, high=1.0, size=None) > (self.epsilon):
            temp = -np.inf
            for a in range(self.no_actions):
              if self.Q[str((state,a))] >= temp:
                action = a
                temp = self.Q[str((state,a))]
          else:
            action = np.random.randint(self.no_actions)
          self.previous_state = state
          self.previous_action = action

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
        if self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc':
          action = 1
          state = 0
        elif self.env_name == 'taxi':
          state = obs
          temp = -np.inf
          for a in range(self.no_actions):
            if self.Q[str((state,a))] >= temp:
              action = a
              temp = self.Q[str((state,a))]
        elif self.env_name == 'acrobot':
          state = self.state_from_obs(obs)
          p = self.p_acro(state)
          action = np.random.choice([-1,0,1],1,p=p)[0]

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
            if self.Q[str((state,a))] >= temp:
              action = a
              temp = self.Q[str((state,a))]

        
        elif self.env_name == 'acrobot':
          state = self.state_from_obs(obs)
          p = self.p_acro(state)
          action = np.random.choice([-1,0,1],1,p=p)[0]

        else:
          state = obs
          temp = -np.inf
          for a in range(self.no_actions):
            if self.Q[str((state,a))] >= temp:
              action = a
              temp = self.Q[str((state,a))]
        #raise NotImplementedError
        return action