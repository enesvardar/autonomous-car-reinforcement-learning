# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:55:52 2021

@author: enesv
"""
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from os import path
import datetime

class DQLAgent:
    
    def __init__(self,_state_size,_action_size):
        
        self.state_size = _state_size
        self.action_size = _action_size    
        self.gamma = 0.95
        self.learning_rate = 0.001 
        self.model = self.build_model()
        self.memory1 = deque(maxlen = 5000) # for done == 1 (game over)
        self.memory2 = deque(maxlen = 5000) # for done == 0
        self.epsilon = 50 
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.loss = 0
        self.loadTrainData()
    
    """
      @brief neural network for deep q learning
    """
    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(64, activation = "relu"))
        model.add(Dense(self.action_size,activation = "linear"))
        
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop",
        )
        
        model.compile(loss = "mse", optimizer = optimizer,metrics=['acc'])       
        model.summary()
        return model
    
    def saveModel(self):
        
        an = datetime.datetime.now()
        date = str(an.year) + '' + str(an.month) + '' + str(an.day) + '' + str(an.hour) + '' + str(an.minute)
        self.model.save_weights('saved_models/' + date + '_my_model.h5')
    
    
    """
      @brief It is used to read pre-recorded training data.
    """
    def loadTrainData(self):
        
        N = self.state_size
       
        for k in range(2):
        
            if k == 0:                
                file = 'logs/trainDn0.txt'                
            else:                
                file = 'logs/trainDn1.txt'
             
            #                            traning file format    
            # |------------state----------|               |--------next--state--------| done
            # L1 L2 L3 L4 L5 C1 C2 C3 C4 C5 action reward L1 L2 L3 L4 L5 C1 C2 C3 C4 C5  0/1  
             
            if(path.exists(file)):
            
                with open(file) as f:
                    lines = f.readlines()                
                count = 0                
                for line in lines:                    
                    if(count == 5000): # size of stroge 5000                      
                        break
                    
                    count += 1
                    data = line.split()
                   
                    if len(data) == N + 10:                                            
                        state = []                        
                        next_state = []
                        
                        for i in range(N):                            
                            state.append(float(data[i]))
                            
                        state = np.array(state).reshape(1,self.state_size)                        
                        action = int(data[N])                        
                        reward = float(data[N + 1])
                        
                        for i in range(N):                            
                            next_state.append(float(data[i + N + 2]))                            
                        next_state = np.array(next_state).reshape(1,self.state_size)
                            
                        if k == 0:                            
                            done = 0
                            self.memory1.append({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':done})                            
                        else:                            
                            done = 1
                            self.memory2.append({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':done})
                            
    def remember(self, state, action, reward, next_state, done):
        # storage
        if(done == 0):
            self.memory1.append({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':done})            
        else: 
            self.memory2.append({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':done})
    
    """
      @brief Given the state, select an action.
      @param 
          state: the current state of the environment
          replay: training flag
      @returns
          action: an integer, compatible with the task's action space
    """             
    def act(self, state, replay):
                
        if np.random.rand() <= self.epsilon and replay == True:          
    
            return random.randrange(self.action_size) # random action
                    
        act_values = self.model.predict(state) # model action                           

        return np.argmax(act_values[0]) 
    
    """
      @brief model training
      @param 
          batchSize: she number of training examples in one forward / backward pass. 
    """
    def replay(self,batchSize):
        
        i = 0;
        
        if len(self.memory1) + len(self.memory2) < batchSize:     
            return
        
        N = int(len(self.memory2) % (batchSize / 2))        
        M = int(batchSize - N)
                                
        minibatch = random.sample(self.memory1,M)               
        minibatch.extend(random.sample(self.memory2,N))
        
        state_t = np.array([ sub['state'] for sub in minibatch ]).reshape(batchSize, self.state_size) # get state data         
        state_t1 = np.array([ sub['next_state'] for sub in minibatch ]).reshape(batchSize, self.state_size) # get next_state data   
        
        target = self.model.predict(state_t)
        Qs = self.model.predict(state_t1)
        
        # Q(s,a) = r(s,a) + γ maxQ(s’,a)
        
        for sub in minibatch:
            
            reward = sub['reward']
            action = sub['action']
            
            if sub['done']:                
                target[i][action] = reward                
            else:
                target[i][action] = reward + self.gamma*np.amax(Qs[i])
                        
            i = i + 1

        history  =  self.model.fit(state_t, target,batch_size=batchSize,epochs=1,verbose=1)
        self.loss = history.history['loss'][0]
        self.adaptiveEGreedy() 
        
    """
      @brief The epsilon value is reduced, allowing the agent to explore sufficiently. (random action)
    """
    def adaptiveEGreedy(self):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
  
        

