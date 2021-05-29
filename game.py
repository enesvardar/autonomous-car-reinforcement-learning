# -*- coding: utf-8 -*-
"""
Created on Sun May  9 01:20:15 2021

@author: enesv
"""

import pygame
import dql_agent as dql
import pygame
import random
import math
import numpy as np
import time
import datetime
from defines import *
from aux_functions import *
from objects import *

class Game:

    def __init__(self):
        
        pygame.init()
        pygame.mixer.init()
        clock = pygame.time.Clock()
        pygame.display.set_caption('autonomous car')

        self.obstacle_sprites = pygame.sprite.Group()
        self.car_sprites = pygame.sprite.Group()
        self.car = Car()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.agent = dql.DQLAgent(ARM_SONAR_NUMBER*2,3)

        self.flagReplay = False
        self.flagRunning = True
        
    # init game objects
    def initialDisplay(self):

        self.obstacle_sprites.empty()
        self.car_sprites.empty()
        self.car = Car();
        self.car_sprites.add(self.car);
        
        # car starts in a random direction
        theta = random.randint(-360,360)
        [self.car.speedX, self.car.speedY] = rotate([self.car.speedX,self.car.speedY],theta)
        self.car.sonarArmStartPoint = rotate(self.car.sonarArmStartPoint,theta)
        
        # position values ​​of obstacles and baits are read
        with open('place.txt') as f:

            lines = f.readlines()
            for line in lines:

                data = line.split()
                color = (int(data[2]),int(data[3]),int(data[4]))
                if color == GREEN:
                    self.car.greenObstacleNbr = self.car.greenObstacleNbr + 1
                self.obstacle_sprites.add(Obstacle(int(data[0]),int(data[1]),color));
        self.car.obstacle_sprites = self.obstacle_sprites

    # render of game objects
    def render(self):

        pygame.event.pump()

        self.screen.fill(BLACK)
        
        # sonar sensor 
        for i in range(len(self.car.sonarArmEndPoint)):
            pygame.draw.line( self.screen, self.car.sonarArmColor[i], sum2points(self.car.rect.center,self.car.sonarArmStartPoint),self.car.sonarArmEndPoint[i])

          
        self.obstacle_sprites.draw(self.screen)
        self.car_sprites.draw(self.screen)

        # *after* drawing everything, flip the display
        pygame.display.flip()

    def guiRoutine(self):

        keystate = pygame.key.get_pressed()

        if keystate[pygame.K_UP]: 
            self.initialDisplay()

        elif keystate[pygame.K_s]: # S key is pressed, the model is saved.
            self.agent.model.save_weights('my_model.h5')

        elif keystate[pygame.K_l]: # L key is pressed, the model is loaded
            self.agent.model.load_weights('my_model.h5')

        elif keystate[pygame.K_r]: # R key is pressed, the training starts.
            self.flagReplay = True

        elif keystate[pygame.K_k]: # K key is pressed, the training is terminated.
            self.flagReplay = False

        elif keystate[pygame.K_e]: # E key is pressed, the game is ended.
            self.flagRunning = False

        pygame.display.set_caption('Autonomous Car ' + '   Total Reward = ' + str("{:.3g}".format(self.totalReward)))

        self.render()

    def loop(self):

        saveTime = time.time()

        self.totalReward = 0
        
        state = self.car.state() # reading the initial state of the car

        while self.flagRunning:

            # reading action for state
            action = self.agent.act(state,self.flagReplay)
            
            # step action
            next_state, reward, done  = self.car.step(action)

            if self.flagReplay == True:

                logWrite(state, action, reward, next_state, done,self.agent.loss, self.car.greenObstacleNbr, self.totalReward)
                # state, action, reward, next_state, done değerleri model kuyruğunda biriktiriliyor
                self.agent.remember(state, action, reward, next_state, done)
                # repeats 5000 in model memory
                self.agent.replay(5000)
                
            state = next_state # state update
            self.totalReward = reward + self.totalReward

            # the game restarts if the car hits an obstacle or has eaten all the bait
            if done or self.car.greenObstacleNbr == 0: 

                totalReward = 0

                self.initialDisplay()

                state = self.car.state()

            self.guiRoutine()
            
            # if 1 hour has passed while traning , the model is saved
            if time.time() - saveTime > 3600:

                saveTime = time.time()
                
                self.agent.saveModel()

        pygame.quit()
