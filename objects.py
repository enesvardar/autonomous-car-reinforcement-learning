# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:11:28 2021

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


class Car(pygame.sprite.Sprite):

    def __init__(self):

        pygame.sprite.Sprite.__init__(self)
        
        # creating car object on screen 
        self.image = pygame.Surface((BALL_WIDTH,BALL_HEIGHT))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, RED, self.rect.center,BALL_HEIGHT / 2)
        
        # the starting point of the car
        self.rect.center = (BALL_START_POS_X,BALL_START_POS_Y)
        
        # The variables that keep the car's forward speed are associated. 
        self.speedX = BALL_SPEED;
        self.speedY = 0;
        
        # float car pos
        self.fltCenterX = self.rect.centerx
        self.fltCenterY = self.rect.centery
        
        # obstacle_sprites
        self.obstacle_sprites = pygame.sprite.Group();

        # the starting position of the sensors on the car 
        self.sonarArmStartPoint = dif2points(self.rect.center , self.rect.midright)
        
        # variable that holds the number of baits in the game
        self.greenObstacleNbr = 0
        
        # sonar arm theta
        self.sonarArmTheta = [0, -40, +40, -20, +20]

        # final position of the sonar arm on the sensor 
        self.sonarArmEndPoint = []

        # sensor arm color (in the color of the object with which it interacts)
        self.sonarArmColor = []

    """
      @brief the previous state of the environment
      @returns 
          state: the current state of the environment [L1 L2 L3 L4 L5 C1 C2 C3 C4 C5]
          reward: reward value after action 
          done: whether the episode is complete (True or False)
    """
    def state(self):   
        
        self.sonarArmEndPoint = [] # final position of the sonar arm 
        self.sonarArmColor = [] # color of detected object 
        sonarArmLenght = [] # sonar start - end 

        for arm in range(ARM_SONAR_NUMBER): 
                                    
            [sonarx,sonary] = rotate([self.speedX/BALL_SPEED,self.speedY/BALL_SPEED],self.sonarArmTheta[arm])
            
            armPoint = []; pointx = 0; pointy = 0; armColor = BLUE; i = 0;
            
            while(True):
                pointx = self.rect.centerx + self.sonarArmStartPoint[0] + sonarx * i
                pointy = self.rect.centery + self.sonarArmStartPoint[1] + sonary * i

                if(0 < pointx and pointx <WINDOW_WIDTH and 0 < pointy and pointy <WINDOW_HEIGHT) and i < ARM_MAX_LENGHT:                    
                    armPoint.append((pointx,pointy))
                else:
                   break;                   
                i = i + 1
            
            dif = 0 # scanning until an obstacle is detected or the maximum detection range is reached

            if len(armPoint) > 0:
                
                bufDif = distance2points(armPoint[0],(pointx,pointy))
                
                for obstacle in self.obstacle_sprites.sprites():                        
                        for p in armPoint:
                            
                            if((obstacle.rect.midleft[0] + 2) <= p[0] and p[0] <= (obstacle.rect.midright[0] - 2)
                               and (obstacle.rect.midtop[1] - 2) <= p[1] and p[1] <= (obstacle.rect.midbottom[1] + 2)):

                                dif = distance2points(armPoint[0],p)
                                
                                if(dif < bufDif):
                                    pointx,pointy =  p
                                    bufDif = dif
                                    armColor = obstacle.color
                                    break                                
                                
                if(bufDif > ARM_MAX_LENGHT):
                    pointx = self.rect.centerx + self.sonarArmStartPoint[0] + sonarx * ARM_MAX_LENGHT
                    pointy = self.rect.centery + self.sonarArmStartPoint[1] + sonary * ARM_MAX_LENGHT
                    bufDif = ARM_MAX_LENGHT
                    armColor = BLUE

                sonarArmLenght.append(bufDif)
                
            self.sonarArmEndPoint.append([pointx,pointy])
            self.sonarArmColor.append(armColor)

        state = []  
        for i in range(ARM_SONAR_NUMBER): # firstly the measured distance information is transferred to the state variable 
            state.append(sonarArmLenght[i])
        for i in range(ARM_SONAR_NUMBER): # secondly, the colors of the detected objects are transferred to the state variable 
            if(self.sonarArmColor[i] == BLUE):
                state.append(0)
            elif(self.sonarArmColor[i] == RED):
                state.append(1)
            elif(self.sonarArmColor[i] == GREEN):
                state.append(2)                
        return (np.array(state).reshape((1,ARM_SONAR_NUMBER*2)));
    
    """
      @brief 
      @params
          action: the agent's  choice of action
      @returns 
          state: the current state of the environment [L1 L2 L3 L4 L5 C1 C2 C3 C4 C5]
          reward: reward value after action 
          done: whether the episode is complete (True or False)
    """ 
    def step(self,action):

        done = 0
        
        if  action == 0: # GO_TURN_RIGHT
            theta = TURN_THETA
        elif action == 1: # GO_TURN_LEFT
            theta = -TURN_THETA
        else: # GO_STRAIGHT
            theta = 0
        
        # the sensor initial position and the sensor direction are rotated up to theta      
        [self.speedX,self.speedY] = rotate([self.speedX,self.speedY],theta)                     
        self.sonarArmStartPoint = rotate(self.sonarArmStartPoint,theta)
        
        # calculating the new position of the car 
        self.fltCenterX += self.speedX
        self.fltCenterY += self.speedY
        self.rect.x = int(self.fltCenterX)
        self.rect.y = int(self.fltCenterY)
        
        
        if action == GO_STRAIGHT:
            reward = +1
        else: # GO_TURN_RIGHT or GO_TURN_LEFT
            reward = -0.1
        
        # if the car goes out of the frame
        if self.rect.midleft[0] < 0 or  WINDOW_WIDTH < self.rect.midright[0] or self.rect.midtop[1] < 0 or  WINDOW_HEIGHT < self.rect.midbottom[1]:

            reward = -200
            done = 1 # game over 
            
        else:
            
            for i in range(len(self.obstacle_sprites.sprites())):
    
                obje = pygame.sprite.Group();
    
                obje.add(self.obstacle_sprites.sprites()[i])
    
                hits = pygame.sprite.spritecollide(self,obje,False)
                
                # -200 points are awarded if the car hits a red object.
                if hits and self.obstacle_sprites.sprites()[i].color == RED:
    
                    reward = -200
                    done = 1
                    
                # +100 points are awarded if the car hits a green object
                elif hits and self.obstacle_sprites.sprites()[i].color == GREEN:
    
                    reward = +100
                    done = 0
                    
                    # the bait is deleted if the car hits the baits 
                    self.obstacle_sprites.sprites()[i].kill()
                    self.greenObstacleNbr = self.greenObstacleNbr - 1
    
                    break

        return self.state(),reward,done

#@brief  obstacle and feed
class Obstacle(pygame.sprite.Sprite):

    def __init__(self,centerx,centery,color):

        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        self.image.fill(color)

        self.color = color

        self.rect = self.image.get_rect()

        self.rect.centerx = centerx
        
        self.rect.centery = centery