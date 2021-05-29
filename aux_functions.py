# -*- coding: utf-8 -*-
"""
Created on Tue May  4 23:51:19 2021

@author: enesv
"""
import math
import datetime
import time
import numpy as np

"""
@brief Created to sum two points on the axes.(2D)
"""
def sum2points(p1,p2):
    
    px = p1[0] + p2[0]
    py = p1[1] + p2[1]
    
    return(px,py)

"""
@brief Created to find the difference between two points.(2D)
"""
def dif2points(p1,p2):
    
    px = p1[0] - p2[0]
    py = p1[1] - p2[1]
    
    return(px,py)

"""
@brief Created to find the distance between two points
"""
def distance2points(p1,p2):
    
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
                            
    dx2 = dx**2
                            
    dy2 = dy**2
                            
    return math.sqrt(dx2 + dy2)

"""
#@brief Date 2 string date
"""
def getDate():
    
    date = datetime.datetime.now()
                
    dateStr = str(date.year) + '' + str(date.month) + '' + str(date.day) + '' + str(date.hour) + '' + str(date.minute) + ' \n'

    return dateStr

"""
#@brief Rotating points
"""
def rotate(P,theta):
    
    theta = math.radians(theta)

    R = [[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]];
        
    return np.dot(R,P)

"""
#@brief Used to log state, action, reward, next_state, done, loss, totalReward, greenObstacleNbr values
@param 
    state: the current state of the environment
    action: the agent's previous choice of action
    reward: last reward received
    next_state: the current state of the environment
    done: whether the episode is complete (True or False)
    loss: model training error value
    totalReward : total reward value until the game is over
    greenObstacleNbr: number of game green boxes     
"""
def logWrite(state, action, reward, next_state, done, loss, totalReward, greenObstacleNbr):

    trainTxtDn0 = open("logs/trainDn0.txt","a")
    trainTxtDn1 = open("logs/trainDn1.txt","a")
    lossTxt = open("logs/loss.txt","a")
    timeTxt = open("logs/time.txt","a")
    rewardTxt = open("logs/reward.txt","a")

    if done == 0:

        for i in range(len(state[0])):

            trainTxtDn0.write(str(state[0][i]) + " ")

        trainTxtDn0.write(str(action) + " ")

        trainTxtDn0.write(str(reward) + " ")

        for i in range(len(next_state[0])):

            trainTxtDn0.write(str(next_state[0][i]) + " ")

        trainTxtDn0.write(str(done) + "\n")

    if done == 1:

        for i in range(len(state[0])):

            trainTxtDn1.write(str(state[0][i]) + " ")

        trainTxtDn1.write(str(action) + " ")

        trainTxtDn1.write(str(reward) + " ")

        for i in range(len(next_state[0])):

            trainTxtDn1.write(str(next_state[0][i]) + " ")

        trainTxtDn1.write(str(done) + "\n")

    lossTxt.write(str("{:.3g}".format(loss)) + " \n")
    
    if greenObstacleNbr == 0: 
    
       rewardTxt.write(" reward = " + str(totalReward) + " \n")

    lossTxt.close()
    trainTxtDn0.close()
    trainTxtDn1.close()
    rewardTxt.close()
