# Autonomous Car with Reinforcement Leraning

### Setup
Insall the requirements:
```
pip install requirements.txt
```
### CUDA, cuDNN and tensorflow-gpu

If you run into any cuda errors, make sure you've got a [compatible set](https://www.tensorflow.org/install/source#tested_build_configurations) of cuda/cudnn/tensorflow versions installed. However, beware of the following:
>The compatibility table given in the tensorflow site does not contain specific minor versions for cuda and cuDNN. However, if the specific versions are not met, there will be an error when you try to use tensorflow. [source](https://stackoverflow.com/a/53727997)

A configuration that works for me is:
- CUDA 10.1.105
- cuDNN 7.6.5
- tensorflow-gpu 2.1.0 (this is automatically installed during with the above script, see [requirements.txt](requirements.txt))

### Environment

<p align="center">
  <img src="https://github.com/enesvardar/autonomous-car-reinforcement-leraning/blob/main/images/game.PNG" width="500">
</p>

The world created as shown in the figure has red boxes, green boxes and a car (red circle). The aim of the car is to avoid hitting these red boxes and to collect the green boxes by accepting bait. Another purpose is that if there is no obstacle in front of it or a bait around it, it goes straight and does not rotate. The game restarts when the car hits any obstacle or leaves the game frame. And the car starts moving in a random direction. In this way, the car can navigate through the game as much as possible and experience every part of the game.

### Actions

<p align="center">
  <img src="https://github.com/enesvardar/autonomous-car-reinforcement-leraning/blob/main/images/actions.png" width="500">
</p>

As shown in the figure, the delegate can perform 3 types of actions. The agent can go straight, turn left or right, depending on the action he chooses. The speed of the agent is constant. The rotation made by the agent corresponds to an angular rotation of 10 degrees in the direction of advancement. A random number between 0-1 is generated for the action selection. If this value is less than the epsilon value, the action selection is made according to the maximum rewarding output of the artificial neural network. If the generated number is greater than the epsilon value, a random action is chosen. After each training phase, the epsilon value is reduced by multiplying it with a number between 0-1. The reason for choosing such an action is to make our agent learn new things by acting randomly at the beginning of the training process. If our agent constantly draws on his own experience, he avoids discovering new things. Or if he is constantly moving to discover new things, it would be too much time wasted. For this, a balance must be struck between these two choices. In this study, the epsilon value was chosen as 50 and multiplied by the value 0.95 after each training phase. Thus, the agent makes random moves at the beginning of the training phase and explores the world he is in as much as possible, and after a certain period of time, he acts according to his experiences by taking actions at the action output of the artificial neural network. And it makes better decisions.

###	Reward

<p align="center">
  <img src="https://github.com/enesvardar/autonomous-car-reinforcement-leraning/blob/main/images/reward.png" width="500">
</p>
