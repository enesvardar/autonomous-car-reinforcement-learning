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

![alt text](https://github.com/enesvardar/autonomous-car-reinforcement-leraning/blob/main/images/game.PNG)

The world created as shown in the figure has red boxes, green boxes and a car (red circle). The aim of the car is to avoid hitting these red boxes and to collect the green boxes by accepting bait. Another purpose is that if there is no obstacle in front of it or a bait around it, it goes straight and does not rotate. The game restarts when the car hits any obstacle or leaves the game frame. And the car starts moving in a random direction. In this way, the car can navigate through the game as much as possible and experience every part of the game.

### Actions

![alt text](https://github.com/enesvardar/autonomous-car-reinforcement-leraning/blob/main/images/actions.PNG)
