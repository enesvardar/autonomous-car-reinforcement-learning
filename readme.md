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
