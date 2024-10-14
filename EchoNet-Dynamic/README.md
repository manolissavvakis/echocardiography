# 3D CNN training, using echocardiography data.

## Step 1: Download dataset.
Dataset used is the EchoNet-Dynamic, which can be found [here](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a).
The .zip file contains 10,030 echocardiogram videos and the corresponding measurements,tracings and calculations for each one.

## Step 2 (optional): Augment training data.
Patients are labeled as having no pathological risk (class 0) and as having pathological risk (class 1) based on their ejection fraction. A value greater or equal than 45. is labeled as 1 and less than 45. is labeled as 0. Echocardiogram videos in class 1 (1636) are much less than those in class 1 (8394). To fix this inbalance, there is "augmentation.py" script to augment randomly sampled videos from class 0, until both classes contain the same amount of data.
To run: 
```console
python scripts/augmentation.py
```

## Step 3: Run an experiment.
**To train a new model** type in the console:
```console
python scripts/main.py --training
```
To change training parameters (epochs, training batch size, validation batch size) edit the values in line 37 of *scripts/main.py*

**To test an already trained model** type in the console:
```console
python scripts/main.py --no-training --epoch_to_load epoch_to_load
```
epoch_to_load: (int) which model saved in checkpoints directory should be loaded.

### Warning
When training a new model, *learning_curve.png*, *accuracy_plot.png*, *training.log* files will be created. In case of a new training session, these files will be overwritten by the new ones. Same thing applies when testing with *testing.log* file.