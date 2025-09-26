# Train A Tiny CNN Model With Less Than 8000 Paramters And Hit 99.4% Accuracy constantly in last 3 epochs.

## Model 1
- **Target** - Have the basic skelton setup with less than 10K parameters.
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
```
**Model Layers and Parameter Count:**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
            Conv2d-3           [-1, 12, 28, 28]             864
              ReLU-4           [-1, 12, 28, 28]               0
         MaxPool2d-5           [-1, 12, 14, 14]               0
            Conv2d-6           [-1, 16, 14, 14]           1,728
              ReLU-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 12, 12]           2,304
              ReLU-9           [-1, 16, 12, 12]               0
        MaxPool2d-10             [-1, 16, 6, 6]               0
           Conv2d-11             [-1, 16, 4, 4]           2,304
             ReLU-12             [-1, 16, 4, 4]               0
           Conv2d-13             [-1, 16, 2, 2]           2,304
             ReLU-14             [-1, 16, 2, 2]               0
AdaptiveAvgPool2d-15             [-1, 16, 1, 1]               0
           Linear-16                   [-1, 10]             170
================================================================
Total params: 9,746
Trainable params: 9,746
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.35
Params size (MB): 0.04
Estimated Total Size (MB): 0.39
----------------------------------------------------------------
```

- **Results :**
  - **Best Train Accuracy -** *99.37*
  - **Best Test Accuracy -** *99.14*
  - **Total Parameters -** *9,746*
```
EPOCH: 0
Loss=0.14111807942390442 Batch_id=1874 Accuracy=82.53: 100%|██████████| 1875/1875 [00:21<00:00, 86.67it/s]

Test set: Average loss: 0.1046, Accuracy: 9688/10000 (96.88%)

EPOCH: 1
Loss=0.0983821228146553 Batch_id=1874 Accuracy=97.71: 100%|██████████| 1875/1875 [00:20<00:00, 91.15it/s]

Test set: Average loss: 0.0568, Accuracy: 9818/10000 (98.18%)

EPOCH: 2
Loss=0.004761046729981899 Batch_id=1874 Accuracy=98.34: 100%|██████████| 1875/1875 [00:22<00:00, 84.11it/s]

Test set: Average loss: 0.0401, Accuracy: 9873/10000 (98.73%)

EPOCH: 3
Loss=0.005260005127638578 Batch_id=1874 Accuracy=98.59: 100%|██████████| 1875/1875 [00:21<00:00, 85.96it/s] 

Test set: Average loss: 0.0439, Accuracy: 9865/10000 (98.65%)

EPOCH: 4
Loss=0.010513199493288994 Batch_id=1874 Accuracy=98.77: 100%|██████████| 1875/1875 [00:22<00:00, 85.16it/s] 

Test set: Average loss: 0.0500, Accuracy: 9846/10000 (98.46%)

EPOCH: 5
Loss=0.00804966315627098 Batch_id=1874 Accuracy=98.92: 100%|██████████| 1875/1875 [00:21<00:00, 85.94it/s]

Test set: Average loss: 0.0327, Accuracy: 9903/10000 (99.03%)

EPOCH: 6
Loss=0.009162095375359058 Batch_id=1874 Accuracy=98.96: 100%|██████████| 1875/1875 [00:21<00:00, 88.76it/s]

Test set: Average loss: 0.0434, Accuracy: 9872/10000 (98.72%)

EPOCH: 7
Loss=0.01664828136563301 Batch_id=1874 Accuracy=99.05: 100%|██████████| 1875/1875 [00:20<00:00, 89.37it/s]

Test set: Average loss: 0.0295, Accuracy: 9902/10000 (99.02%)

EPOCH: 8
Loss=0.3410673141479492 Batch_id=1874 Accuracy=99.15: 100%|██████████| 1875/1875 [00:20<00:00, 90.21it/s] 

Test set: Average loss: 0.0438, Accuracy: 9867/10000 (98.67%)

EPOCH: 9
Loss=0.0017661042511463165 Batch_id=1874 Accuracy=99.21: 100%|██████████| 1875/1875 [00:21<00:00, 89.04it/s]

Test set: Average loss: 0.0326, Accuracy: 9894/10000 (98.94%)

EPOCH: 10
Loss=0.0008599875727668405 Batch_id=1874 Accuracy=99.24: 100%|██████████| 1875/1875 [00:21<00:00, 87.76it/s]

Test set: Average loss: 0.0264, Accuracy: 9925/10000 (99.25%)

EPOCH: 11
Loss=8.969243936007842e-05 Batch_id=1874 Accuracy=99.34: 100%|██████████| 1875/1875 [00:21<00:00, 88.08it/s] 

Test set: Average loss: 0.0365, Accuracy: 9899/10000 (98.99%)

EPOCH: 12
Loss=0.04132688045501709 Batch_id=1874 Accuracy=99.27: 100%|██████████| 1875/1875 [00:20<00:00, 89.35it/s]

Test set: Average loss: 0.0423, Accuracy: 9874/10000 (98.74%)

EPOCH: 13
Loss=1.967899879673496e-05 Batch_id=1874 Accuracy=99.36: 100%|██████████| 1875/1875 [00:20<00:00, 90.11it/s]

Test set: Average loss: 0.0344, Accuracy: 9912/10000 (99.12%)

EPOCH: 14
Loss=2.8345180908218026e-05 Batch_id=1874 Accuracy=99.37: 100%|██████████| 1875/1875 [00:21<00:00, 88.75it/s]

Test set: Average loss: 0.0332, Accuracy: 9914/10000 (99.14%)
```
- **Analysis :** The model skeleton is setup right, Total parameter count is less than 10K. Model is doing a good job with best train accuracy of 99.37 and test accuracy of 99.14 (little overfitting). The model can be regularized to reduce the gap between train and test accurracy.

## Model 2
- **Target :** Add Regularization (BatchNorm and Dropout) to remove Overfitting and increase the train and test accuracy of the model.
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
```

**Model Layers and Parameter Count**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 12, 28, 28]             864
              ReLU-6           [-1, 12, 28, 28]               0
       BatchNorm2d-7           [-1, 12, 28, 28]              24
           Dropout-8           [-1, 12, 28, 28]               0
         MaxPool2d-9           [-1, 12, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           1,728
             ReLU-11           [-1, 16, 14, 14]               0
      BatchNorm2d-12           [-1, 16, 14, 14]              32
          Dropout-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 12, 12]           2,304
             ReLU-15           [-1, 16, 12, 12]               0
      BatchNorm2d-16           [-1, 16, 12, 12]              32
          Dropout-17           [-1, 16, 12, 12]               0
        MaxPool2d-18             [-1, 16, 6, 6]               0
           Conv2d-19             [-1, 16, 4, 4]           2,304
             ReLU-20             [-1, 16, 4, 4]               0
      BatchNorm2d-21             [-1, 16, 4, 4]              32
          Dropout-22             [-1, 16, 4, 4]               0
           Conv2d-23             [-1, 16, 2, 2]           2,304
             ReLU-24             [-1, 16, 2, 2]               0
      BatchNorm2d-25             [-1, 16, 2, 2]              32
          Dropout-26             [-1, 16, 2, 2]               0
AdaptiveAvgPool2d-27             [-1, 16, 1, 1]               0
           Linear-28                   [-1, 10]             170
================================================================
Total params: 9,914
Trainable params: 9,914
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.68
Params size (MB): 0.04
Estimated Total Size (MB): 0.72
----------------------------------------------------------------
```

- **Results :**
  - **Best Train Accuracy -** *99.11*
  - **Best Test Accuracy -** *99.40*
  - **Total Parameters -** *9,914*
 ```
EPOCH: 0
Loss=0.14013588428497314 Batch_id=1874 Accuracy=95.36: 100%|██████████| 1875/1875 [00:24<00:00, 77.27it/s]

Test set: Average loss: 0.0423, Accuracy: 9869/10000 (98.69%)

EPOCH: 1
Loss=0.006970836315304041 Batch_id=1874 Accuracy=98.20: 100%|██████████| 1875/1875 [00:23<00:00, 79.09it/s]

Test set: Average loss: 0.0321, Accuracy: 9895/10000 (98.95%)

EPOCH: 2
Loss=0.1753063052892685 Batch_id=1874 Accuracy=98.39: 100%|██████████| 1875/1875 [00:24<00:00, 75.01it/s]

Test set: Average loss: 0.0343, Accuracy: 9902/10000 (99.02%)

EPOCH: 3
Loss=0.06548615545034409 Batch_id=1874 Accuracy=98.53: 100%|██████████| 1875/1875 [00:25<00:00, 74.48it/s]

Test set: Average loss: 0.0257, Accuracy: 9928/10000 (99.28%)

EPOCH: 4
Loss=0.0019103474915027618 Batch_id=1874 Accuracy=98.72: 100%|██████████| 1875/1875 [00:25<00:00, 74.21it/s]

Test set: Average loss: 0.0234, Accuracy: 9928/10000 (99.28%)

EPOCH: 5
Loss=0.07474738359451294 Batch_id=1874 Accuracy=98.76: 100%|██████████| 1875/1875 [00:24<00:00, 75.72it/s]

Test set: Average loss: 0.0329, Accuracy: 9898/10000 (98.98%)

EPOCH: 6
Loss=0.005956639535725117 Batch_id=1874 Accuracy=98.87: 100%|██████████| 1875/1875 [00:25<00:00, 74.08it/s]

Test set: Average loss: 0.0270, Accuracy: 9921/10000 (99.21%)

EPOCH: 7
Loss=0.00783977285027504 Batch_id=1874 Accuracy=98.88: 100%|██████████| 1875/1875 [00:24<00:00, 75.36it/s]

Test set: Average loss: 0.0198, Accuracy: 9935/10000 (99.35%)

EPOCH: 8
Loss=0.0004677710239775479 Batch_id=1874 Accuracy=98.98: 100%|██████████| 1875/1875 [00:24<00:00, 76.17it/s]

Test set: Average loss: 0.0223, Accuracy: 9929/10000 (99.29%)

EPOCH: 9
Loss=0.08025066554546356 Batch_id=1874 Accuracy=98.95: 100%|██████████| 1875/1875 [00:25<00:00, 73.94it/s]

Test set: Average loss: 0.0209, Accuracy: 9931/10000 (99.31%)

EPOCH: 10
Loss=0.003736080601811409 Batch_id=1874 Accuracy=99.06: 100%|██████████| 1875/1875 [00:24<00:00, 75.15it/s]

Test set: Average loss: 0.0244, Accuracy: 9929/10000 (99.29%)

EPOCH: 11
Loss=0.0009105286444537342 Batch_id=1874 Accuracy=99.09: 100%|██████████| 1875/1875 [00:25<00:00, 73.37it/s]

Test set: Average loss: 0.0224, Accuracy: 9938/10000 (99.38%)

EPOCH: 12
Loss=0.1751810759305954 Batch_id=1874 Accuracy=99.09: 100%|██████████| 1875/1875 [00:25<00:00, 72.97it/s]

Test set: Average loss: 0.0223, Accuracy: 9927/10000 (99.27%)

EPOCH: 13
Loss=0.08499334007501602 Batch_id=1874 Accuracy=99.04: 100%|██████████| 1875/1875 [00:25<00:00, 72.75it/s]

Test set: Average loss: 0.0226, Accuracy: 9925/10000 (99.25%)

EPOCH: 14
Loss=0.0008189385407604277 Batch_id=1874 Accuracy=99.11: 100%|██████████| 1875/1875 [00:25<00:00, 74.37it/s]

Test set: Average loss: 0.0199, Accuracy: 9940/10000 (99.40%)
```

**Analysis :**
  - Adding BatchNorm and Dropout has reduced the overfitting. The model in underfitting now with best train accuracy of 99.11 and best test accuracy of 99.4. As Next step the aim is to reduce the paramters count and add data augmentation to reach the target of 99.4 test accuracy with less than 8000 paramters.

# Model - 3
**Target :**
  - Modify the model to have less than 8000 parameters and add data augmentation to reduce overfitting of the model.

 ```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=11, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(11),
            nn.Dropout(0.1)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(10, 10, bias = False)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
```
**Model Layers and Parameter Count :**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 12, 28, 28]             864
              ReLU-6           [-1, 12, 28, 28]               0
       BatchNorm2d-7           [-1, 12, 28, 28]              24
           Dropout-8           [-1, 12, 28, 28]               0
         MaxPool2d-9           [-1, 12, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           1,728
             ReLU-11           [-1, 16, 14, 14]               0
      BatchNorm2d-12           [-1, 16, 14, 14]              32
          Dropout-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 12, 12]           2,304
             ReLU-15           [-1, 16, 12, 12]               0
      BatchNorm2d-16           [-1, 16, 12, 12]              32
          Dropout-17           [-1, 16, 12, 12]               0
        MaxPool2d-18             [-1, 16, 6, 6]               0
           Conv2d-19             [-1, 11, 4, 4]           1,584
             ReLU-20             [-1, 11, 4, 4]               0
      BatchNorm2d-21             [-1, 11, 4, 4]              22
          Dropout-22             [-1, 11, 4, 4]               0
           Conv2d-23             [-1, 10, 2, 2]             990
             ReLU-24             [-1, 10, 2, 2]               0
      BatchNorm2d-25             [-1, 10, 2, 2]              20
          Dropout-26             [-1, 10, 2, 2]               0
AdaptiveAvgPool2d-27             [-1, 10, 1, 1]               0
           Linear-28                   [-1, 10]             100
================================================================
Total params: 7,788
Trainable params: 7,788
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 0.03
Estimated Total Size (MB): 0.71
----------------------------------------------------------------
```
**Result :**
  - **Best Train Accuracy -** *98.71*
  - **Best Test Accuracy -** *99.45*
  - **Total Parameters -** *7,788*

```
EPOCH: 0
Loss=0.05117357522249222 Batch_id=1874 Accuracy=93.83: 100%|██████████| 1875/1875 [00:28<00:00, 66.49it/s]

Test set: Average loss: 0.0461, Accuracy: 9867/10000 (98.67%)

EPOCH: 1
Loss=0.023698857054114342 Batch_id=1874 Accuracy=97.29: 100%|██████████| 1875/1875 [00:27<00:00, 67.80it/s]

Test set: Average loss: 0.0338, Accuracy: 9890/10000 (98.90%)

EPOCH: 2
Loss=0.07147856801748276 Batch_id=1874 Accuracy=97.65: 100%|██████████| 1875/1875 [00:27<00:00, 67.04it/s]

Test set: Average loss: 0.0326, Accuracy: 9886/10000 (98.86%)

EPOCH: 3
Loss=0.11392983049154282 Batch_id=1874 Accuracy=97.80: 100%|██████████| 1875/1875 [00:29<00:00, 63.58it/s]

Test set: Average loss: 0.0252, Accuracy: 9916/10000 (99.16%)

EPOCH: 4
Loss=0.0076786065474152565 Batch_id=1874 Accuracy=98.03: 100%|██████████| 1875/1875 [00:27<00:00, 67.85it/s]

Test set: Average loss: 0.0242, Accuracy: 9921/10000 (99.21%)

EPOCH: 5
Loss=0.040524642914533615 Batch_id=1874 Accuracy=98.15: 100%|██████████| 1875/1875 [00:27<00:00, 67.32it/s]

Test set: Average loss: 0.0247, Accuracy: 9922/10000 (99.22%)

EPOCH: 6
Loss=0.04864704608917236 Batch_id=1874 Accuracy=98.26: 100%|██████████| 1875/1875 [00:27<00:00, 68.17it/s]

Test set: Average loss: 0.0258, Accuracy: 9918/10000 (99.18%)

EPOCH: 7
Loss=0.050457913428545 Batch_id=1874 Accuracy=98.28: 100%|██████████| 1875/1875 [00:27<00:00, 67.93it/s]

Test set: Average loss: 0.0221, Accuracy: 9924/10000 (99.24%)

EPOCH: 8
Loss=0.05666813999414444 Batch_id=1874 Accuracy=98.55: 100%|██████████| 1875/1875 [00:29<00:00, 64.64it/s]

Test set: Average loss: 0.0209, Accuracy: 9934/10000 (99.34%)

EPOCH: 9
Loss=0.023745547980070114 Batch_id=1874 Accuracy=98.67: 100%|██████████| 1875/1875 [00:27<00:00, 68.27it/s]

Test set: Average loss: 0.0199, Accuracy: 9937/10000 (99.37%)

EPOCH: 10
Loss=0.04520298168063164 Batch_id=1874 Accuracy=98.66: 100%|██████████| 1875/1875 [00:27<00:00, 67.85it/s]

Test set: Average loss: 0.0186, Accuracy: 9942/10000 (99.42%)

EPOCH: 11
Loss=0.002779433038085699 Batch_id=1874 Accuracy=98.67: 100%|██████████| 1875/1875 [00:27<00:00, 67.90it/s]

Test set: Average loss: 0.0201, Accuracy: 9941/10000 (99.41%)

EPOCH: 12
Loss=0.005277360323816538 Batch_id=1874 Accuracy=98.70: 100%|██████████| 1875/1875 [00:27<00:00, 68.22it/s]

Test set: Average loss: 0.0193, Accuracy: 9942/10000 (99.42%)

EPOCH: 13
Loss=0.005125211086124182 Batch_id=1874 Accuracy=98.71: 100%|██████████| 1875/1875 [00:27<00:00, 67.64it/s]

Test set: Average loss: 0.0187, Accuracy: 9945/10000 (99.45%)

EPOCH: 14
Loss=0.2332158088684082 Batch_id=1874 Accuracy=98.68: 100%|██████████| 1875/1875 [00:27<00:00, 68.38it/s]

Test set: Average loss: 0.0182, Accuracy: 9942/10000 (99.42%)
```

**Analysis :**
  - The model is constantly hitting the 99.4% accuracy from epoch 10. Adding augmentation helped model to increase the test accuracy.
  - But still there is underfitting as we are limited by number of parameters.
  - If we increase the paramters we can increase the train accuracy and pass on the learning on test to increase the test accuracy. Refined the model skeleton little bit to decrease the parameters mostly changed the kernel size.
  - Adding scheduler (stepLR) helped to stabalize the model and not overshoot.
