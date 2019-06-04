This is an application that runs several layers of a Deep Neural Network (DNN) model in TrustZone.

This application is based on [Darknet DNN framework](https://pjreddie.com/darknet/) and need to be run with [OP-TEE](https://www.op-tee.org/), a open source framework for Arm TrustZone.

# Prerequisites
You can run this application with real TrustZone or a simulated one by using QEMU.

**Required System**: Ubuntu-based distributions

**For simulation**, no additional hardware is needed.

**For real TrustZone, an additional board is required**. Raspberry Pi 3, HiKey Board, ARM Juno board, and so on. Check this [List](https://optee.readthedocs.io/building/devices/index.html) for more info.

# Setup
1) Follow **step1** ~ **step5** in "**Get and build the solution**" to build the OP-TEE solution.
https://optee.readthedocs.io/building/gits/build.html#get-and-build-the-solution

2) **For real boards**: If you are using boards, keep follow **step6** ~ **step7** in the above link to flash the devices. This step is device specific.

   **For simulation**: If you have chosen QEMU-v7/v8, to run the bellow command to start QEMU console.
```
make run
(qemu)c
```

3) Follow **step8** ~ **step9** to test whether OP-TEE works or not. Run:
```
tee-supplicant -d
xtest
```

## (2) Build Darknetp
1) clone codes and datasets
```
git clone -b comunicate https://github.com/mofanv/darknetp.git
git clone https://github.com/mofanv/tz_datasets.git
```
Let `$PATH_OPTEE$` be the path of OPTEE, `$PATH_darknetp$` be the path of darknetp, and `$PATH_tz_datasets$` be the path of tz_datasets.

2) copy Darknetp to example dir
```
mkdir $PATH_OPTEE$/optee_examples/darknetp
cp -a $PATH_darknetp$/. $PATH_OPTEE$/optee_examples/darknetp/
```

3) copy datasets to root dir
```
cp -a $PATH_tz_datasets$/. $PATH_OPTEE$/out-br/target/root/
```

4) rebuild the OP-TEE

**For simulation**, to run `make run` again.

**For read board**, to run `make all` again, and flash the OP-TEE to your device.

5) after boot your devices or QEMU, to test by command
```
darknetp
```
You should get the output:
 ```
# usage: ./darknet <function>
 ```
Awesome! You are really to run DNN layers in TrustZone.

# Train Models

1) To train a model from scratch 
```
darknetp classifier train -pp 4 cfg/mnist.dataset cfg/mnist_lenet.cfg
```
You can choose the partition point from layers in TEE by adjusting argument `-pp`.

You will see output from the Normal World like this:
```
# Prepare session with the TA
# Begin darknet
# mnist_lenet
# 1
# layer     filters    size              input                output
#     0 conv      6  5 x 5 / 1    28 x  28 x   3   ->    28 x  28 x   6  0.001 BFLOPs
#     1 max          2 x 2 / 2    28 x  28 x   6   ->    14 x  14 x   6
#     2 conv      6  5 x 5 / 1    14 x  14 x   6   ->    14 x  14 x   6  0.000 BFLOPs
#     3 max          2 x 2 / 2    14 x  14 x   6   ->     7 x   7 x   6
#     4 connected_TA                          294  ->   120
#     5 connected_TA                          120  ->    84
#     6 dropout_TA    p = 0.80                 84  ->    84
#     7 connected_TA                           84  ->    10
#     8 softmax_TA                                       10
#     9 cost_TA                                          10
# Learning Rate: 0.01, Momentum: 0.9, Decay: 5e-05
# 1000
# 28 28
# Loaded: 0.197170 seconds
# 1, 0.050: 0.000000, 0.000000 avg, 0.009999 rate, 3.669898 seconds, 50 images
# Loaded: 0.000447 seconds
# 2, 0.100: 0.000000, 0.000000 avg, 0.009998 rate, 3.651714 seconds, 100 images
...
```

Layers with `_TA` are running in the TrustZone. THe training loss is calculated based on outputs of the model which belong to the last layer in the TrustZone, so it can only be seen from the Trusted world. The output from the Trusted world is like this:
```
I/TA:  loss = 1.62141, avg loss = 1.62540 from the TA
I/TA:  loss = 1.58659, avg loss = 1.61783 from the TA
I/TA:  loss = 1.57328, avg loss = 1.59886 from the TA
I/TA:  loss = 1.52641, avg loss = 1.57889 from the TA
...
```

2) To use pre-trained models

You can also load a pre-trained model into both Normal World and Trusted World and then fine-tune the model, by commands:
```
darknetp classifier train -pp 4 cfg/mnist.dataset cfg/mnist_lenet.cfg models/mnist/mnist_lenet.weights
```

# License
```
Copyright (c) 2019. Queen Mary University of London
Fan Mo, f.mo18@imperial.ac.uk

This file is part of SensingKit-Android library.
For more information, please visit http://www.sensingkit.org.

SensingKit-Android is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SensingKit-Android is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with SensingKit-Android.  If not, see <http://www.gnu.org/licenses/>.
```

More things to be added ;)