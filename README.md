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
usage: ./darknet <function>
 ```
Awesome! You are really to run DNN layers in TrustZone.

# Train Models

To train a model from scratch 
```
darknetp classifier train -pp 4 cfg/mnist.dataset cfg/mnist_lenet.cfg
```
You can choose the partition point from layers in TEE by adjusting argument `-pp`.


Prepare session with the TA
Begin darknet
mnist_lenet
1
layer     filters    size              input                output
    0 conv      6  5 x 5 / 1    28 x  28 x   3   ->    28 x  28 x   6  0.001 BFLOPs
    1 max          2 x 2 / 2    28 x  28 x   6   ->    14 x  14 x   6
    2 conv      6  5 x 5 / 1    14 x  14 x   6   ->    14 x  14 x   6  0.000 BFLOPs
    3 max          2 x 2 / 2    14 x  14 x   6   ->     7 x   7 x   6
    4 connected_TA                          294  ->   120
    5 connected_TA                          120  ->    84
    6 dropout_TA    p = 0.80                 84  ->    84
    7 connected_TA                           84  ->    10
    8 softmax_TA                                       10
    9 cost_TA                                          10
Learning Rate: 0.01, Momentum: 0.9, Decay: 5e-05
1000
28 28
Loaded: 0.197170 seconds
1, 0.050: 0.000000, 0.000000 avg, 0.009999 rate, 3.669898 seconds, 50 images
user CPU start: 0.059942; end: 3.057188
kernel CPU start: 8.142247; end: 9.270185
Max: 8212  kilobytes
vmsize:52132; vmrss:7996; vmdata:47156; vmstk:132; vmexe:224; vmlib:1872
Loaded: 0.000447 seconds
2, 0.100: 0.000000, 0.000000 avg, 0.009998 rate, 3.651714 seconds, 100 images
user CPU start: 3.061479; end: 5.769754
kernel CPU start: 9.270185; end: 10.432980
Max: 8676  kilobytes
vmsize:52132; vmrss:8304; vmdata:47424; vmstk:132; vmexe:224; vmlib:1872
Loaded: 0.006637 seconds



