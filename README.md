This is an application that runs several layers of a Deep Neural Network (DNN) model in TrustZone.

Darknetp (Darknet Partitioned) is based on [Darknet DNN framework](https://pjreddie.com/darknet/) and need to be run with [OP-TEE](https://www.op-tee.org/), a open source framework for Arm TrustZone.

# Prerequisites
You can run this application with a real TrustZone or a simulated one by using QEMU.

**Required System**: Ubuntu-based distributions

**For simulation**, no additional hardware is needed.

**For real TrustZone, an additional board is required**. Raspberry Pi 3, HiKey Board, ARM Juno board, and so on. Check this [List](https://optee.readthedocs.io/building/devices/index.html) for more info.

# Setup
## (1) Set up OP-TEE

1) Follow **step1** ~ **step5** in "**Get and build the solution**" to build the OP-TEE solution.
https://optee.readthedocs.io/building/gits/build.html#get-and-build-the-solution

2) **For real boards**: If you are using boards, keep follow **step6** ~ **step7** in the above link to flash the devices. This step is device specific.

**For simulation**: If you have chosen QEMU-v7/v8, to run the bellow command to start QEMU console.
```
make run
(qemu)c
```

4) Follow **step8** ~ **step9** to test whether OP-TEE works or not. Run:
```
tee-supplicant -d
xtest
```

## (2) Build Darknetp

