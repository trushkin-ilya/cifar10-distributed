# Distributed image classification ![](https://img.shields.io/badge/tensorflow-1.15-informational?logo=tensorflow&logoColor=ccc) ![](https://img.shields.io/badge/horovod-0.19-informational)

Training distributed system for CIFAR-10 image classification using [TensorFlow](https://github.com/tensorflow/tensorflow) and [horovod](https://github.com/horovod/horovod). Data is divided into parts equal to the number of workers. Each worker is assigned a separate data slice. The total loss is produced by averaging workers losses during training. Weight updates are produced by averaging workers' local gradients. These updates apply to single neural network maintained by the chief process. Then updated weights are copied to other workers' CNNs and new training epoch is started. After each training epoch chief worker evaluates loss on the whole test dataset in a separate thread. **Only CPU training is supported**.
 
* [Get started](#get-started)
     * [Docker image](#docker-image)
        * [Prerequisites](#prerequisites)
        * [Installation](#installation)
     * [Manual setup](#manual-setup)
        * [Prerequisites](#prerequisites-1)
        * [Installation](#installation-1)
* [Running](#running)
    * [Docker](#docker)


## Get started 
### Docker image
#### Prerequisites
 - [Docker](https://www.docker.com/get-started)
#### Installation
1. Build docker image:
```
docker build -t horovod:latest horovod-docker-cpu
```


### Manual setup

#### Prerequisites
 - [Open MPI 4.0.0](https://www.open-mpi.org/software)
 - [Ubuntu 18.04](https://releases.ubuntu.com/18.04/) with packages installed:
    `build-essential g++-7 cmake git curl vim wget ca-certificates libjpeg-dev libpng-dev python3 python3-dev`
 - [pip](https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py)
    
#### Installation
1. Set environment variables to skip installing redundant frameworks:
```
export HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1
```
2. Install requirements:
```
pip install -r requirements.txt
```
3. Install OpenSSH for MPI to communicate between processes:
```
apt-get install -y --no-install-recommends openssh-client openssh-server
mkdir -p /var/run/sshd
```
4. Allow OpenSSH to talk to processes without asking for confirmation:
```
cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
```
5. Download CIFAR-10 dataset:
```
mkdir data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzvf cifar-10-python.tar.gz -C data
rm cifar-10-python.tar.gz
```
## Running
1. (optional) Launch Tensorboard in `checkpoints` dir:
```
tensorboard --logdir checkpoints
```
2. Run distributed training in 3 worker processes:
```
horovodrun -np 3 python train.py [--batch-size 1]
                                 [--save-dir checkpoints]
                                 [--epochs 10]
                                 [--data-dir data/cifar-10-batches-py]
                                 [--lr 3e-4]

```
### Docker

1. Run docker container:
```
docker run -it horovod:latest
```
 You can use `--privileged` parameter to avoid spam warnings in output.\
 To forward default TensorBoard port 6006, use `-p 6006:6006`.

2. (optional) Launch Tensorboard in `checkpoints` dir:
```
tensorboard --logdir checkpoints
```

3. Run distributed training in 3 worker processes:
```
horovodrun -np 3 python train.py [--batch-size 1]
                                 [--save-dir checkpoints]
                                 [--epochs 10]
                                 [--data-dir data/cifar-10-batches-py]
                                 [--lr 3e-4]

```
[View TensorBoard](https://tensorboard.dev/experiment/M8pRQI50R6iZ5G9RzHn7IQ)
