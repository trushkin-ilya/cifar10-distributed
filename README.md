# Distributed image classification

Training distributed system for CIFAR-10 image classification using [TensorFlow](https://github.com/tensorflow/tensorflow) and [horovod](https://github.com/horovod/horovod). Data is divided into 3 parts. Each part is used by the corresponding worker. After each training epoch all workers evaluate loss on whole test dataset. Total loss is produced by averaging workers losses. Weight updates are produced by averaging workers local gradients. These updates apply to single neural network maintained by the master process. Then updated weights are copied to other workers' CNNs and new training epoch is started.
 


## Get started 
All you need to have is single-CPU machine. Code does not support GPU learning since it does not pin processes to GPU units.

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
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1
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
1. Run docker container:
```
docker run -it horovod:latest
```
**NOTE**: you can use `--privileged` parameter to avoid spam warnings in output.
2. Run distributed training in 3 worker processes:
```
horovodrun -np 3 python train.py
```
[View TensorBoard](https://tensorboard.dev/experiment/M8pRQI50R6iZ5G9RzHn7IQ)