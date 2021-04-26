* Draft: 2021-04-26 (Mon)

# Setting up the Computing Environment



## Prepare a machine (Amazon EC2)

The official PyTorch image will not used because it lacks some essential commands such as `nvidia-smi`, `wget` and so on.

Create an Amazon EC2 instance and ssh to the created instance.

```bash
(local) $ ./ssh-deep_learning_ami_ver42_1-0-p3_8xlarge-ec2_seoul
  ...
Last login: Thu Apr 22 02:20:28 2021 from 234.567.890.12
ubuntu@node0:~$
```

For simplicity, `ubuntu@node0:~` will be omitted below. So 

```bash
ubuntu@node0:~$
# is equivalent to
$
```

The deep learning AMI ver 42.1 provides

* Ubuntu 18.04.5 LTS
* and the choice of deep learning framework.

## Prepare the PyTorch environment

To select PyTorch 1.4 with Python3 (CUDA 10.1 and Intel MKL), activate the virtual environment.

```bash
$ source activate pytorch_p36
(pytorch_p36) $
```

Notice the prompt has the leading `(pytorch_p36)` which indicates you are in the PyTorch 1.4. virtual environment.

```bash
(pytorch_p36) ubuntu@node0:~$
(pytorch_p36) $
```

## Verify the PyTorch environment

### Prepare the project directory

[py_files/intro2pytorch-quickstart-1.py](py_files/intro2pytorch-quickstart-1.py)Let's create the project directory `distributed_training`.

```bash
$ mkdir test-distributed_training
$ cd test-distributed_training/
```

### Prepare simple PyTorch source codes

Create the test code from [PyTorch > QUICKSTART](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).

There are different ways to do it. I've copied and pasted the following files directly into the project directory.

```bash
$ ls
intro2pytorch-quickstart-1.py  intro2pytorch-quickstart-2.py
$
```

intro2pytorch-quickstart-1.py, https://github.com/aimldl/pytorch/blob/main/en/hands-on/py_files/intro2pytorch-quickstart-1.py

intro2pytorch-quickstart-2.py, https://github.com/aimldl/pytorch/blob/main/en/hands-on/py_files/intro2pytorch-quickstart-2.py

### Run the source codes to check if PyTorch runs successfully

Go to [PyTorch > QUICKSTART](01-intro2pytorch-quickstart.md) and run the PyTorch codes to verify the installation.