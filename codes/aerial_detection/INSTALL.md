* Draft: 2020-10-03 (Sat)

# Install & Configure the Computing Environment

## Summary

```bash
#!/bin/bash
# install_aerialdetection
# Usage:
#   $ chmod +x install_aerialdetection 
#   $ ./install_aerialdetection

# Install Aerialdetection
# a. Create a conda virtual environment and activate it. Then install Cython.
conda create --name AerialDetection --clone pytorch_p36
source activate AerialDetection
conda install -y cython

# b. Install PyTorch stable or nightly and torchvision following the official instructions.
# c. Clone the AerialDetection repository.
cd SageMaker/
# if folder AerialDetection doesn't exist
# git clone https://github.com/dingjiansw101/AerialDetection.git
cd AerialDetection

# d. Compile cuda extensions.
chmod +x compile.sh
./compile.sh

# e. Install AerialDetection (other dependencies will be installed automatically).
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```



[AerialDetection](https://github.com/dacon-ai/AerialDetection)/[INSTALL.md](https://github.com/dacon-ai/AerialDetection/blob/master/INSTALL.md)

> ### Requirements
>
> - Linux
> - Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
> - PyTorch 1.1
> - CUDA 9.0+
> - NCCL 2+
> - GCC 4.9+
> - [mmcv](https://github.com/open-mmlab/mmcv)
>
> are tested on:
>
> - OS: Ubuntu 16.04/18.04 and CentOS 7.2
> - CUDA: 9.0/9.2/10.0
> - NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
> - GCC: 4.9/5.3/5.4/7.3



> ### Install AerialDetection
>
> a. Create a conda virtual environment and activate it. Then install Cython.
>
> ```bash
> $ conda create -n AerialDetection python=3.7 -y
> $ source activate AerialDetection
> (AerialDetection)$ conda install cython
> ```
>
> b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).
>
> c. Clone the AerialDetection repository.
>
> d. Compile cuda extensions.
>
> ```bash
> $ ./compile.sh
> ```
>
> e. Install AerialDetection (other dependencies will be installed automatically).
>
> ```bash
> $ pip install -r requirements.txt
> $ python setup.py develop
> # or "pip install -e ."
> ```

## SageMaker

### Cloning `pytorch_p36` to `AerialDetection`

The `--clone` option allows to clone an existing Conda environment.

> ```bash
> $ conda create --name mydestination --clone mysource
> ```
>
> Source: [How can you “clone” a conda environment into the root environment?](https://stackoverflow.com/questions/40700039/how-can-you-clone-a-conda-environment-into-the-root-environment)

`pytorch_p36` is cloned to a new Conda environment `AerialDetection`.

```bash
sh-4.2$ conda create --name AerialDetection --clone pytorch_p36
Source:      /home/ec2-user/anaconda3/envs/pytorch_p36
Destination: /home/ec2-user/anaconda3/envs/AerialDetection
  ...
sh-4.2$
```

Close the terminal and restart it.

### Activate `AerialDetection`

```bash
sh-4.2$ source activate AerialDetection
(AerialDetection) sh-4.2$
```

### Install `cython`

```bash
(AerialDetection) sh-4.2$ conda install -y cython
```

### Python 3.5+

The `python --version` command shows the version.

```bash
(AerialDetection) sh-4.2$ python --version
Python 3.6.10 :: Anaconda, Inc.
(AerialDetection) sh-4.2$
```

### PyTorch 1.1

```
(AerialDetection) sh-4.2$ python -c "import torch; print(torch.__version__)"
1.4.0
(AerialDetection) sh-4.2$
```

### CUDA 9.0+

The `nvidia-smi` command shows the CUDA version. In the next example, `CUDA Version: 11.0` shows that CUDA 11.0 is installed.

```bash
(AerialDetection) $ nvidia-smi
Sat Oct  3 13:05:25 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.05    Driver Version: 450.51.05    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
  ...
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
(AerialDetection) $
```



```bash
(AerialDetection) sh-4.2$ cd SageMaker/
(AerialDetection) sh-4.2$ git clone https://github.com/dingjiansw101/AerialDetection.git
(AerialDetection) sh-4.2$ cd AerialDetection
```



e. Install AerialDetection (other dependencies will be installed automatically).

```bash
$ pip install -r requirements.txt
$ python setup.py develop
```

Install DOTA_devkit

The instructions are as follows.

```bash
$ sudo apt-get install swig
$ cd DOTA_devkit
$ swig -c++ -python polyiou.i
$ python setup.py build_ext --inplace
```

But SageMaker doesn't have either `apt-get` or `apt`.

```bash
$ sudo apt-get install swig
sudo: apt-get: command not found
$ sudo apt install swig
sudo: apt: command not found
$
```

Instead, use the `yum` command.

```bash
$ sudo yum install swig
Loaded plugins: dkms-build-requires, priorities, update-motd, upgrade-helper, versionlock
amzn-main                                                                                         | 2.1 kB  00:00:00     
amzn-updates                                                                                      | 3.8 kB  00:00:00     
Package swig-2.0.10-4.24.amzn1.x86_64 already installed and latest version
Nothing to do
$ cd DOTA_devkit/
$ ls
DOTA2COCO.py               nms.py                  prepare_dota1_5.py            rotate.py
DOTA.py                    polyiou.cpp             prepare_dota1_aug.py          rotate_test.py
dota_utils.py              polyiou.h               prepare_dota1.py              setup.py
geojson2coco.py            polyiou.i               readme.md                     SplitOnlyImage_multi_process.py
HRSC2COCO.py               polyiou.py              ResultMerge_multi_process.py  SplitOnlyImage.py
ImgSplit_multi_process.py  poly_overlaps_test.py   ResultMerge.py                utils.py
ImgSplit.py                prepare_dota1_5_aug.py  results_obb2hbb.py
$ swig -c++ -python polyiou.i
$ python setup.py build_ext --inplace
running build_ext
building '_polyiou' extension
creating build
  ...
/home/ec2-user/SageMaker/AerialDetection/DOTA_devkit/_polyiou.cpython-36m-x86_64-linux-gnu.so
$
```



> d. Compile cuda extensions.
>
> ```bash
> ./compile.sh
> ```

Refer to [compile.sh Fails to Compile](troubleshooting/compile_sh_fails_to_compile.md). 

```
Notice
You can run python(3) setup.py develop or pip install -e . to install AerialDetection if you want to make modifications to it frequently.

If there are more than one AerialDetection on your machine, and you want to use them alternatively. Please insert the following code to the main file

import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
or run the following command in the terminal of corresponding folder.

export PYTHONPATH=`pwd`:$PYTHONPATH
```

