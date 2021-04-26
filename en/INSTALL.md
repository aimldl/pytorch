* Draft: 2020-10-16 (Fri)

# Install PyTorch

## Standard installation

[GET STARTED](https://pytorch.org/get-started/locally/) > [START LOCALLY](https://pytorch.org/get-started/locally/#start-locally)

```bash
$ sudo apt install python
```

## Docker installation

The official pytorch image released by pytorch is available at [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)

Assuming docker has been installed already,


```bash
$ docker pull pytorch/pytorch
```

The full message looks like:

```bash
$ docker pull pytorch/pytorch
Using default tag: latest
latest: Pulling from pytorch/pytorch
23884877105a: Pull complete 
bc38caa0f5b9: Pull complete 
2910811b6c42: Pull complete 
36505266dcc6: Pull complete 
3472d01858ba: Pull complete 
4a98b57681ff: Pull complete 
f3b419d1e6d5: Pull complete 
Digest: sha256:9c3aa4653f6fb6590acf7f49115735be3c3272f4fa79e5da7c96a2c901631352
Status: Downloaded newer image for pytorch/pytorch:latest
docker.io/pytorch/pytorch:latest
$
```

To verify if the `pytorch/pytorch` exists in the local machine, run:

```bash
$ docker images
REPOSITORY                            TAG                        IMAGE ID            CREATED             SIZE
  ...
pytorch/pytorch                       latest                     6a2d656bcf94        2 months ago        3.47GB
  ...
$
```

To run the image, use the `docker run` command.

```bash
$ docker run -it pytorch/pytorch bash
root@8b185ecaa66e:/workspace# 
```

Check to see if the PyTorch package exists in the container.

```bash
root@8b185ecaa66e:/workspace# python
Python 3.8.8 (default, Feb 24 2021, 21:46:12) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> 
```

Of course, `import torch` does not spit any error.

However `nvidia-smi` is not available.

```bash
root@8b185ecaa66e:/workspace# nvidia-smi
bash: nvidia-smi: command not found
root@8b185ecaa66e:/workspace# 
>>>
```

Exiting the `pytorch/pytorch` container...

```bash
root@8b185ecaa66e:/workspace# exit
exit
$
```

