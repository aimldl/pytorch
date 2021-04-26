* Draft: 2021-04-26 (Mon)

# Setting up the Computing Environment





Pull the official PyTorch image.

```bash
(base) ~$ docker pull pytorch/pytorch
Using default tag: latest
latest: Pulling from pytorch/pytorch
92dc2a97ff99: Pull complete 
be13a9d27eb8: Pull complete 
c8299583700a: Pull complete 
70a80b9c7100: Pull complete 
9dda6e51b6e4: Pull complete 
ab1504e75c6c: Pull complete 
Digest: sha256:9ebb176339b25a2d155e6f127c5948968b3f61e5f720c4598ef79cf450db8bfe
Status: Downloaded newer image for pytorch/pytorch:latest
docker.io/pytorch/pytorch:latest
(base) ~$
```

Check if the image exists in the local machine.

```bash
$ docker images
REPOSITORY                           TAG                      IMAGE ID       CREATED         SIZE
pytorch/pytorch                      latest                   5ffed6c83695   4 weeks ago     7.25GB
$
```



```bash
$ docker run -it pytorch/pytorch /bash
```

