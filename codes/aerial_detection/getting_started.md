

## Prepare DOTA dataset.

It is recommended to symlink the dataset root to `AerialDetection/data`. Here, we give an example for single scale data preparation of DOTA-v1.0. First, make sure your initial data are in the following structure.

```bash
data/dota
├── train
│   ├──images
│   └── labelTxt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```

### Symbolic link

```bash
$ ln -s /home/ec2-user/SageMaker/data /home/ec2-user/SageMaker/AerialDetection/data
```



Checking the available disk space.

The DOTA dataset has 6,555 items totaling 21.3GB.

```bash
sh-4.2$ df -h
Filesystem      Size  Used Avail Use% Mounted on
devtmpfs         30G   72K   30G   1% /dev
tmpfs            30G     0   30G   0% /dev/shm
/dev/xvda1      109G   80G   29G  74% /
/dev/xvdf       4.9G  988M  3.7G  22% /home/ec2-user/SageMaker
sh-4.2$ 
```

