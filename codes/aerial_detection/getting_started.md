

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

