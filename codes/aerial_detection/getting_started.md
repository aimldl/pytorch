

## [AerialDetection](https://github.com/dacon-ai/AerialDetection)/[GETTING_STARTED.md](https://github.com/dacon-ai/AerialDetection/blob/master/GETTING_STARTED.md)

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



> Inference with pretrained models
> Test a dataset
>
>  single GPU testing
>  multiple GPU testing
> You can use the following commands to test a dataset.
>
> single-gpu testing
>
> python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]
>
> multi-gpu testing
>
> ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
>
> Examples:
>
> Assume that you have already downloaded the checkpoints to work_dirs/.
>
> Test Faster R-CNN.
>
> ```bash
> $ python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py \
>     work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth \ 
>     --out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/results.pkl
> ```



Problem1

When I opened a new terminal, `ModuleNotFoundError` has occured.

```bash
sh-4.2$ python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth --out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/results.pkl
Traceback (most recent call last):
  File "tools/test.py", line 6, in <module>
    import mmcv
ModuleNotFoundError: No module named 'mmcv'
sh-4.2$
```

Solution1

`mmcv` is installed in Conda virtual environment `pytorch_p36`. Activating the environment resolves the problem.

```bash
sh-4.2$ source activate pytorch_p36
(pytorch_p36) sh-4.2$
```

Problem2

```
$ python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth --out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/results.pkl
Traceback (most recent call last):
  File "tools/test.py", line 12, in <module>
    from mmdet.apis import init_dist
  File "/home/ec2-user/SageMaker/AerialDetection/mmdet/apis/__init__.py", line 2, in <module>
    from .train import train_detector
  File "/home/ec2-user/SageMaker/AerialDetection/mmdet/apis/train.py", line 12, in <module>
    from mmdet import datasets
  File "/home/ec2-user/SageMaker/AerialDetection/mmdet/datasets/__init__.py", line 16
    from .ROKSI2020 import ROKSI2020, ROKSI2020Dataset_v3x`
                                                          ^
SyntaxError: invalid syntax
```

