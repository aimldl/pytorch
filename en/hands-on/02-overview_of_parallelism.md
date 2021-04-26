



When DDP is combined with model parallel, each DDP process would use model parallel, and all processes collectively would use data parallel.

### DataParallel

* is single-process, multi-thread, and only works on a single machine.
* [OPTIONAL: DATA PARALLELISM](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html), Sung Kim and Jenny Kang

### DistributedDataParallel (DDP)

* is multi-process and works for both single- and multi- machine training.
* [GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), Shen Li (Joe Zhu)

### [Comparison between `DataParallel` and `DistributedDataParallel`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#comparison-between-dataparallel-and-distributeddataparallel)

* `DistributedDataParallel` is usually faster than `DataParallel`
  *  even on a single machine due to:
    * GIL contention across threads,
    * per-iteration replicated model,
    * and additional overhead introduced by scattering inputs and gathering outputs.
* `DistributedDataParallel` is usually  more portable than `DataParallel`.
  * If your model is too large to fit on a single GPU, you must use **model parallel** to split it across multiple GPUs.
  * `DistributedDataParallel` works with **model parallel**; `DataParallel` does not at this time.

### Model parallelism

* [SINGLE-MACHINE MODEL PARALLEL BEST PRACTICES](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html), Shen Li
* 