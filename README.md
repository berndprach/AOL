
# Almost-Orthogonal Lipschitz (AOL) Layers

Code for paper 
[Almost-Orthogonal Layers for Efficient
General-Purpose Lipschitz Networks](https://arxiv.org/abs/2208.03160). 
It includes code for the layers we propose,
[AOL-Dense](src/aol_code/layers/aol/aol_dense.py)
and
[AOL-Conv2D](src/aol_code/layers/aol/aol_conv2d.py)
as well as code for models, our proposed loss function
and metrics.

## Requirements:
- Python 3.8
- Tensorflow 2.7

## How to train an AOL-Network
Specify the setting using the Run-ID.
For example in order to train AOL-Small on CIFAR10 use

    python train_aol_network.py 0

Run-IDs correspond to the following settings:

| Run ID | Model      | Dataset  |  Loss Offset   |
|:-------|:-----------|:---------|:--------------:|
| 0      | AOL-Small  | CIFAR10  |   $\sqrt{2}$   |
| 1      | AOL-Medium | CIFAR10  |   $\sqrt{2}$   |
| 2      | AOL-Large  | CIFAR10  |   $\sqrt{2}$   |
| 10     | AOL-FC     | CIFAR10  |   $\sqrt{2}$   |
| 11     | AOL-CONV   | CIFAR10  |   $\sqrt{2}$   |
| 12     | AOL-STD    | CIFAR10  |   $\sqrt{2}$   |
| 20     | AOL-Small  | CIFAR10  | $\sqrt{2}$/16  |
| 21     | AOL-Small  | CIFAR10  |  $\sqrt{2}$/4  |
| 22     | AOL-Small  | CIFAR10  |   $\sqrt{2}$   |
| 23     | AOL-Small  | CIFAR10  |  4 $\sqrt{2}$  |
| 24     | AOL-Small  | CIFAR10  | 16 $\sqrt{2}$  |
| 30     | AOL-Small  | CIFAR100 |   $\sqrt{2}$   |
| 31     | AOL-Medium | CIFAR100 |   $\sqrt{2}$   |
| 32     | AOL-Large  | CIFAR100 |   $\sqrt{2}$   |



## Results
The results for Cifar10 and Cifar100 can be found
in the tables below.
We report the standard accuracy (Std Acc) as well as the
Certified Robust Accuracy (CRA) for different amounts of 
input perturbations. 

| CIFAR 10   |   Std Acc | CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:-----------|----------:|-----------:|-----------:|------------:|------:|
| AOL-Small  |     70.3% |      62.9% |      55.0% |       47.7% | 22.3% |
| AOL-Medium |     71.3% |      64.0% |      56.1% |       49.1% | 23.3% |
| AOL-Large  |     71.4% |      64.1% |      56.5% |       49.4% | 23.8% |

Results on CIFAR100:

| CIFAR 100  | Std Acc | CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:-----------|--------:|-----------:|-----------:|------------:|------:|
| AOL-Small  |   42.4% |      32.5% |      24.8% |       19.2% |  6.7% |
| AOL-Medium |   43.2% |      33.7% |      26.0% |       20.2% |  7.2% |
| AOL-Large  |   43.7% |      33.7% |      26.3% |       20.7% |  7.8% |

### Ablation Studies
We also report the results for the two ablation studies.
We trained the AOL-Small model on CIFAR10 with different
offset for the loss function. (We also rescaled the temperature
parameter proportionally.)
The results are in the table below.

| Loss Offset   | Std Acc | CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:--------------|--------:|-----------:|-----------:|------------:|------:|
| $\sqrt{2}$/16 |   80.3% |      46.4% |      17.5% |        3.6% |  0.0% |
| $\sqrt{2}$/4  |   77.8% |      63.0% |      47.5% |       32.8% |  2.2% |
| $\sqrt{2}$    |   70.6% |      62.8% |      54.5% |       47.7% | 22.5% |
| 4 $\sqrt{2}$  |   59.9% |      55.0% |      50.6% |       46.3% | 31.1% |
| 16 $\sqrt{2}$ |   48.5% |      45.4% |      42.4% |       39.7% | 29.0% |

We also report the results for different models,
including a network that consisting purely of fully-connected 
layers (AOL-FC),
a relatively standard convolutional achitecture where the 
number of channels is multiplied by 2 whenever the resolutions
decreases (AOL-STD),
as well as
a convolutional networks that multiplies the number of channels
by 4 whenever the spatial resolution is decreases in order to
keep the number of activations constant for the first few layers 
(AOL-Conv):

| CIFAR 10 | Std Acc |    CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:---------|--------:|--------------:|-----------:|------------:|------:|
| AOL-FC   |   67.1% |         58.5% |      50.3% |       42.4% | 17.6% |
| AOL-STD  |   65.4% |         56.9% |      48.3% |       40.5% | 16.2% |
| AOL-Conv |   68.5% |         60.2% |      52.3% |       45.2% | 19.5% |

## Citations


