
# Almost-Orthogonal Lipschitz (AOL) Layers

Code for paper 
[Almost-Orthogonal Layers for Efficient
General-Purpose Lipschitz Networks](). 
<!--- TODO: Add Link --->
It includes code for the layers we propose,
[AOL-Dense](src/framework/achitectures/layers/aol/aol_dense.py)
and
[AOL-Conv2D](src/framework/achitectures/layers/aol/aol_conv2d.py)
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
The code generates results slightly better than reported in the paper.
(This is because of using ConcatenationPooling instead of strided convolutions.)
We report the standard accuracy (Std Acc) as well as the
Certified Robust Accuracy (CRA) for different amounts of input perturbation.

| CIFAR 10   |   Std Acc | CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:-----------|----------:|-----------:|-----------:|------------:|------:|
| AOL-Small  |     70.3% |      62.9% |      55.0% |       47.7% | 22.3% |
| AOL-Medium |     71.3% |      64.0% |      56.1% |       49.1% | 23.3% |
| AOL-Large  |     71.4% |      64.1% |      56.5% |       49.4% | 23.8% |

Results on CIFAR100:

| CIFAR 100  | Std Acc | CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:-----------|--------:|-----------:|-----------:|------------:|------:|
| AOL-Small  |   56.1% |      47.3% |      39.1% |       31.7% | 11.2% |
| AOL-Medium |   57.8% |      48.2% |      39.8% |       32.1% | 11.1% |
| AOL-Large  |   57.4% |      48.3% |      39.3% |       31.7% | 10.6% |

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
a network consisting purely of fully-connected layers (AOL-FC),
a convolutional networks that keeps the number of activations
constant for the first few layers (AOL-Conv)
as well as a relatively standard convolutional achitecture,
where the number of channels is multiplied by 2 whenever the resolutions
decreases (AOL-STD):

| CIFAR 10 | Std Acc | CRA 36/255 | CRA 72/255 | CRA 108/255 | CRA 1 |
|:---------|--------:|-----------:|-----------:|------------:|------:|
| AOL-FC   |   62.4% |      52.7% |      43.8% |       35.2% | 12.1% |
| AOL-Conv |   68.6% |      60.6% |      53.1% |       45.4% | 20.3% |
| AOL-STD  |   61.7% |      52.3% |      42.7% |       34.7% | 11.1% |

## Citations


