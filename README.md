# A Nonconvex Framework for Structured Dynamic Covariance Recovary


## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Setup](#setup)
* [Examples](#examples)
* [License](#license)
* [References](#references)

## Genral Info

This repo is the implementation of the paper [[1]](#1) along with the competing methods listed in Table 1 in [[1]](#1). Please check Table 1 for details.

## Requirements
Install the [ssm package](https://github.com/slinderman/ssm) to run comppeting methods.
It is recommended to install [Anaconda](https://docs.anaconda.com/anaconda/install/) and create a new environment (python>=3.7) to run this code.

```bash
conda create --name snscov python=3.7
conda activate snscov
```
## Setup

```bash
python setup.py install
```

## Examples
Some simulation examples. 


1. Run Simulation 1. Variations of waveforms
 
 
```bash
cd tests/simulation1
python simulation1.py 
```
Models are stored in folder `./results` and plots are stored in `./figures`. Specify the waveform by using `--waveform`

2. Run Simulation 2. Comparison with other mehtods


```bash
cd tests/comparison1
python compare_all.py
```

Test different number of test subjects by running the follow bash file.
Select the waveform (sine, square, mixing) by changing `--waveform` in `compare.sh`
```bash
cd tests/comparison1
bash compare.sh
```


3. Run Simulation 3. Comparison with other mehtods (high-dimensional data)
```bash
cd tests/comparison2
python test_large_scale.py
```



## License 

Distributed under the MIT License. See `LICENSE.txt` for more information.


## References
<a id="1">[1]</a> 
Tsai, K., Kolar, M., & Koyejo, O. (2020). 
A Nonconvex Framework for Structured Dynamic Covariance Recovery. 
arXiv preprint arXiv:2011.05601.
