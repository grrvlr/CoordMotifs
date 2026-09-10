# SubTSMD

***Discovering subspace motifs with temporal variations in multivariate time series.***

This project provides the code for SubTMSD, a method for discovering subspace motifs in multivariate time series. The key features of SubTSMD include: (1) being able to discover subspace motifs of any dimensionality, despite exponential number of possible subspaces, and (2) finding motifs with temporal variations across attributes, i.e., the motifs do not need to perfectly align in time. 

## Installation

This project was developed using Python 3.10.11, but it should also work with other versions, as long as the dependencies can be installed. To install SubTSMD, run the following command:
```bash 
pip install git+https://github.com/ML-KULeuven/SubTSMD
```
Alternatively, you can also clone the code and run and run the following command in the directory:
```bash
pip install .
```

## Usage

We offer an object-oriented interface to discover subspace motifs, as illustrated below. The example can also be found in the [example.ipynb](notebooks%2Fexample.ipynb) notebook. 

```python 
from sub_tsmd import SubTSMD, plot_motif_sets, load

# Load the data
X, y = load('data/synthetic/demonstration/subspace.pkl')

# Discover the subspace motifs
subspace_motif_discovery = SubTSMD(
    l_min=40,
    l_max=60,
    rho=0.65,
    warping=False
)
subspace_motifs = subspace_motif_discovery.apply(X)

# Plot the discovered motifs
plot_motif_sets(X, subspace_motifs)
```
![example.png](notebooks%2Fexample.png)

Above code relies on LoCoMotif for discovering motifs in the individual attributes. To use another motif discovery method, you can use the ``apply_sub_tsm`` as follows, in which you manually discover motifs in each attribute:

```python 
from sub_tsmd import apply_sub_tsmd

"""
Initialize the univariate motifs using your own motif discovery method.

A single motif set is defined as a tuple: (mask, indices), in which the mask 
defines the attributes on which the motif set appears and the indices are the 
time steps at which the motif appears. The indices have three dimensions: 
- Axis 0: the number of motifs occurring in the motif set
- Axis 1: the start and end index of the motifs (thus always dimension of 2)
- Axis 2: to which channel the indices correspond. Thus, for univariate motifs, 
  this should be of dimension 1. This is necessary to allow for temporal variations. 
  
For each attribute (or set of attributes), there can be multiple motif sets, which 
leads to a list of motif sets. 

The independent_motif_sets variable consequently equals a list of such lists of motif
sets, in which each inner list corresponds to the motifs on a specific attribute or
subspace. 
"""
independent_motif_sets = ...

subspace_motifs = apply_sub_tsmd(independent_motif_sets, delta=0.9)
```

## Contact

If you want to contribute, report bugs, or need help applying SubTSMD for your application, feel free to reach out by opening a GitHub issue or contacting us via email at [louis.carpentier@kuleuven.be](mailto:louis.carpentier@kuleuven.be).

## Data and experimental results

All data used within this work can be publicly accessed or generated. Check out [data/README.md](data/README.md) for more information. The scripts to reproduce our experiments are available in the [experiments](experiments) folder. We describe how to use these scripts in the [experiments/README.md](experiments/README.md) file.

## Acknowledgements

If you use SubTSMD in your work, please consider adding the following citation.
```bibtex
@article{carpentier2026subtsmd,
  title={{SubTSMD: discovering subspace motifs with temporal variations in multivariate time series}},
  author={Carpentier, Louis and Devos, Laurens and Meert, Wannes and Mathias, Verbeke},
  journal={Data Mining and Knowledge Discovery},
  doi = {10.1007/s10618-026-01218-7},
  year = {2026}
}
```
> Carpentier, L., Devos, L., Meert, W. et al. (2026) "SubTSMD: discovering subspace motifs with temporal variations in multivariate time series." _Data Mining and Knowledge Discovery_, doi: [10.1007/s10618-026-01218-7](https://doi.org/10.1007/s10618-026-01218-7).

## License

    MIT License
    
    Copyright (c) 2025 KU Leuven, DTAI Research Group
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
