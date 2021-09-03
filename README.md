# Compartmental Epidomiology Modeling


![tests](https://github.com/ur-whitelab/py0/actions/workflows/tests.yml/badge.svg) ![docs](https://github.com/ur-whitelab/py0/actions/workflows/docs.yml/badge.svg)

``py0`` is a python implementation of compartmental disease modeling.

![](docs/source/img/py_0.gif)

## Installation

To install ``py0``:
```sh
pip install py0@git+https://github.com/ur-whitelab/py0.git
```

## Maximum Entropy Biasing

``py0`` can be coupled with [MaxEnt](https://ur-whitelab.github.io/maxent/) to modify epidomiology parameters to find the best fit to disease trajectory given a set of observations and also infer the true origin of the outbreak (patient-zero). These observations are time-averaged fractional values that can come from different compartments (S, E, A, I and R) of a known synthetic reference trajectory or real pandemic spread data. 

### Creating an ensmble of trajectories

![](docs/source/img/sampling.gif)

### Fitting using MaxEnt

![](docs/source/img/fit.gif)

### MaxEnt Installation

The package uses Keras (Tensorflow). To install:
```sh
pip install maxent@git+https://github.com/ur-whitelab/maxent.git
```

## License

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

## Authors

``py0`` is developed by [Mehrad Ansari](mehrad.ans@gmail.com), [Rainier Barrett](rbarret8@ur.rochester.edu) and [Andrew White](andrew.white@rochester.edu).
