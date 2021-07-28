# Maximum Entropy Biasing of Epidomiology Models

This python module implements [maximum entropy biasing](https://github.com/ur-whitelab/maxent) to modify epidomiology parameters to find the best fit to disease trajectory given a set of observations. These observations are time-averaged fractional values that can come from different compartments (S, E, A, I and R) of a known synthetic reference trajectory or real pandemic spread data.  

## Installation

The package uses Keras (Tensorflow), geopandas and networkx.
```sh
pip install maxent@git+https://github.com/ur-whitelab/maxent.git
```

## License

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

## Authors

maxentep was developed by [Mehrad Ansari](mehrad.ans@gmail.com), [Rainier Barrett](rbarret8@ur.rochester.edu) and [Andrew White](andrew.white@rochester.edu).
