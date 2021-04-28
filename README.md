maxentep
========
This python module implements maximum entropy method to modify epidomiology parameters to find the best fit to disease trajectory given a set of observations. These observations are time-averaged fractional values that can come from different compartments (S, E, A, I and R) of a known synthetic reference trajectory or real pandemic spread data.
Installation
------------
1. Clone the repository and change direcotry to maxentep:
```sh
git clone https://github.com/ur-whitelab/maxent-epidemiology.git
cd maxent-epidemiology
```

2. Install the module using pip:
```sh
pip install . -e
```
3. Run example notebooks.

License
-------
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

Authors
-------

maxentep was developed by [Mehrad Ansari](Mehrad.ansari@rochester.edu), [Rainier Barrett](rbarret8@ur.rochester.edu), and [Andrew White](andrew.white@rochester.edu).
