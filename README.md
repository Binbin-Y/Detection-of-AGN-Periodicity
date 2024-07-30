# AGN Periodicity Detection - GitHub Guidelines
## Overview
This repository contains code for analyzing the periodicity of Active Galactic Nuclei (AGN) using six different algorithms. The primary script, analyze_process.py, processes light curve data to detect periodic signals, calculate false alarm probabilities, and estimate the confidence levels. The six different algorithms are Power-law Lomb-Scargle periodogram, Bootstrap Lomb-Scargle periodogram, Simulating LCs, REDFIT, Phase Dispersion Minimization, and Markov Chain Monte Carlo Sinusoidal Fitting.
## Getting Started
### Prerequisites
Ensure you have Python 3.7 or higher installed. You will also need the following Python packages:

•	pandas

•	numpy

•	matplotlib

•	scipy

•	astropy

•	emcee

•	corner

•	scikit-learn

•	statsmodels

Install the required packages using the command:

```pip install pandas numpy matplotlib scipy astropy emcee corner scikit-learn statsmodels```

### Usage
1.	Input Data: Place the file path of your gamma ray data provided by the Fermi Large Telescope in the data/ directory. The file should be in CSV format contains the following columns:

o	Julian Date

o	Photon Flux [0.1-100 GeV](photons cm-2 s-1)

o	Photon Flux Error(photons cm-2 s-1)

2.	Run the Analysis: Modify the example usage in the analyze_process.py script with the path to your input data file and initial parameter values for the MCMC fit. Then, run the script.

## Functions and Algorithms
### ```analyze_light_curve(filepath, O, A, T, theta)```

This function performs the complete analysis of the light curve data, including:

•	Plotting the light curve

•	Calculating the Lomb-Scargle periodogram

•	Identifying the best period and its significance for all six techniques

•	Performing bootstrap resampling for uncertainty estimation

•	Conducting MCMC simulations for parameter estimation

•	Plotting and reporting the results

Parameters:

•	```filepath``` (str): Path to the CSV file containing the light curve data.

•	```O``` (float): Initial offset value for MCMC fit.

•	```A``` (float): Initial amplitude value for MCMC fit.

•	```T``` (float): Initial period value (in years) for MCMC fit.

•	```theta``` (float): Initial phase angle (in degrees) for MCMC fit.

## Code Details

### Import Statements
The script begins with necessary import statements for data manipulation, numerical operations, plotting, and statistical analysis:

```
import pandas as pd
from pandas.core.frame import DataFrame

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
%matplotlib inline

import scipy
import scipy.optimize as opt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import lombscargle
from scipy.optimize import minimize
from scipy.stats import uniform
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.stats import norm

#from google.colab import drive
#drive.mount ('/content/gdrive')

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.timeseries import LombScargle # Lomb–Scargle periodogram (LSP)
from astropy.io import fits # convert the FITS format
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries

from PyAstronomy.pyTiming import pyPDM

import emcee

import corner

from sklearn.utils import resample

from statsmodels.tsa.ar_model import AutoReg
```

## Results
Results include plots and data summaries. Significant findings such as best-fit periods, uncertainties, and significance levels will be displayed in the console.
## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.
