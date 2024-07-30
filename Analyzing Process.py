#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from pandas.core.frame import DataFrame

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
get_ipython().run_line_magic('matplotlib', 'inline')

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
#from astropy.timeseries import PhaseDispersionMinimization as PDM

from PyAstronomy.pyTiming import pyPDM

import emcee

import corner

from sklearn.utils import resample

from statsmodels.tsa.ar_model import AutoReg

def analyze_light_curve(filepath, O, A, T, theta):
    data = pd.read_csv(filepath)

    # light curve
    dates = data['Julian Date']
    photon_flux = data['Photon Flux [0.1-100 GeV](photons cm-2 s-1)']
    photon_flux_error = data['Photon Flux Error(photons cm-2 s-1)']

    julian_dates = dates.values
    flux = photon_flux.values
    flux_error = photon_flux_error.values

    plt.figure(figsize=(12, 6))
    plt.errorbar(dates, photon_flux, yerr=photon_flux_error, fmt='.k', capsize=3, label='Data')
    plt.xlabel('Julian Date')
    plt.ylabel('Photon Flux [0.1-100 GeV] (photons cm$^{-2}$ s$^{-1}$)')
    plt.title('Light Curve for Imported AGN')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the Lomb-Scargle periodogram
    frequency, power = LombScargle(julian_dates, flux, flux_error).autopower()

    # Convert frequency to period in years
    periods_years = 1 / frequency / 365.25

    # Select periods between 1 and 10 years
    valid_indices = (periods_years > 1) & (periods_years < 10)
    valid_periods = periods_years[valid_indices]
    valid_power = power[valid_indices]

    # Identify the best period
    best_index = np.argmax(valid_power)
    best_period_years = valid_periods[best_index]

    # Calculate the false alarm probability
    fap = LombScargle(julian_dates, flux, flux_error).false_alarm_probability(valid_power[best_index])

    # Determine confidence levels
    confidence_levels = [0.1, 0.05, 0.01]  # for 90%, 95%, 99% confidence levels
    fap_confidence_levels = LombScargle(julian_dates, flux, flux_error).false_alarm_level(confidence_levels)

    # Plot the periodogram with the confidence levels
    plt.figure(figsize=(12, 6))
    plt.plot(periods_years, power, label='Lomb-Scargle Power')
    plt.axvline(best_period_years, color='r', linestyle='--', label=f'Detected Period: {best_period_years:.4f} years')
    plt.xscale('log')
    plt.xlabel('Period (years) [Log scale]')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram')

    # Plot confidence levels
    for level, fap_level in zip(confidence_levels, fap_confidence_levels):
        plt.axhline(y=fap_level, linestyle='--', label=f'{(1-level)*100:.0f}% Confidence Level')

    plt.legend()
    plt.grid()
    plt.show()

    print(f"Best Period (1 to 10 years): {best_period_years:.4f} years")

    if fap < 0.001:
        print("Significance Level: >4σ")
    else:
        print(f"Significance Level: {sigma_level:.2f}σ")

    for level, fap_level in zip(confidence_levels, fap_confidence_levels):
        print(f"Power threshold for {(1-level)*100:.0f}% confidence level: {fap_level:.4e}")

    # Bootstrap
    min_period = 1  # in years
    max_period = 10  # in years

    # Convert period range to frequency range
    min_frequency = 1 / (max_period * 365.25)
    max_frequency = 1 / (min_period * 365.25)

    # Compute Lomb-Scargle periodogram
    ls = LombScargle(julian_dates, flux, flux_error)
    frequency, power = ls.autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)

    # Bootstrap significance analysis
    n_bootstrap = 1000
    bootstrap_powers = np.zeros((n_bootstrap, len(frequency)))

    for i in range(n_bootstrap):
        resampled_indices = np.random.choice(len(flux), size=len(flux), replace=True)
        resampled_flux = flux[resampled_indices]
        resampled_flux_error = flux_error[resampled_indices]
        bootstrap_powers[i] = LombScargle(julian_dates, resampled_flux, resampled_flux_error).power(frequency)

    significance_threshold = np.percentile(bootstrap_powers, 95, axis=0)

    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency
    best_period_years = best_period / 365.25

    bootstrap_best_frequencies = [frequency[np.argmax(bootstrap_powers[i])] for i in range(n_bootstrap)]
    bootstrap_best_periods = 1 / np.array(bootstrap_best_frequencies)
    bootstrap_best_periods_years = bootstrap_best_periods / 365.25

    period_uncertainty_years = np.std(bootstrap_best_periods_years)

    # Monte Carlo simulations for FAP
    n_random = 1000
    random_max_powers = np.zeros(n_random)
    for i in range(n_random):
        random_flux = np.random.permutation(flux)
        frequency_random, power_random = LombScargle(julian_dates, random_flux, flux_error).autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)
        random_max_powers[i] = np.max(power_random)

    # Calculate FAP
    observed_max_power = np.max(power)
    fap = np.sum(random_max_powers >= observed_max_power) / n_random

    # Converting FAP to sigma
    sigma_level = np.sqrt(2) * scipy.special.erfcinv(fap * 2)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(1 / frequency / 365.25, power, label='Lomb-Scargle Power')
    plt.plot(1 / frequency / 365.25, significance_threshold, linestyle='--', label='95% Significance Level')
    plt.axvline(best_period_years, color='r', linestyle='--', label=f'Detected Period: {best_period_years:.4f} years')
    plt.xscale('log')
    plt.xlabel('Period (years) [Log scale]')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram with Bootstrap Significance')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Best Period: {best_period_years:.4f} years")
    print(f"Period Uncertainty: {period_uncertainty_years:.4f} years")
    print(f"False Alarm Probability (FAP): {fap:.3f}")
    if fap < 0.001:
        print("Significance Level: >4σ")
    else:
        print(f"Significance Level: {sigma_level:.2f}σ")

    # Simulating LCs
    frequency, power = LombScargle(julian_dates, flux, flux_error).autopower()

    # Function to simulate light curves
    def simulate_light_curve(julian_dates, flux, n_simulations=1000):
        simulated_lcs = []
        psd_freq, psd_power = welch(flux)

        for _ in range(n_simulations):
            simulated_flux = np.random.normal(np.mean(flux), np.std(flux), len(flux))
            simulated_flux_psd = np.interp(np.fft.fftfreq(len(julian_dates)), psd_freq, psd_power)
            simulated_flux_fft = np.fft.fft(simulated_flux)
            simulated_flux_ifft = np.fft.ifft(simulated_flux_fft * np.sqrt(simulated_flux_psd)).real
            simulated_lcs.append(simulated_flux_ifft)

        return simulated_lcs

    # Number of simulations
    n_simulations = 1000
    # Simulate light curves
    simulated_lcs = simulate_light_curve(julian_dates, flux, n_simulations)

    # Compute Lomb-Scargle periodograms for the simulated light curves
    simulated_powers = []
    for simulated_flux in simulated_lcs:
        sim_frequency, sim_power = LombScargle(julian_dates, simulated_flux).autopower()
        simulated_powers.append(sim_power)

    simulated_powers = np.array(simulated_powers)

    # Combine original and simulated powers
    all_powers = np.vstack([power, simulated_powers])

    # Convert frequencies to periods in years and filter them
    periods_years = 1 / frequency / 365.25
    mask = (periods_years >= 1) & (periods_years <= 10)
    filtered_periods_years = periods_years[mask]
    filtered_power = power[mask]
    filtered_frequency = frequency[mask]

    # Find the best period within the filtered range
    combined_max_power_index = np.argmax(np.mean(all_powers[:, mask], axis=0))
    best_frequency = filtered_frequency[combined_max_power_index]
    best_period = 1 / best_frequency
    best_period_years = best_period / 365.25

    # Find the best periods from simulated data within the filtered range
    simulated_best_frequencies = [filtered_frequency[np.argmax(sim_power[mask])] for sim_power in simulated_powers]
    simulated_best_periods = 1 / np.array(simulated_best_frequencies)
    simulated_best_periods_years = simulated_best_periods / 365.25
    period_uncertainty_years = np.std(simulated_best_periods_years)

    # Calculate FAP
    observed_max_power = np.max(simulated_powers)
    random_max_powers = np.max(simulated_powers[:, mask], axis=1)
    fap = np.sum(random_max_powers >= observed_max_power) / n_simulations

    # Convert FAP to sigma
    sigma_level = np.sqrt(2) * scipy.special.erfcinv(fap * 2)

    # Plot the periodogram
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_periods_years, filtered_power, label='Original Lomb-Scargle Power')
    plt.plot(filtered_periods_years, np.mean(simulated_powers[:, mask], axis=0), linestyle='--', label='Average Simulated Power')
    plt.axvline(best_period_years, color='r', linestyle='--', label=f'Detected Period: {best_period_years:.4f} years')
    plt.xscale('log')
    plt.xlabel('Period (years) [Log scale]')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram for OJ 014 with Simulated Light Curves')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(17, 10))
    for sim_power in simulated_powers:
        plt.plot(1 / frequency / 365.25, sim_power, color='dimgrey', alpha=0.1)

    plt.plot(1 / frequency / 365.25, power, label='Original Lomb-Scargle Power', color='deepskyblue')
    plt.plot(1 / frequency / 365.25, np.mean(simulated_powers, axis=0), linestyle='--', label='Average Simulated Power', color='orange')
    plt.axvline(best_period_years, color='r', linestyle='--', label=f'Detected Period: {best_period_years:.4f} years')
    plt.xscale('log')
    plt.xlabel('Period (years) [Log scale]')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram for OJ 014 with Simulated Light Curves')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Best Period: {best_period_years} years")
    print(f"Period Uncertainty: {period_uncertainty_years} years")
    print(f"False Alarm Probability (FAP): {fap:.3f}")

    if fap < 0.001:
        print("Significance Level: >4σ")
    else:
        print(f"Significance Level: {sigma_level:.2f}σ")

    # Redfit
    def detrend(data):
        t = np.arange(len(data))
        p = np.polyfit(t, data, 1)
        return data - np.polyval(p, t)

    detrended_flux = detrend(photon_flux)

    # Fit AR(1) model
    model = AutoReg(detrended_flux, lags=1, old_names=False).fit()
    ar1_params = model.params
    ar1 = ar1_params[1]

    # Lomb-Scargle periodogram
    def lomb_scargle(julian_date, flux):
        f, pgram = LombScargle(julian_date, flux, flux_error).autopower()
        return f, pgram

    frequencies, pgram = lomb_scargle(julian_dates, detrended_flux)

    # Filter periods within the range 1 to 10 years
    valid_indices = (1 / frequencies / 365.25 > 1) & (1 / frequencies / 365.25 < 10)
    valid_frequencies = frequencies[valid_indices]
    valid_pgram = pgram[valid_indices]

    # Monte Carlo simulations to determine confidence levels
    n_simulations = 1000
    pgrams = np.zeros((n_simulations, len(valid_frequencies)))
    for i in range(n_simulations):
        red_noise = np.random.normal(size=len(detrended_flux))
        for j in range(1, len(red_noise)):
            red_noise[j] += ar1 * red_noise[j-1]
        _, pgram_sim = lomb_scargle(julian_dates, red_noise)
        pgrams[i, :] = pgram_sim[valid_indices]

    # Compute significance levels
    confidence_levels = np.percentile(pgrams, [90, 95, 99], axis=0)

    periods_years = 1 / valid_frequencies / 365.25

    best_period_index = np.argmax(valid_pgram)
    best_period = 1 / valid_frequencies[best_period_index] / 365.25

    # Estimate the uncertainty of the best period
    half_max_power = valid_pgram[best_period_index] / 2

    # Find left index for half maximum power
    left_indices = np.where(valid_pgram[:best_period_index] < half_max_power)[0]
    if left_indices.size > 0:
        left_index = left_indices[-1]
    else:
        left_index = 0

    # Find right index for half maximum power
    right_indices = np.where(valid_pgram[best_period_index:] < half_max_power)[0]
    if right_indices.size > 0:
        right_index = right_indices[0] + best_period_index
    else:
        right_index = len(valid_pgram) - 1

    # Calculate the uncertainty as the width at half maximum power
    period_left = 1 / valid_frequencies[left_index] / 365.25
    period_right = 1 / valid_frequencies[right_index] / 365.25
    best_period_uncertainty = (period_right - period_left) / 2

    plt.figure(figsize=(10, 6))
    plt.plot(periods_years, valid_pgram, label='Lomb-Scargle Periodogram')
    plt.plot(periods_years, confidence_levels[0], '--', label='90% Confidence Level')
    plt.plot(periods_years, confidence_levels[1], '--', label='95% Confidence Level')
    plt.plot(periods_years, confidence_levels[2], '--', label='99% Confidence Level')
    plt.xlabel('Period (years)')
    plt.ylabel('Power')
    plt.title('REDFIT Analysis')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # Identify significant peaks using the 90% confidence level
    significant_indices_90 = np.where(valid_pgram > confidence_levels[0])[0]

    significant_periods_90 = []
    uncertainties_90 = []

    for index in significant_indices_90:
        # Get the period corresponding to the peak
        period = 1 / valid_frequencies[index]
        significant_periods_90.append(period)

        # Find the full width at half maximum (FWHM)
        half_max_power = valid_pgram[index] / 2

        # Find left index for half maximum power
        left_indices = np.where(valid_pgram[:index] < half_max_power)[0]
        if left_indices.size > 0:
            left_index = left_indices[-1]
        else:
            left_index = 0

        # Find right index for half maximum power
        right_indices = np.where(valid_pgram[index:] < half_max_power)[0]
        if right_indices.size > 0:
            right_index = right_indices[0] + index
        else:
            right_index = len(valid_pgram) - 1

        # Calculate the uncertainty as the width at half maximum power
        period_left = 1 / valid_frequencies[left_index]
        period_right = 1 / valid_frequencies[right_index]
        period_uncertainty = (period_right - period_left) / 2
        uncertainties_90.append(period_uncertainty)

    # Convert periods and uncertainties to years
    significant_periods_90_years = np.array(significant_periods_90) / 365.25
    uncertainties_90_years = np.array(uncertainties_90) / 365.25

    # Combine periods and uncertainties into a DataFrame
    periods_uncertainties_90_df = pd.DataFrame({
        'Period (years)': significant_periods_90_years,
        'Uncertainty (years)': uncertainties_90_years
    })

    print(periods_uncertainties_90_df)

    # Calculate FAP
    observed_max_power = np.max(valid_pgram)
    random_max_powers = np.max(pgrams, axis=1)
    fap = np.sum(random_max_powers >= observed_max_power) / n_simulations

    # Convert FAP to sigma
    sigma_level = np.sqrt(2) * scipy.special.erfcinv(fap * 2)

    print(f"Best Period: {best_period:.2f} years")
    print(f"Period Uncertainty: ±{best_period_uncertainty:.2f} years")
    print(f"False Alarm Probability (FAP): {fap:.3f}")

    if fap < 0.001:
        print("Significance Level: >4σ")
    else:
        print(f"Significance Level: {sigma_level:.2f}σ")

    x = data['Julian Date']
    y = data['Photon Flux [0.1-100 GeV](photons cm-2 s-1)']

    def pdm_analysis(x, y):
        S = pyPDM.Scanner(minVal=1, maxVal=4.8, dVal=0.1, mode="period")
        P = pyPDM.PyPDM(x, y)
        f2, t2 = P.pdmEquiBinCover(10, 3, S)
        best_period = f2[np.argmin(t2)]
        return best_period

    best_period_initial = pdm_analysis(x, y)

    n_bootstrap = 1000
    best_periods = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = np.random.randint(0, len(x), len(x))
        x_resampled = x.values[indices]
        y_resampled = y.values[indices]

        best_periods[i] = pdm_analysis(x_resampled, y_resampled)

    period_mean = np.mean(best_periods)
    period_std = np.std(best_periods)

    confidence_level = 0.95
    z_score = 1.96  # For 95% confidence level

    confidence_interval = z_score * period_std
    sigma_confidence = confidence_interval / period_std

    S = pyPDM.Scanner(minVal=1, maxVal=4.8, dVal=0.1, mode="period")
    P = pyPDM.PyPDM(x, y)
    f2, t2 = P.pdmEquiBinCover(10, 3, S)

    plt.figure(facecolor='white')
    plt.title("Result of PDM analysis")
    plt.xlabel("Period")
    plt.ylabel("Theta")
    plt.plot(f2, t2, 'gp-')
    plt.legend(["PDM"])
    plt.show()

    print(f"Best period: {best_period_initial:.3f} years")
    print(f"Uncertainty: ±{period_std:.3f} years")
    print(f"Confidence level: {sigma_confidence:.3f}")
    
    # MCMC

    def sinusoidal_model(t, O, A, T, theta):
        return O + A * np.sin(2 * np.pi * t / T + np.deg2rad(theta))

    def log_likelihood(params, t, y, yerr):
        O, A, T, theta = params
        model = sinusoidal_model(t, O, A, T, theta)
        sigma2 = yerr**2
        return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    def log_prior(params):
        O, A, T, theta = params
        if 0 < O < 150e-6 and 0 < A < 80e-6 and 1 < T < 10 and 0 < theta < 360:
            return 0.0
        return -np.inf

    def log_probability(params, t, y, yerr):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, t, y, yerr)

    initial = np.array([O, A, T, theta])
    nwalkers = 32
    ndim = len(initial)
    pos = initial + 1e-6 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(julian_dates, flux, flux_error))

    sampler.run_mcmc(pos, 5000, progress=True)

    samples = sampler.get_chain(discard=1000, thin=15, flat=True)

    O_mcmc, A_mcmc, T_mcmc, theta_mcmc = np.percentile(samples, 50, axis=0)
    O_err, A_err, T_err, theta_err = np.std(samples, axis=0)
    
    fig = corner.corner(samples, labels=["O", "A", "T", "theta"], truths=[O_mcmc, A_mcmc, T_mcmc, theta_mcmc])
    plt.show()

    plt.figure(figsize=(17, 10))
    plt.errorbar(julian_dates, flux, yerr=flux_error, fmt=".k", label="Observed Data")
    t_fit = np.linspace(min(julian_dates), max(julian_dates), 1000)
    plt.plot(t_fit, sinusoidal_model(t_fit, O_mcmc, A_mcmc, T_mcmc, theta_mcmc), label="Best-Fit Model")
    plt.xlabel("Julian Date")
    plt.ylabel("Photon Flux [0.1-100 GeV] (photons cm-2 s-1)")
    plt.legend()
    plt.show()

    print(f"Best-fit parameters: O = {O_mcmc}, A = {A_mcmc}, T = {T_mcmc}, theta = {theta_mcmc}")
    print(f"Parameter uncertainties: O_err = {O_err}, A_err = {A_err}, T_err = {T_err}, theta_err = {theta_err}")

    print(f"Best Period (1 to 10 years): {T_mcmc} years")
    print(f"Period Uncertainty: {T_err} years")

# Example usage, the four value after the file path should be the inital value for MCMC fit
analyze_light_curve('/4FGL_J1555.7+1111_monthly_7_10_2024.csv', 75e-6, 51e-6, 2.201491, 190)


# In[ ]:




