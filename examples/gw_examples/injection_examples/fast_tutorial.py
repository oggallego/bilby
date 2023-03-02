#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

import numpy as np
import bilby

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'fast_tutorial'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=2., mass_2=24., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108, lambdaG=1e15)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomZPHM',
                          reference_frequency=50., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 1,
    maximum=injection_parameters['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
            'dec', 'geocent_time', 'phase', 'lambdaG']:
    priors[key] = injection_parameters[key]
priors['lambdaG'] = bilby.core.prior.Uniform(minimum=1e15, maximum=1e19, name='lambdaG', latex_label='$\lambda_G$')
#priors['redshift'] = bilby.core.prior.Uniform(0,1)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='Nessai', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

lambdaG_sample = result.posterior['lambdaG']
print(lambdaG_sample[:10])
lambdaG_log = np.log(lambdaG_sample)
print(lambdaG_log[:10])

result.posterior['lambdaG'] = lambdaG_log
print(result.posterior['lambdaG'])


# Make a corner plot.
result.plot_corner()

#parameters = ["mass_ratio", "chirp_mass", "luminosity_distance", "theta_jn"]
#parameters.append(bilby.core.prior.Constraint(name="log_lambdaG", minimum=np.log10(1e15), maximum=np.log10(1e19)))
#result.plot_corner(parameters=parameters, truths=injection_parameters)

#print(result.priors.keys())

#result.plot_corner(parameters=["mass_ratio", "chirp_mass", "luminosity_distance", "theta_jn", "lambdaG"], include_log=True,
                   #log_args={'lambdaG': {'base': 10, 'label': r'$\log_{10}(\lambdaG)$'}})


#result.plot_corner(parameters=["mass_ratio", "chirp_mass", "luminosity_distance", "theta_jn", "lambdaG"],
                   #priors=priors, truths=[41.5, 32.5, 440, 0.4, np.log10(1e15)], show_titles=True,
                   #title_kwargs={'fontsize': 12}, labels=None,
                   #label_kwargs={'fontsize': 16}, truths_kwargs={'color': 'blue'},
                   #hist_kwargs={'density': True}, kde_kwargs={'cut': 3, 'bw': 'scott'},
                   #quantiles=[0.025, 0.5, 0.975], include_log=True,
                   #log_args={'lambdaG': {'base': 10, 'label': r'$\log_{10}(\lambdaG)$'}})

#result.plot_corner(parameters=["mass_ratio", "chirp_mass", "luminosity_distance", "theta_jn", "lambdaG"],
                    #truths=[41.5, 32.5, 440, 0.4, np.log10(1e10)],
                    #truth_color='red', show_titles=True, title_kwargs={'fontsize': 12})


#median = np.median(lambdaG_samples)
#lower, upper = np.percentile(lambdaG_samples, [5, 95])
#print(f"Median lambdaG = {median:.2f} ({lower:.2f} - {upper:.2f})")

# Set the range of the plot based on the 5th and 95th percentiles of the posterior
#lims = [(np.percentile(lambdaG_samples, q=5), np.percentile(lambdaG_samples, q=95))]