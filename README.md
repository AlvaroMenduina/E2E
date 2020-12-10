# E2E - HARMONI End-to-End Model Performance Analysis :octocat:

Welcome to the **E2E** GitHub repo: a set of performance analysis simulations for the HARMONI End-to-End Model, an optical model of the complete system including all subsystems: Focal Plane Relay System, Preoptics, Integral Field Unit, Spectrograph, Detectors :telescope:. This repo contains Python code that uses the Zemax Optics Studio (ZOS) API to analyse the HARMONI E2E files.

### Note: This project is still under construction 🚧

## What do we want to achieve? (*project rationale*)

*What are the End-to-End Models?* The E2E models are optical models in Zemax Optics Studio that represent the complete HARMONI instrument. They are created by joining together optical models of each subsystem provided by different teams within the [HARMONI consortium](https://auditore.cab.inta-csic.es/harmoni/consortium/) all across Europe :gb: :es: :fr: :de:. One of the main roles of the E2E models is to _demonstrate that the performance of HARMONI will meet the requirements_. Given the complexity of the HARMONI instrument and the myriad of possible configurations, this task cannot be done manually with a handful of Zemax analyses. This is where this repo comes in.

*What does the E2E repo do?* At the University of Oxford we are in charge of putting together the E2E models and **analysing their performance**; and this is where I come in. The E2E repo contains a bunch of Python code to automate this task, including custom performance analyses designed to match the peculiarities of HARMONI, as well as post-processing routines to plot and summarise the instrument's performance.

*Why is this needed?* An E2E model file represents a single instance of HARMONI for a specific choice of system parameters. But HARMONI is a workhorse instrument designed to work over a wide parameters space and has:

* **Spaxel Scales**: 4 indepedent choices of 4x4, 10x10, 20x20, 60x30 [milliarcseconds].
* **Spectral Bands**: 11 different gratings covering multiple bands VIS, IZJ, IZ, Z_HIGH, J, H, K, HK, H_HIGH, K_LONG, K_SHORT.
* **Integral Field Unit Channel**: 4 pairs of independent IFU channels AB, CD, EF, GH, each leading to a different spectrograph + detector.
* **Adaptive Optics modes**: 4 choices of Adapative Optics NOAO, SCAO, LTAO, HCAO (High Contrast).

This means that just for the nominal design, we can create 704 separate E2E files. Consequently, we need a toolbox capable of running a set of performance analyses across many files. Moreover, if we look at Monte Carlo analyses, where each subsystem provides not just its nominal file, but around a hundred random instances, the possibilities are almost endless.

Analysis implemented so far include:

* **Raytracing** (general): to validate the field definitions along the optical path across all spaxel scales
* **RMS Wavefront Error**: can be calculated at any arbitrary surface with custom sampling
* **FWHM PSF**: geometric PSF based on raytracing + diffraction effects + detector effects
* **Ensquared Energy**: geometric value in spatial direction
* **Spot Diagrams**: see example below at the Detector Plane

As an example, we can calculate the RMS Wavefront Error map at the detector plane for all 4 IFU channels for a given set of spaxel scale, spectral band, system configuration, etc.

![RMS WFE](images/rms_wfe_map.png?raw=true)

or we can display the results using box and whisker plots to compare the RMS Wavefront Error across multiple spectral bands

![BoxRMS](images/boxplot.png?raw=true "Boxplot")

We can also calculate Spot Diagrams at the detector plane:

![Spot Diagrams Detector Plane](images/sample_detector_spots.png?raw=true "Detector")
