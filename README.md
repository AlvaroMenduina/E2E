# E2E - HARMONI End-to-End Model Performance Analysis

A set of performance analysis simulations for the HARMONI End-to-End Model, an optical model of the complete system including all subsystem: Focal Plane Relay System, Preoptics, Integral Field Unit, Spectrograph, Detectors.

Using the Zemax Optics Studio (ZOS) API in Python we analyse the E2E Zemax files for HARMONI across:

* **Spaxel Scales** [milliarcseconds]: 4x4, 10x10, 20x20, 60x30
* **Spectral Bands**: VIS, IZJ, IZ, Z_HIGH, J, H, K, HK, H_HIGH, K_LONG, K_SHORT
* **Integral Field Unit Channel**: 4 pairs of channels AB, CD, EF, GH
* **Adaptive Optics modes**: NOAO, SCAO, LTAO, HCAO (High Contrast)
* **System Configurations**: Integral Field Spectrograph (IFS), full HARMONI, and complete with E-ELT

The shear number of possible combinations of spaxel scales, spectral bands, type of analysis, etc, calls for an automated approach to performance analysis. This package provides such functionality

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
