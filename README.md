# E2E - HARMONI End-to-End Model Performance Analysis

A set of performance analysis simulations for the HARMONI End-to-End Model, an optical model of the complete system including all subsystem: Focal Plane Relay System, Preoptics, Integral Field Unit, Spectrograph, Detectors.

Using the Zemax Optics Studio (ZOS) API in Python we analyse the E2E Zemax files for HARMONI across:

* **Spaxel Scales** [milliarcseconds]: 4x4, 10x10, 20x20, 60x30
* **Spectral Bands**: VIS, IZJ, IZ, Z_HIGH, J, H, K, HK, H_HIGH, K_LONG, K_SHORT
* **Integral Field Unit Channel**: 4 pairs of channels AB, CD, EF, GH
* **Adaptive Optics modes**: NOAO, SCAO, LTAO, HCAO (High Contrast)
* **System Configurations**: Integral Field Spectrograph (IFS), full HARMONI, and complete with E-ELT

Analysis implemented so far include:

* **Raytracing** (general): to validate the field definitions along the optical path across all spaxel scales
* **RMS Wavefront Error**: can be calculated at any arbitrary surface with custom sampling
* **FWHM PSF**: geometric PSF based on raytracing + diffraction effects + detector effects
* **Ensquared Energy**: geometric value in spatial direction
* **Spot Diagrams**: see example below at the Detector Plane

![Spot Diagrams Detector Plane](sample_detector_spots.png?raw=true "Detector")
