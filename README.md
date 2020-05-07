# E2E - HARMONI End-to-End Model Performance Analysis

A set of performance analysis simulations for the HARMONI End-to-End Model, using the Python API for Zemax Optics Studio.


Using the Zemax Optics Studio (ZOS) API in Python we analyse the E2E Zemax files for HARMONI across all spaxel scales (4x4, 10x10, 20x20, and 60x30 milliarcseconds), spectral bands / grating configurations (VIS, IZJ, IZ, Z_HIGH, J, H, K, HK, H_HIGH, K_LONG, K_SHORT), adaptive optics modes (NOAO, SCAO, LTAO, HCAO), and system configurations (IFS, HARMONI, E-ELT).

Another possible analysis is the Spot Diagrams at the Detector Plane for a given IFU channel across a spectral band; see below:

![Spot Diagrams Detector Plane](sample_detector_spots.png?raw=true "Detector")
