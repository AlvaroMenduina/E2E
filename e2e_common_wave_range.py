"""

Common Wavelength Range

Author: Alvaro Menduina
Date: July 2020

Description:

After John Capone updated the Spectrograph files we noticed that some rays started
falling outside the active area of the Detector, suggesting the wavelengths defined
in the E2E Model Zemax files are "optimistic".
We decided to recalculate the Common Wavelength Range, i.e. the shortest and longest
wavelengths for which ALL configuration (slices) fall within the active area of the
detector.

Here's how we do it:

(1) We trace a chief ray for all Field Points of a slice, for all configurations
    and all wavelengths, up to the Detector plane
(2) For each wavelength we calculate the maximum and minimum Y (spectral direction)
    and linearly interpolate Wavelength -> [Min or Max] Spectral coordinate
(3) We solve for the wavelength that has the Spectral coordinate equal to the
    max or min detector aperture

"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
from scipy import interpolate


def interpolate_wave(foc_y_coord, wavelengths, limit):
    """
    Perform linear interpolation between X: Focal Coodinates Spectral Direction
    and Y: the Wavelengths

    And use that interpolation to calculate at which Wavelength the focal coordinates
    exceed the detector aperture (limit):
    Max or Min wavelength = f_interpol (X = detector aperture)

    :param foc_y_coord: array containing the (max or min) detector spectral coordinates for each wavelength
    :param wavelengths: array containing the wavelengths
    :param limit: value of the detector aperture
    :return:
    """
    x = foc_y_coord
    y = wavelengths
    f_int = interpolate.interp1d(x, y)

    return f_int(limit)


def common_wavelength_range(zosapi, sys_mode, ao_modes, spaxel_scale, grating, files_path, results_path):
    """
    Calculate the Common Wavelength Range

    For a given spaxel scale and spectral band, we have 4 IFU channels available. There can be some
    variability on the detector raytrace coordinates so we should calculate the min and max wavelength
    that works for all IFU channels

    Method:
        (1) For each IFU, we get the raytrace results at the detector plane.
        (2) We calculate, for each wavelength, the minimum and maximum Y (spectral) coordinate;
            i.e. how far up / down the detector each set of slitlets lands.
        (3) Shorter wavelengths have positive (+) spectral coordinates, longer wavelengths
            have negative (-) spectral coordinates
        (4) So to calculate the SHORTEST possible wavelength for an IFU channel we linearly interpolate
            the wavelengths and the maximum spectral coordinate for each wavelength.
                ** If for some wavelength the rays fall outside the detector, we use
                    the interpolation to calculate the wavelength at which we reach the detector boundary
                ** If none of the rays for the shortest Zemax defined wavelength falls outside
                    then we say that is the SHORTEST wavelength (no recalculation)
        (5) For the LONGEST possible wavelength we do the equivalent process, but with the
            minimum spectral coordinates.
        (6) We combine the results for each IFU to calculate the SHORTEST wavelength as
            "the longest of the SHORTEST wavelengths of all 4 IFUs", and the equivalent for
            the LONGEST wavelength, thus defining the Common Wavelength Range

    :param zosapi: the Zemax API
    :param sys_mode: [str] the system mode, either 'HARMONI', 'ELT' or 'IFS
    :param ao_modes: [list] containing the AO mode we want. Only 1 should be used
    :param spaxel_scale: [str] the spaxel scale, ex. '60x30'
    :param grating: [str] the spectral band to analyze, ex. 'HK'
    :param files_path: path to the E2E files to load for the analysis
    :param results_path: path to save the results
    :return:
    """

    analysis_dir = os.path.join(results_path, 'CWR')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    analysis = e2e.CommonWavelengthRange(zosapi=zosapi)

    focal_coord = []
    short_waves, long_waves = [], []

    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=None, surface=None)

        # Get the Detector XY coordinates and the Wavelengths
        foc_xy, waves = list_results[0]
        focal_coord.append(foc_xy)

        _foc = foc_xy[:, :, :, 1]
        # Calculate the Maximum and Minimum spectral coordinates Y for each wavelength
        max_foc = np.max(_foc, axis=(0, 2))
        min_foc = np.min(_foc, axis=(0, 2))
        try:
            # The short wavelengths have (+) positive Y, so we use the Maximum Y coordinates
            # to interpolate and calculate the SHORTEST wavelength
            short_wave = interpolate_wave(foc_y_coord=max_foc, wavelengths=waves, limit=30.66)
        except ValueError:
            # We cannot interpolate outside the defined range of wavelengths
            # So if there is no problem of rays for the shortest wavelength falling outside
            # the detector area, the interpolation won't work. The shortest Zemax wavelength
            # is perfectly valid, so no need to interpolate
            short_wave = np.min(waves)

        try:
            # Long wavelengths have (-) negative Y
            long_wave = interpolate_wave(foc_y_coord=min_foc, wavelengths=waves, limit=-30.66)
        except ValueError:
            long_wave = np.max(waves)

        print("\nIFU-%s | Maximum Y detector: %.4f microns -> %.3f mm & %.4f microns -> %.3f mm" %
              (ifu_section, waves[0], max_foc[0], waves[1], max_foc[1]))
        print("Interpolated Shortest Wavelength: %.4f microns" % short_wave)
        print("IFU-%s | Minimum Y detector: %.4f microns -> %.3f mm & %.4f microns -> %.3f mm" %
              (ifu_section, waves[-2], min_foc[-2], waves[-1], min_foc[-1]))
        print("Interpolated Longest Wavelength: %.4f microns" % long_wave)

        short_waves.append(short_wave)
        long_waves.append(long_wave)

    # There should be very little variation between IFU channels, but just for the sake of
    # consistency, we calculate the SHORTEST wavelength as the longest of the shortest wavelengths
    min_wave = np.max(short_waves)
    max_wave = np.min(long_waves)
    print("\nCommon Wavelength Range: [%.4f, %.4f] microns" % (min_wave, max_wave))
    cwr = [min_wave, max_wave]

    return focal_coord, waves, cwr


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\June_John2020")
    results_path = os.path.abspath("D:\End to End Model\Results_Report\Mode_LTAO\Scale_20x20")
    analysis_dir = os.path.join(results_path, 'CWR')

    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    sys_mode = 'HARMONI'
    ao_modes = ['LTAO']
    spaxel_scale = '20x20'
    gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    # gratings = ['VIS']

    # We will save the results in a .txt file 
    file_name = 'Common_Wavelength_Ranges_%s_%s_%s.txt' % (sys_mode, ao_modes[0], spaxel_scale)
    with open(os.path.join(analysis_dir, file_name), 'w') as f:
        f.write('MODE: %s, AO: %s, Spaxel Scale: %s' % (sys_mode, ao_modes[0], spaxel_scale))
        f.write('\nCommon Wavelength Ranges')
        for grating in gratings:
            focal_coord, waves, cwr = common_wavelength_range(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                                                    grating=grating,files_path=files_path, results_path=results_path)
            f.write('\nSpectral band: %s' % grating)
            f.write('\nCommon Wavelength Range: [%.5f, %.5f] microns' % (cwr[0], cwr[1]))

    f.close()

