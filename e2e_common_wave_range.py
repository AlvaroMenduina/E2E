"""

Common Wavelength Range


Author: Alvaro Menduina
Date: July 2020

Description:

XXX

"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import interpolate
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from time import time


def interpolate_wave(foc_y_coord, wavelengths, limit):
    x = foc_y_coord
    y = wavelengths
    f = interpolate.interp1d(x, y)
    return f(limit)

def common_wavelength_range(zosapi, sys_mode, ao_modes, spaxel_scale, grating, files_path, results_path):

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
                                                wavelength_idx=None, configuration_idx=None,
                                                surface=None)

        foc_xy, waves = list_results[0]
        focal_coord.append(foc_xy)

        _foc = foc_xy[:, :, :, 1]
        max_foc = np.max(_foc, axis=(0, 2))
        min_foc = np.min(_foc, axis=(0, 2))
        try:
            short_wave = interpolate_wave(foc_y_coord=max_foc, wavelengths=waves, limit=30.66)
        except ValueError:
            short_wave = np.min(waves)

        try:
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
    results_path = os.path.abspath("D:\End to End Model\Results_Report\Mode_NOAO\Scale_20x20")
    analysis_dir = os.path.join(results_path, 'CWR')

    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '20x20'
    gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    # gratings = ['VIS']

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

