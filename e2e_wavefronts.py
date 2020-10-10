"""

Wavefronts

Author: Alvaro Menduina
Date: Oct 2020

Description:




"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt


def wavefronts(zosapi, sys_mode, ao_modes, spaxel_scale, grating, sampling, files_path, results_path):
    """

    """

    # We will create a separate folder within results_path to save the RMS WFE results
    analysis_dir = os.path.join(results_path, 'WavefrontMaps')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    analysis = e2e.WavefrontsAnalysis(zosapi=zosapi)      # The analysis object

    rms_maps, object_coord, focal_coord = [], [], []        # Lists to save the results for each IFU channel
    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=[1], configuration_idx=[1],
                                                surface=None, sampling=sampling, remove_slicer_aperture=True)

        wavefront, foc_xy, waves = list_results[0]  # Only 1 item on the list, no Monte Carlo files
        rms_maps.append(wavefront)

    return rms_maps

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '60x30'
    spaxels_per_slice = 3       # How many field points per Slice to use
    pupil_sampling = 4          # N x N grid per pupil quadrant. See Zemax Operand help for RWRE
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    gratings = ['H']
    files_path = os.path.abspath("D:\End to End Model\August_2020")
    results_path = os.path.abspath("D:\End to End Model\Wavefronts")
    # [*] This is the bit we have to change when you run the analysis in your system [*]

    maps = wavefronts(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                      grating=gratings[0], sampling=1024, files_path=files_path, results_path=results_path)