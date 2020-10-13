"""




"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt

def psf_dynamic(zosapi, sys_mode, ao_modes, spaxel_scale, grating, wavelength_idx, N_rays, files_path, results_path):
    """

    """

    # We will create a separate folder within results_path to save the FWHM results
    analysis_dir = os.path.join(results_path, 'PSF_DYNAMIC')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.PSFDynamic(zosapi=zosapi)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    configs = [1]

    for ifu_section in ifu_sections:
        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=configs, N_rays=N_rays)


    return


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '60x30'
    # gratings = ['VIS', 'IZ', 'J', 'IZJ', 'Z_HIGH', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    gratings = ['H']
    files_path = os.path.abspath("D:\End to End Model\August_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_ReportAugust\Mode_%s\Scale_%s" % (ao_modes[0], spaxel_scale))
    # [*] This is the bit we have to change when you run the analysis in your system [*]


    psf_dynamic(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale, grating='H',
                wavelength_idx=[1], N_rays=500, files_path=files_path, results_path=results_path)