"""


"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
from scipy import stats
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from time import time


def fwhm_rmswfe_detector(zosapi, sys_mode, ao_modes, spaxel_scale, grating, N_configs, N_waves, N_rays, files_path, results_path):
    """

    """

    # We will create a separate folder within results_path to save the FWHM results
    analysis_dir = os.path.join(results_path, 'RMSWFE_FWHM')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    fwhm_analysis = e2e.FWHM_PSF_FastAnalysis(zosapi=zosapi)
    rms_analysis = e2e.RMS_WFE_FastAnalysis(zosapi=zosapi)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    wavelength_idx = np.linspace(1, 23, N_waves).astype(int)
    # Select the config index. Using all of them takes forever
    # We jump every N_configs
    odd_configs = np.arange(1, 76, 2)[::N_configs]
    even_configs = odd_configs + 1
    # We use both even and odd configurations to cover both A and B paths of the IFU
    configs = list(np.concatenate([odd_configs, even_configs]))
    N = len(configs)
    print("Total Configurations: ", len(configs))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fx, fy = [], []
    focal_coord, object_coord = [], []
    rms_maps = []
    x, y = [], []
    for ifu_section in ifu_sections:
        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}
        focal_plane = e2e.focal_planes[sys_mode][spaxel_scale][ifu_section]['DET']

        list_results = fwhm_analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=configs,
                                                surface=focal_plane, N_rays=N_rays)

        # Only 1 list, no Monte Carlo
        fwhm, obj_xy, foc_xy, wavelengths = list_results[0]
        focal_coord.append(foc_xy)
        object_coord.append(obj_xy)
        fwhm_x, fwhm_y = fwhm[:, :, 0], fwhm[:, :, 1]
        fx.append(fwhm_x)
        fy.append(fwhm_y)

        list_results = rms_analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=configs,
                                                surface=None, spaxels_per_slice=3,
                                                pupil_sampling=4, save_hdf5=False)

        rms_wfe, obj_xy, foc_xy, waves = list_results[0]    # Only 1 item on the list, no Monte Carlo files
        rms_maps.append(rms_wfe)

        x.append(rms_wfe[:, :, 1])
        y.append(fwhm_x)

        colors = cm.Reds(np.linspace(0.5, 1.0, N_waves))
        for i_wave in range(N_waves):
            _rms = rms_wfe[:, i_wave, 1]
            _fwhm = fwhm_x[:, i_wave]

            if ifu_section == 'AB':
                # we add the wavelength label
                ax.scatter(_rms, _fwhm, s=10, color=colors[i_wave], label=r'%.3f $\mu$m' % waves[i_wave])
            else:
                ax.scatter(_rms, _fwhm, s=10, color=colors[i_wave])

    # Add linear regression
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_fit = np.linspace(np.min(x), np.max(x), 50)
    y_fit = intercept + slope * x_fit
    ax.plot(x_fit, y_fit, linestyle='--', color='black')
    ax.text(x=np.min(x), y=np.max(y), s=r'%.3f $\mu$m / nm' % (slope),
            bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2))

    ax.set_xlabel(r'RMS WFE [nm]')
    ax.set_ylabel(r'FWHM X PSF [$\mu$m]')
    ax.legend(title='Wavelength', loc=4)
    ax.grid(True)

    title = r'%s mas | %s SPEC | %s %s' % (spaxel_scale, grating, sys_mode, ao_modes[0])
    ax.set_title(title)
    fig_name = "RMSWFE_VS_FWHM_%s_SPEC_%s_MODE_%s_%s" % (spaxel_scale, grating, sys_mode, ao_modes[0])

    save_path = os.path.join(results_path, analysis_dir)
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))


    fx = np.array(fx)
    fy = np.array(fy)
    rms = np.array(rms_maps)

    return rms, fx, fy, object_coord, focal_coord, wavelengths


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '60x30'
    N_rays = 500
    N_waves = 5
    N_configs = 2  # Jump every N_configs, not the total
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    grating = 'H'
    files_path = os.path.abspath("D:\End to End Model\June_John2020")
    results_path = os.path.abspath("D:\End to End Model\Results_Report\Mode_NOAO\Scale_%s" % spaxel_scale)

    results = fwhm_rmswfe_detector(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                                   grating=grating, N_configs=N_configs, N_waves=N_waves, N_rays=N_rays,
                                   files_path=files_path, results_path=results_path)
    rms, fx, fy, object_coord, focal_coord, wavelengths = results
    plt.show()