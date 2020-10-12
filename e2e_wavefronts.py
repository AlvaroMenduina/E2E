"""

Wavefronts

Author: Alvaro Menduina
Date: Oct 2020

Description:




"""

import os
import numpy as np
import matplotlib.pyplot as plt
import e2e_analysis as e2e
import zernike as zern


class ZernikeFit(object):
    """


    (1 - 3) Piston, Tilts
    (4) Oblique Astigmatism
    (5) Defocus
    (6) Horizontal Astigmatism
    (7) Oblique Trefoil
    (8) Horizontal Coma
    (9) Vertical Coma
    (10) Horizontal Trefoil
    (11) Oblique Quadrefoil


    """

    def __init__(self, N_PIX, N_levels):
        self.N_PIX = N_PIX
        self.create_zernike_matrix(N_levels=N_levels)
        self.create_least_squares_matrix()

        self.aberrations = ['Obliq. Astig.', 'Defocus', 'Horiz. Astig.',
                            'Obliq. Trefoil', 'Horiz. Coma', 'Vert. Coma', 'Horiz. Trefoil',
                            'Obliq. Quadref.', '2nd Obliq. Coma', 'Spherical', '2nd Horiz. Coma', 'Horiz. Quadref.']

        return

    def create_zernike_matrix(self, N_levels):
        # Calculate how many Zernikes we need, to reach N_levels
        levels = zern.triangular_numbers(N_levels)
        N_zern = levels[N_levels]

        x = np.linspace(-1, 1, self.N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        pupil = (rho <= 1.0)
        self.pupil_mask = pupil

        rho, theta = rho[pupil], theta[pupil]
        zernike = zern.ZernikeNaive(mask=pupil)
        _phase = zernike(coef=np.zeros(N_zern), rho=rho, theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        self.H_flat = zernike.model_matrix
        self.zernike_matrix = zern.invert_model_matrix(self.H_flat, pupil)
        self.N_zern = self.H_flat.shape[-1]  # Update the total number of Zernikes

        return

    def create_least_squares_matrix(self):
        N_matrix = np.dot(self.H_flat.T, self.H_flat)
        inverse_N = np.linalg.inv(N_matrix)
        self.LS_matrix = np.dot(inverse_N, self.H_flat.T)

        return

    def fit_wavefront(self, wavefront_map):
        # apply pupil mask to remove values outside the pupil
        wavefront_flat = wavefront_map[self.pupil_mask]

        # remove potential NaN values
        nan_mask = np.isnan(wavefront_flat)
        wavefront_flat[nan_mask] = 0.0

        # Least Squares fit
        x_fit = np.dot(self.LS_matrix, wavefront_flat)

        return x_fit


def wavefronts(zosapi, sys_mode, ao_modes, spaxel_scale, grating, wavelength_idx, config_idx, sampling, files_path, results_path):
    """

    """

    # We will create a separate folder within results_path to save the RMS WFE results
    analysis_dir = os.path.join(results_path, 'WavefrontMaps')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    # ifu_sections = ['AB']
    analysis = e2e.WavefrontsAnalysis(zosapi=zosapi)      # The analysis object

    rms_maps, focal_coord = [], []        # Lists to save the results for each IFU channel

    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=[wavelength_idx], configuration_idx=config_idx,
                                                surface=None, sampling=sampling, remove_slicer_aperture=True)

        wavefront, foc_xy, waves = list_results[0]  # Only 1 item on the list, no Monte Carlo files
        rms_maps.append(wavefront)

    return rms_maps, waves


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '4x4'
    sampling = 256
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    gratings = ['H']
    files_path = os.path.abspath("D:\End to End Model\August_2020")
    results_path = os.path.abspath("D:\End to End Model\Wavefronts")
    # [*] This is the bit we have to change when you run the analysis in your system [*]

    maps, wavelengths = wavefronts(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                      grating=gratings[0], sampling=sampling, wavelength_idx=1, config_idx=None, files_path=files_path, results_path=results_path)

    map_s = np.concatenate(maps, axis=0)

    # Fit the Wavefront maps to Zernike polynomials
    N_levels = 5
    zernike_fit = ZernikeFit(N_PIX=sampling, N_levels=N_levels)

    N_paths = 4
    N_config = 76
    N_points = N_paths * N_config
    N_zern = zernike_fit.N_zern
    coefs = np.zeros((N_points, N_zern))
    for k in range(N_points):
        wavefront = map_s[k, 0]
        coef_fit = zernike_fit.fit_wavefront(wavefront)
        coefs[k] = coef_fit

    for k in range(N_zern):
        plt.figure()
        plt.imshow(zernike_fit.zernike_matrix[:, :, k], cmap='jet')
        plt.title('%d' % (k + 1))
    plt.show()


    import pandas as pd
    import seaborn as sns

    data = pd.DataFrame(coefs[:, 3:], columns=zernike_fit.aberrations)

    fig_boxplot, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(data=data, ax=ax, hue_order=zernike_fit.aberrations, palette=sns.color_palette("Reds"))
    ax.set_ylabel(r'Coefficient [$\lambda$]')
    ax.set_title(r'NCPA Detector | %s scale | %s %s' % (spaxel_scale, sys_mode, ao_modes[0]))
    plt.show()

    # Show the correlations
    paths = ['AB', 'CD', 'EF', 'GH']
    markers = ['^', 'v', 'd', 'o']
    colors = ['lightblue', 'red', 'green', 'black']
    i = 6
    plt.figure()
    for j, path_name, in enumerate(paths):
        plt.scatter(coefs[j*N_config:(j+1)*N_config, i], coefs[j*N_config:(j+1)*N_config, i+1],
                    label=path_name, color=colors[j], s=10, marker=markers[j])
    plt.legend()
    plt.show()


    true_map = map_s[2, 0]
    map_flat = true_map[zernike_fit.pupil_mask]
    rms_true = np.std(map_flat[np.isfinite(map_flat)])
    zemax_mask = np.isfinite(true_map)
    coef_fit = zernike_fit.fit_wavefront(true_map)
    fit_map = np.dot(zernike_fit.zernike_matrix, coef_fit)

    res = true_map - fit_map
    err = res[zernike_fit.pupil_mask]
    rms_err = np.std(err[np.isfinite(err)])
    print(rms_err)

    max_zem = max(-np.nanmin(true_map), np.nanmax(true_map))
    max_fit = max(-np.nanmin(fit_map), np.nanmax(fit_map))
    max_val = max(max_zem, max_fit)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    img1 = ax1.imshow(true_map, cmap='jet', origin='lower')
    plt.colorbar(img1, ax=ax1, orientation='horizontal')
    img1.set_clim(-max_val, max_val)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)


    img2 = ax2.imshow(fit_map, cmap='jet', origin='lower')
    plt.colorbar(img2, ax=ax2, orientation='horizontal')
    img2.set_clim(-max_val, max_val)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    img3 = ax3.imshow(res, cmap='RdBu', origin='lower')
    img3.set_clim(-max_val, max_val)
    plt.colorbar(img3, ax=ax3, orientation='horizontal')
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)

    plt.show()

    import corner
    figure = corner.corner(coefs[:, 3:7])

    plt.figure()
    plt.scatter(coefs[:76, 3], coefs[:76, 5], s=5, color='blue')
    plt.scatter(coefs[76:2*76, 3], coefs[76:2*76, 5], s=5, color='red')
    plt.scatter(coefs[2*76:3*76, 3], coefs[2*76:3*76, 5], s=5, color='green')
    plt.show()

