"""

E2E - Monte Carlo Wavefront Analysis

Author: Alvaro Menduina
Date: December 2020

Description:
The idea is to explore how the Wavefront Maps vary across a population of E2E Monte Carlo files
This will give us two very important things:
    - A way to quantify the performance of HARMONI statiscally
    - A way to generate realistic Wavefront Maps that can be used as proxies of NCPA
        to train and test Machine Learning Calibration Models.

In this script, we loop over the available E2E MC files
For each one, we run a Wavefront Analysis (one for each IFU path),
looking at the Wavefront at the centre of the slice, for all configurations,
and 3 wavelengths [1, 12, 23].

We then fit those sample wavefront to Zernike Polynomials

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import e2e_analysis as e2e
import zernike as zern
import pandas as pd
import seaborn as sns


def monte_carlo_str(k):
    """
    Monte Carlo instance number follow the string format of '0001'
    this function generates the MC instance string for a given integer k
    :param k:
    :return:
    """
    n_zeros = 4 - len(str(k))
    mc = n_zeros * '0' + str(k)
    return mc


def concatenate_images(data, N_row, N_col):
    """
    Concatenate the Wavefront Maps for a given set of configurations
    into a single array [N_row x N_col] to be shown with imshow
    :param data:
    :param N_row:
    :param N_col:
    :return:
    """
    imagex = []
    for i in range(N_row):
        imagey = []
        for j in range(N_col):
            k = N_col * i + j
            imagey.append(data[k])
        imagey = np.concatenate(imagey, axis=1)
        imagex.append(imagey)
    imagex = np.concatenate(imagex, axis=0)
    return imagex


class ZernikeFit(object):
    """
    Object in charge of receiving some wavefront maps and fitting them to Zernike polynomials

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
        """

        :param N_PIX: Number of pixels in the wavefront maps
        :param N_levels: How many Zernike radial levels to consider
        """
        self.N_PIX = N_PIX
        self.create_zernike_matrix(N_levels=N_levels)
        self.create_least_squares_matrix()

        self.aberrations = ['Obliq. Astig.', 'Defocus', 'Horiz. Astig.',
                            'Obliq. Trefoil', 'Horiz. Coma', 'Vert. Coma', 'Horiz. Trefoil',
                            'Obliq. Quadref.', '2nd Obliq. Coma', 'Spherical', '2nd Horiz. Coma', 'Horiz. Quadref.']

        return

    def create_zernike_matrix(self, N_levels):
        """
        Initialize a Zernike matrix containing the Zernike polynomials
        up to radial order N_levels
        :param N_levels: How many Zernike radial levels to consider
        :return:
        """

        # Calculate how many Zernikes we need, to reach N_levels
        levels = zern.triangular_numbers(N_levels)
        N_zern = levels[N_levels]

        # We use a circular un-obscured pupil
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
        """
        In order to fit the wavefront maps to a set of Zernike polynomials
        we follow a Least-Squares approach in the form:
            y_obs = H * x_fit + eps
        where y_obs is the wavefront map, x_fit is the Zernike coefficients, and H is the Zernike matrix

        Following the LS approach:
            H.T * y_obs = (H.T * H) * x_fit + eps
            N = (H.T * H)  the "Normal" Matrix
            x_fit ~= (inv(N) * H.T) * y_obs
        In order to efficiently do the fitting for many instances of y_obs,
        we calculate the LS Matrix (inv(N) * H.T) only once, as it depends on the Zernike polynomials
        and we resuse afterwards
        :return:
        """

        N_matrix = np.dot(self.H_flat.T, self.H_flat)
        inverse_N = np.linalg.inv(N_matrix)
        self.LS_matrix = np.dot(inverse_N, self.H_flat.T)

        return

    def fit_wavefront(self, wavefront_map):
        """
        Given a wavefront map from Zemax in the form N_PIX x N_PIX
        we fit it to Zernike polynomials using a Least Squares approach
        :param wavefront_map: [N_PIX x N_PIX] wavefront map from Zemax
        :return:
        """

        # apply pupil mask to remove values outside the pupil
        wavefront_flat = wavefront_map[self.pupil_mask]

        # remove potential NaN values (sometimes the edges of the Zemax pupil do not match the circular pupil)
        nan_mask = np.isnan(wavefront_flat)
        wavefront_flat[nan_mask] = 0.0

        # Least Squares fit
        x_fit = np.dot(self.LS_matrix, wavefront_flat)

        return x_fit


def calculate_wavefronts(zosapi, file_options, sampling, files_path, results_path):
    """
    Funtion that takes care of calculating the Wavefront Maps for all IFU paths
    of a given E2E MC model

    :param zosapi:
    :param file_options:
    :param sampling:
    :param files_path:
    :param results_path:
    :return:
    """

    spaxel_scale = file_options['SPAX_SCALE']
    grating = file_options['GRATING']
    ao_mode = file_options['AO_MODE']
    mc_str = file_options['FPRS_MC']
    # Here we assume that all subsystems have the same MC instance. Only exception is the ISP, which varies for IFUs

    # We will create a separate folder within results_path to save the results
    analysis_dir = os.path.join(results_path, 'WavefrontMaps')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    # ifu_sections = ['AB']
    analysis = e2e.WavefrontsAnalysisMC(zosapi=zosapi)      # The analysis object

    wavefront_maps, rms_wfes = [], []                       # We save the Maps as well as their RMS WFE
    for ifu_section in ifu_sections:

        # change the IFU path option to the current IFU section
        file_options['IFU_PATH'] = ifu_section
        # read the MC instance number of the specific IFU path
        ifu_mc = file_options['IFU_PATHS_MC'][ifu_section]
        file_options['IFU_MC'] = ifu_mc

        # select the proper ISP MC instance, according to the IFU path
        isp_mc = file_options['IFU_ISP_MC'][ifu_section]
        file_options['ISP_MC'] = isp_mc

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=file_options, results_path=results_path,
                                                wavelength_idx=[1, 12, 23], configuration_idx=None,
                                                surface=None, sampling=sampling, remove_slicer_aperture=True)

        wavefront, rms_wfe, obj_xy, waves = list_results[0]  # Only 1 item on the list, no Monte Carlo files
        wavefront_maps.append(wavefront)
        rms_wfes.append(rms_wfe)

    # Wavefront Maps of shape [N_IFU, N_config, N_waves, N_pix, N_pix]
    wavefront_maps = np.array(wavefront_maps)
    rms_wfes = np.array(rms_wfes)
    # Note! Both the wavefront maps and RMS WFE at this point are in physical units [nm]

    # Save the results
    filesufix = '_%s_SPAX-%s_SPEC-%s_MC%s' % (ao_mode, spaxel_scale, grating, mc_str)
    np.save(os.path.join(analysis_dir, 'WAVEMAPS' + filesufix), wavefront_maps)
    np.save(os.path.join(analysis_dir, 'RMSWFE' + filesufix), rms_wfes)
    np.save(os.path.join(analysis_dir, 'WAVELENGTHS' + filesufix), np.array(waves))

    # save an example figure of the wavefront
    i_wave, i_ifu, i_config = 0, 0, 0                       # we show the Min. Wave, IFU-AB, 1st Config
    wavelength = waves[i_wave]
    data = wavefront_maps[i_ifu, i_config, i_wave]
    rms0 = rms_wfes[i_ifu, i_config, i_wave]                # we also read the RMS WFE for that particular map

    data -= np.nanmean(data)
    clim = max(-np.nanmin(data), np.nanmax(data))
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(data, cmap='Reds')
    # img.set_clim(-clim, clim)
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.set_xlabel('[nm]')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title('%s | %s | IFU-%s | %s band (%.3f $\mu$m) | RMS %.1f nm' %
                 (ao_mode, spaxel_scale, ifu_sections[0], grating, wavelength, rms0))
    fig_name = "WAVEFRONT" + filesufix
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig.savefig(os.path.join(analysis_dir, fig_name))

    plt.close('all')

    return wavefront_maps, rms_wfes, waves


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] Monte Carlo Instances [*]
    ao_mode = 'NOAO'
    spaxel_scale = '4x4'
    sampling = 64
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    gratings = ['H']

    files_path = os.path.abspath("D:/End to End Model/Monte_Carlo_Dec/ManyMC")
    results_path = os.path.abspath("D:/End to End Model/Monte_Carlo_Dec/ManyMC/Results/Mode_%s/Scale_%s" % (ao_mode, spaxel_scale))
    # [*] Monte Carlo Instances [*]

    # (0) Define the directory to save the results
    analysis_dir = os.path.join(results_path, 'WavefrontMaps')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # (1) Zernike Polynomial fit parameters
    N_levels = 12           # How many levels of the Zernike pyramid, aka radial orders
    zernike_fit = ZernikeFit(N_PIX=sampling, N_levels=N_levels)
    pupil_mask = zernike_fit.pupil_mask
    N_coef = zernike_fit.zernike_matrix.shape[-1]       # How many coefficients will the fit consider

    # (2) Wavefront Analysis MC
    N_files = 30
    for k_mc in np.arange(1, N_files + 1):

        # Define the dictionary containing all the MC instances for the different subsystems
        mc = monte_carlo_str(k_mc)
        filesufix = '_%s_SPAX-%s_SPEC-%s_MC%s' % (ao_mode, spaxel_scale, gratings[0], mc)
        file_options = {'MONTE_CARLO': True, 'AO_MODE': ao_mode, 'SPAX_SCALE': spaxel_scale, 'GRATING': gratings[0],
                        'FPRS_MC': mc, 'IPO_MC': mc,
                        'IFU_PATHS_MC': {'AB': mc, 'CD': mc, 'EF': mc, 'GH': mc},
                        'IFU_ISP_MC': {'AB': mc, 'CD': monte_carlo_str(k_mc + 1),
                                       'EF': monte_carlo_str(k_mc + 2), 'GH': monte_carlo_str(k_mc + 3)}}
        # Since the MC instances are not ordered by performance (they are randomly distributed), we can use the same
        # number for all IFU paths (AB, CD...). But to make it more realistic, each IFU path should lead to a
        # different ISP, so we have to shift the ISP_MC instance for each IFU

        # Run the Analysis for a given set of E2E-IFU files
        wavefronts, rms_wfe, waves = calculate_wavefronts(zosapi=psa, file_options=file_options, files_path=files_path,
                                                          results_path=results_path, sampling=sampling)
        # wavefronts here are of shape [N_IFU, N_config, N_waves, N_pix, N_pix]

        # Show a compound image of 8 x 9 = 72 wavefront maps, one for each configuration
        # This assumes we have only run IFU-AB
        N_row, N_col = 8, 9
        data = concatenate_images(wavefronts[0, :, 0], N_row, N_col)

        # data -= np.nanmean(data)
        # clim = max(-np.nanmin(data), np.nanmax(data))
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        img = ax.imshow(data, cmap='Reds', origin='lower')
        # for i in range(N_col):
        #     for j in range(N_row):
        #         m = i * N_row + j
        #         ax.text(sampling//10 + i * sampling, sampling//10 + j * sampling, m + 1,
        #                 {'color': 'black', 'ha': 'center', 'va': 'center',
        #                  'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        # img.set_clim(-clim, clim)
        cbar = plt.colorbar(img, ax=ax)
        cbar.ax.set_xlabel('[nm]')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('%s | %s | IFU-%s | %s band (%.3f $\mu$m) | MC %s | Configurations' %
                     (ao_mode, spaxel_scale, 'AB', file_options['GRATING'], waves[0], mc))
        fig_name = "WAVEFRONT_COMBINED_MC" + filesufix

        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        fig.savefig(os.path.join(analysis_dir, fig_name))

        # Fit the wavefronts to Zernike polynomials
        wave_data = wavefronts.reshape((-1, sampling, sampling))
        _rms = rms_wfe.flatten()
        # First we have to filter out the vignetted cases, which have been flagged with RMS WFE = np.nan
        finite_mask = np.isfinite(_rms)
        wave_data_finite = wave_data[finite_mask]
        N_samples = wave_data_finite.shape[0]           # this number will vary for each file
        zern_coef = np.zeros((N_samples, N_coef))
        for k in range(N_samples):
            zern_coef[k] = zernike_fit.fit_wavefront(wave_data[k])
        np.save(os.path.join(analysis_dir, 'ZERN_COEF_MC' + filesufix), zern_coef)

    # ================================================================================================================ #
    #                                 MC Performance Analysis
    # ================================================================================================================ #
    # We can take advantage of the MC simulations to explore the behaviour of the system statistically

    # Limitations: we only look at field point at the centre of the slice.

    # Nominal requirements for RMS WFE: Static IQ
    RMS_WFE_REQUIREMENTS = {'4x4': 81, '10x10': 123, '20x20': 254, '60x30': 590}

    rms_mc, coef = [], []
    N_files = 20
    for k_mc in np.arange(1, N_files + 1):
        mc = monte_carlo_str(k_mc)
        filesufix = '_%s_SPAX-%s_SPEC-%s_MC%s' % (ao_mode, spaxel_scale, gratings[0], mc)
        _x = np.load(os.path.join(analysis_dir, 'RMSWFE' + filesufix + '.npy'))
        _z = np.load(os.path.join(analysis_dir, 'ZERN_COEF_MC' + filesufix + '.npy'))
        rms_mc.append(_x)
        coef.append(_z)
    # RMS_MC will be of shape [N_files, N_IFU, N_config, N_waves]
    rms_mc = np.array(rms_mc)
    zern_coef = np.array(coef)

    # [1] Let's look at the whole population, irrespective of the file / IFU path separation
    # At the moment, to avoid issues with the clipping, we mask out NaN values
    rms_population = rms_mc[np.isfinite(rms_mc)]

    fig, ax = plt.subplots(1, 1)
    MAX = RMS_WFE_REQUIREMENTS[spaxel_scale]
    N_bins = 50
    bins = np.linspace(0, 1.5 * MAX, N_bins)
    ax.hist(rms_population, bins=bins)
    ax.axvline(x=MAX, linestyle='--', color='red', label='Static IQ')
    ax.set_ylabel(r'Frequency [ ]')
    ax.set_xlabel(r'RMS WFE [nm]')
    ax.legend()
    title = r'%s | %s | %s band | MC runs: %d' % (ao_mode, spaxel_scale, gratings[0], N_files)
    ax.set_title(title)
    plt.show()

    # [2] Cumulative distribution
    N_points = 100
    range_rms = np.linspace(0, 1.15 * MAX, N_points)
    cumul = np.zeros(N_points)
    for j in range(N_points):
        mask = rms_population < range_rms[j]
        cumul[j] = 100 * np.sum(mask) / mask.shape[0]

    fig, ax = plt.subplots(1, 1)
    ax.plot(range_rms, cumul)
    ax.set_xlabel(r'RMS Wavefront [nm]')
    ax.set_ylabel(r'Cumulative Fraction [percent]')
    ax.grid(True)
    ax.set_xlim([range_rms[0], range_rms[-1]])
    ax.axvline(x=MAX, linestyle='--', color='red', label='Static IQ')
    ax.set_ylim([0, 100])
    ax.set_yticks(np.arange(0, 110, 10))
    ax.legend(loc=2)
    title = r'%s | %s | %s band | MC runs: %d' % (ao_mode, spaxel_scale, gratings[0], N_files)
    ax.set_title(title)
    plt.show()

    # [3] Corner plots of the Zernike coefficients?
    rms_max = np.nanmax(rms_mc, axis=(1, 2, 3))

    # rms_grating = np.array(rms_grating).T
    import matplotlib.cm as cm
    plt.figure()
    reds = cm.Reds(np.linspace(0.25, 1, N_files))
    for j in range(N_files):
        plt.scatter(zern_coef[j, :, 5], zern_coef[j, :, 7], s=4, color=reds[j])
    # plt.scatter(zern_coef[1, :, 5], zern_coef[1, :, 6], s=5)
    plt.xlim([-150, 150])
    plt.ylim([-150, 150])
    plt.show()

    import corner

    samples = zern_coef[:, :, 3:8]
    samples = samples.reshape((-1, samples.shape[-1]))
    mask_err = np.linalg.norm(samples, axis=1) < 400
    samples = samples[mask_err]
    figure = corner.corner(samples)
    plt.show()



    # Fit the Wavefront maps to Zernike polynomials
    # levels = np.arange(5, 16)
    # errors, N_coefs = [], []
    # for N_levels in levels:
    #     # N_levels = 10
    #     zernike_fit = ZernikeFit(N_PIX=sampling, N_levels=N_levels)
    #     pupil_mask = zernike_fit.pupil_mask
    #     wavef_ = wavefronts[0, 0, 0]
    #     rms0 = np.std(wavef_[np.isfinite(wavef_)])
    #     coef_fit = zernike_fit.fit_wavefront(wavef_)
    #     N_coefs.append(coef_fit.shape[0])
    #     fit = np.dot(zernike_fit.zernike_matrix, coef_fit)
    #     diff = pupil_mask * (fit - wavef_)
    #     mask = np.isfinite(diff)
    #     error = np.std(diff[mask]) / rms0 * 100
    #     errors.append(error)
    #
    # plt.figure()
    # plt.plot(levels, errors)
    #
    # plt.figure()
    # plt.plot(levels, N_coefs)
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(wavef_)
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(fit)
    # plt.colorbar()
    #
    #
    # clim = max(-np.nanmin(diff[pupil_mask]), np.nanmax(diff[pupil_mask]))
    # plt.figure()
    # plt.imshow(diff * pupil_mask, cmap='RdBu')
    # plt.clim(-clim, clim)
    # plt.colorbar()
    # plt.show()


