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
import pandas as pd
import seaborn as sns

import psf
import utils
import calibration


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


def wavefronts(zosapi, sys_mode, ao_modes, spaxel_scale, grating, wavelength_idx, config_idx, sampling, files_path, results_path):
    """

    """

    # We will create a separate folder within results_path to save the RMS WFE results
    analysis_dir = os.path.join(results_path, 'WavefrontMaps')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # ifu_sections = ['AB', 'CD', 'EF', 'GH']
    ifu_sections = ['AB']
    analysis = e2e.WavefrontsAnalysis(zosapi=zosapi)      # The analysis object

    rms_maps, focal_coord = [], []        # Lists to save the results for each IFU channel

    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=[wavelength_idx], configuration_idx=None,
                                                surface=None, sampling=sampling,
                                                coef_path=results_path, remove_slicer_aperture=True)

        wavefront, foc_xy, waves = list_results[0]  # Only 1 item on the list, no Monte Carlo files
        rms_maps.append(wavefront)

    return rms_maps, waves

## Parameters ##

N_paths = 2
N_config = 76
N_points = N_paths * N_config

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '4x4'
    sampling = 128
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    gratings = ['H']
    # files_path = os.path.abspath("D:\End to End Model\August_2020")
    # results_path = os.path.abspath("D:\End to End Model\Wavefronts")
    files_path = os.path.abspath("D:/End to End Model/Monte_Carlo/Max")
    results_path = os.path.abspath("D:/End to End Model/Monte_Carlo/Nominal/Results/Mode_%s/Scale_%s" % (ao_modes[0], spaxel_scale))
    # [*] This is the bit we have to change when you run the analysis in your system [*]

    maps, wavelengths = wavefronts(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                                   grating=gratings[0], sampling=sampling, wavelength_idx=1, config_idx=None, files_path=files_path, results_path=results_path)

    maps_all_97 = np.concatenate(maps, axis=0)
    maps_ab = maps_all_97

    # Show some maps
    # maps_ab = maps_all[3*N_config:4*N_config, 0]
    maps_odd = maps_ab[::2]
    maps_even = maps_ab[1::2]
    N_rows = 2
    N_cols = N_config // (2 * N_rows)

    # do the odd first
    wavefront_odd = np.zeros((N_rows * sampling, N_cols * sampling))
    wavefront_even = np.zeros((N_rows * sampling, N_cols * sampling))

    for j in range(N_cols):
        wavefront_odd[:sampling, j*sampling:(j+1)*sampling] = maps_odd[j]
        wavefront_odd[sampling:N_rows*sampling, j*sampling:(j+1)*sampling] = maps_odd[j + N_cols]
        wavefront_even[:sampling, j*sampling:(j+1)*sampling] = maps_even[j]
        wavefront_even[sampling:N_rows*sampling, j*sampling:(j+1)*sampling] = maps_even[j + N_cols]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    img1 = ax1.imshow(wavefront_odd, cmap='jet')
    img1.set_clim(np.nanmin(maps_ab), np.nanmax(maps_ab))
    cbar1 = plt.colorbar(img1, ax=ax1, orientation='horizontal', label='[$\lambda$]')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title(r'Image Slicer A')

    img2 = ax2.imshow(wavefront_even, cmap='jet')
    img2.set_clim(np.nanmin(maps_ab), np.nanmax(maps_ab))
    cbar2 = plt.colorbar(img2, ax=ax2, orientation='horizontal', label='[$\lambda$]')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title(r'Image Slicer B')
    plt.show()

    # Fit the Wavefront maps to Zernike polynomials
    N_levels = 10
    zernike_fit = ZernikeFit(N_PIX=sampling, N_levels=N_levels)

    N_zern = zernike_fit.N_zern
    coefs = np.zeros((N_points, N_zern))
    for k in range(N_points):
        # wavefront = maps_all[k, 0]
        wavefront = maps_ab[k, 0]
        coef_fit = zernike_fit.fit_wavefront(wavefront)
        coefs[k] = coef_fit

    for k in range(N_zern):
        plt.figure()
        plt.imshow(zernike_fit.zernike_matrix[:, :, k], cmap='jet')
        plt.title('%d' % (k + 1))
    plt.show()


    import matplotlib.cm as cm

    idx = [3, 4, 5, 7, 8]
    labels = ['Oblique Astig.', 'Defocus', 'Astigmatism', 'Coma', 'V Coma']
    colors = cm.Reds(np.linspace(0.25, 1, len(idx)))
    markers = ['o', '^', 'v', '*', 's']

    plt.figure()
    for k, label, col, mark in zip(idx, labels, colors, markers):
        data = coefs[:, k]
        plt.scatter(np.arange(N_config), data, s=10, label=label, color=col, marker=mark)
    plt.legend(title=r'Aberration')
    # plt.ylim([-0.035, 0.035])
    plt.xlabel(r'Slice Number')
    plt.show()

    data = pd.DataFrame(coefs[:, 3:], columns=zernike_fit.aberrations[:7])

    fig_boxplot, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(data=data, ax=ax, hue_order=zernike_fit.aberrations, palette=sns.color_palette("Reds"))
    ax.set_ylabel(r'Coefficient [$\lambda$]')
    ax.set_title(r'NCPA Detector | %s scale | %s %s' % (spaxel_scale, sys_mode, ao_modes[0]))

    # fig_name = "NCPA_DET_%s_%s_%s" % (spaxel_scale, sys_mode, ao_modes[0])
    # analysis_dir = os.path.join(results_path, 'WavefrontMaps')
    # if os.path.isfile(os.path.join(analysis_dir, fig_name)):
    #     os.remove(os.path.join(analysis_dir, fig_name))
    # fig_boxplot.savefig(os.path.join(analysis_dir, fig_name))
    # plt.show()
    #
    # # Show the correlations
    # paths = ['AB', 'CD', 'EF', 'GH']
    # markers = ['^', 'v', 'd', 'o']
    # colors = ['blue', 'red', 'green', 'yellow']
    # i_aberr = 4
    # j_aberr = 5
    # plt.figure()
    # for j, path_name, in enumerate(paths):
    #     plt.scatter(coefs[j*N_config:(j+1)*N_config, i_aberr], coefs[j*N_config:(j+1)*N_config, j_aberr],
    #                 label=path_name, color=colors[j], s=10, marker=markers[j])
    #     plt.xlabel(zernike_fit.aberrations[i_aberr - 3] + r' [$\lambda$]')
    #     plt.ylabel(zernike_fit.aberrations[j_aberr - 3] + r' [$\lambda$]')
    # plt.legend(title='IFU Path')
    # plt.title(r'NCPA Detector | %s scale | %s %s' % (spaxel_scale, sys_mode, ao_modes[0]))
    # plt.show()

    true_map = maps_all[0, 0]
    map_flat = true_map[zernike_fit.pupil_mask]
    rms_true = np.std(map_flat[np.isfinite(map_flat)])
    zemax_mask = np.isfinite(true_map)
    coef_fit = zernike_fit.fit_wavefront(true_map)
    fit_map = np.dot(zernike_fit.zernike_matrix, coef_fit)

    res = true_map - fit_map
    err = res[zernike_fit.pupil_mask]
    rms_err = np.std(err[np.isfinite(err)])
    print("\nRMS wavefront: %.4f waves" % rms_true)
    print("RMS error LS fit: %.4f waves (%.1f percent)" % (rms_err, rms_err / rms_true * 100))

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

    # ================================================================================================================ #
    #                                Machine Learning calibration
    # ================================================================================================================ #

    ### PARAMETERS ###

    # PSF bits
    N_PIX = 256  # pixels for the Fourier arrays
    pix = 30  # pixels to crop the PSF images
    WAVE = 1.5  # microns | reference wavelength
    SPAX = 4.0  # mas | spaxel scale
    RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
    RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
    utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

    # Machine Learning bits
    N_train, N_test = 5000, 500  # Samples for the training of the models
    coef_strength = 0.01
    rescale = 0.35  # Rescale the coefficients to cover a wide range of RMS
    layer_filters = [16, 8]  # How many filters per layer
    kernel_size = 3
    input_shape = (pix, pix, 2,)
    epochs = 10  # Training epochs
    SNR = 500

    wave_zemax = wavelengths[0]

    # Rescale to physical units (* wave_zemax) then divide it to the reference wavelength
    # also divide by 2 Pi, since the way we compute the PSF is by multiplying the coefficients by 2 Pi
    test_coef = coefs * wave_zemax / WAVE / (2 * np.pi)
    test_coef = test_coef[:, 3:]        # remove piston and tilts

    # (1) We begin by creating a Zernike PSF model with Defocus as diversity
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    N_zern = zernike_matrix.shape[-1]
    diversity_coef = np.zeros((zernike_matrix.shape[-1]))
    diversity_coef[1] = 1 / (2*np.pi)
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=diversity_coef)

    # For the Nominal HARMONI design, the performance is almost perfect
    # so if we do not rescale the coefficients, we'll get absurdly large Strehls to begin with
    # Thus, we should probably find a scaling law to "select" how much Strehl we want,
    # while keeping the same aberration content

    def calculate_average_strehl(alpha, coef):

        N_PSF = coef.shape[0]
        strehls = np.zeros(N_PSF)
        for k in range(N_PSF):
            p, _s = PSF_zernike.compute_PSF(alpha * coef[k])
            strehls[k] = _s
        mean_strehl = np.mean(strehls)
        return mean_strehl

    N_alphas = 20
    alphas = np.logspace(0, 2, N_alphas)
    strehls = np.zeros(N_alphas)
    for k in range(N_alphas):
        print(k)
        strehls[k] = calculate_average_strehl(alpha=alphas[k], coef=test_coef)

    plt.figure()
    plt.plot(alphas, strehls)
    plt.xlabel(r'Alpha')
    plt.ylabel(r'Strehl')
    plt.show()

    # Generate training and test datasets (clean PSF images)
    train_PSF, train_coef, test_p, test_c = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                         coef_strength=coef_strength, rescale=rescale)

    # Check Defocus / Nominal ratio
    peaks_nom = np.max(train_PSF[:, :, :, 0], axis=(1, 2))
    peaks_foc = np.max(train_PSF[:, :, :, 1], axis=(1, 2))
    plt.figure()
    plt.plot(peaks_nom)
    plt.plot(peaks_foc)
    plt.show()

    # Generate the Test Set using the coefficients
    N_PSF = test_coef.shape[0]
    test_PSF = np.zeros((N_PSF, pix, pix, 2))
    for k in range(N_PSF):
        pnom, _s = PSF_zernike.compute_PSF(test_coef[k])
        pfoc, _s = PSF_zernike.compute_PSF(test_coef[k], diversity=True)
        test_PSF[k, :, :, 0] = pnom
        test_PSF[k, :, :, 1] = pfoc

    # Initialize Convolutional Neural Network model for calibration
    calibration_model = calibration.create_cnn_model(layer_filters, kernel_size, input_shape,
                                                     N_classes=N_zern, name='CALIBR', activation='relu')

    # Train the calibration model
    train_history = calibration_model.fit(x=train_PSF, y=train_coef,
                                          validation_data=(test_PSF, test_coef),
                                          epochs=epochs, batch_size=32, shuffle=True, verbose=1)

    from numpy.linalg import norm
    # Evaluate performance
    guess_coef = calibration_model.predict(test_PSF)
    residual_coef = test_coef - guess_coef
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after = np.mean(norm(residual_coef, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.5f" % norm_before)
    print("After : %.5f" % norm_after)

    plt.figure()
    plt.scatter(test_coef[:, 1], guess_coef[:, 1], s=5)
    plt.show()

    # Problem: these coefficients are in the basis of Python Zernikes
    # if we want to apply a correction in Zemax, we have to transform them back



