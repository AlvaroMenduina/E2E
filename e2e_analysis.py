"""

Performance Analysis for the End-to-End Models

This is the main module that contains all the stuff needed to
construct a performance analysis

Author: Alvaro Menduina
Date Created: March 2020
Latest version: August 2020

"""

import os
import numpy as np
from numpy.fft import ifft2, fft2, fftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
import seaborn as sns
from time import time
import h5py
import datetime
from sklearn.neighbors import KernelDensity
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import functools

from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import constants
from win32com.client import CastTo

# Get the module version from Git
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# We need to know the Zemax surface number for all Focal Planes
# for each mode (ELT, HARMONI, IFS) and for each spaxel scale (4x4, 10x10, 20x20, 60x30)
# for the different focal planes

# Keywords for Focal Plane are:
# PO: PreOptics, IS: Image Slicer, SL: Slit, DET: Detector

# These are the old April-May values
# TODO: [***] This should probably go at some point, we no longer analyze the IFS mode
focal_planes = {}
focal_planes['IFS'] = {'4x4': {'AB': {'FPRS': 6, 'PO': 41, 'IS': 70, 'SL': 88, 'DET': None},
                               'CD': {'FPRS': 6, 'PO': 41, 'IS': 69, 'SL': 88, 'DET': None},
                               'EF': {'FPRS': 6, 'PO': 41, 'IS': 70, 'SL': 88, 'DET': None},
                               'GH': {'FPRS': 6, 'PO': 41, 'IS': 69, 'SL': 88, 'DET': None}},
                       '10x10': {'AB': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'CD': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None},
                                 'EF': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'GH': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None}},
                       '20x20': {'AB': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'CD': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None},
                                 'EF': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'GH': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None}},
                       '60x30': {'AB': {'FPRS': 6, 'PO': 30, 'IS': 59, 'SL': 77, 'DET': None},
                                 'CD': {'FPRS': 6, 'PO': 30, 'IS': 58, 'SL': 77, 'DET': None},
                                 'EF': {'FPRS': 6, 'PO': 30, 'IS': 59, 'SL': 77, 'DET': None},
                                 'GH': {'FPRS': 6, 'PO': 30, 'IS': 58, 'SL': 77, 'DET': None}}}

# Update for June values
# TODO: [***] This is not very robust. Ideally, we'd like to search for the Image Slicer surface by its Zemax comment
# but at the moment, the naming on the Zemax files is inconsistent, so this will have to wait
focal_planes['HARMONI'] = {'4x4': {'AB': {'IS': 97, 'DET': None},
                                   'CD': {'IS': 96, 'DET': None},
                                   'EF': {'IS': 97, 'DET': None},
                                   'GH': {'IS': 96, 'DET': None}},
                           '10x10': {'AB': {'IS': 92, 'DET': None},
                                     'CD': {'IS': 91, 'DET': None},
                                     'EF': {'IS': 92, 'DET': None},
                                     'GH': {'IS': 91, 'DET': None}},
                           '20x20': {'AB': {'IS': 93, 'DET': None},
                                     'CD': {'IS': 92, 'DET': None},
                                     'EF': {'IS': 93, 'DET': None},
                                     'GH': {'IS': 92, 'DET': None}},
                           '60x30': {'AB': {'IS': 85, 'DET': None},
                                     'CD': {'IS': 84, 'DET': None},
                                     'EF': {'IS': 85, 'DET': None},
                                     'GH': {'IS': 84, 'DET': None}}
                           }


# Taken from Zemax example code
class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)

        # EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #      {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException(
                "License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if (self.TheApplication.LicenseStatus is
                constants.LicenseStatusType_PremiumEdition):
            return "Premium"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_ProfessionalEdition):
            return "Professional"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_StandardEdition):
            return "Standard"
        else:
            return "Invalid"


MAX_WAVES = 23
@functools.lru_cache(maxsize=MAX_WAVES)
def airy_and_slicer(surface, wavelength, scale_mas, PSF_window, N_window):
    """

    Since the calculations here are rather time-consuming
    and this function will be called multiple times (once for each configuration / slice)
    we use LRU cache to recycle previous results

    :param wavelength: wavelength in [microns]
    :param scale_mas: spaxel scale in [mas]
    :param PSF_window: length of the PSF window [microns]
    :param N_window: number of points of the PSF window [pixels]
    :return:
    """

    # Print message to know we are updating the cache
    print('Recalculating Airy Pattern for %.3f microns' % wavelength)

    # Plate scales [Px, Py] for each spaxel scale in mm / arcsec
    plate_scales = {'IS': {4.0: [125, 250], 10.0: [50, 100], 20.0: [25, 50], 60.0: [16.67, 16.67]},
                    'DET': {4.0: [3.75, 7.5], 10.0: [1.5, 3.0], 20.0: [0.75, 1.5], 60.0: [0.5, 0.5]}}
    plate_x = plate_scales[surface][scale_mas][0]
    plate_y = plate_scales[surface][scale_mas][1]

    # We know how many Microns the pixels of the Geometric PSF span [PSF_window / N_window]
    pix_sampling = PSF_window / N_window  # micron at the detector plane
    # Using the plate scale we calculate how many m.a.s each of those pixels have to span
    pix_scale_x = pix_sampling / plate_x  # milliarcsec / pixel
    pix_scale_y = pix_sampling / plate_y  # milliarcsec / pixel

    # Calculate the relative size of the pupil aperture needed to ensure the PSF is
    # sampled with the given pix_scale at the detector
    ELT_DIAM = 39
    MILIARCSECS_IN_A_RAD = 206265000
    pix_rad_x = pix_scale_x / MILIARCSECS_IN_A_RAD  # radians / pixel
    pix_rad_y = pix_scale_y / MILIARCSECS_IN_A_RAD
    RHO_APER_x = pix_rad_x * ELT_DIAM / (wavelength * 1e-6)
    RHO_APER_y = pix_rad_y * ELT_DIAM / (wavelength * 1e-6)
    RHO_OBSC_x = 0.30 * RHO_APER_x  # ELT central obscuration
    RHO_OBSC_y = 0.30 * RHO_APER_y  # ELT central obscuration

    # Sanity check
    PIX_RAD_x = RHO_APER_x * wavelength / ELT_DIAM * 1e-6
    PIX_RAD_y = RHO_APER_y * wavelength / ELT_DIAM * 1e-6
    PIX_MAS_x = PIX_RAD_x * MILIARCSECS_IN_A_RAD
    PIX_MAS_y = PIX_RAD_y * MILIARCSECS_IN_A_RAD

    # Define the ELT pupil mask
    N = 2048
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)

    # To get the anamorphic scaling we define the equation for an ellipse
    rho = np.sqrt((xx / RHO_APER_x) ** 2 + (yy / RHO_APER_y) ** 2)

    # (1) Propagate the Image Slicer Focal plane
    elt_mask = (RHO_OBSC_x / RHO_APER_x < rho) & (rho < 1.0)
    pupil = elt_mask * np.exp(1j * elt_mask)
    image_electric = fftshift(fft2(pupil))

    # plt.figure()
    # plt.imshow(elt_mask)

    # (1.1) Add slicer effect by masking
    # We mask the PSF covering a band of size 1x SPAXEL, depending on the scale
    # If we have 4x4 mas, then we cover a band of 4 mas over the PSF
    x_min, x_max = -N/2 * PIX_MAS_x, N/2 * PIX_MAS_x
    y_min, y_max = -N/2 * PIX_MAS_y, N/2 * PIX_MAS_y
    x_slice = np.linspace(x_min, x_max, N, endpoint=True)
    y_slice = np.linspace(y_min, y_max, N, endpoint=True)
    x_grid, y_grid = np.meshgrid(x_slice, y_slice)
    slicer_mask = np.abs(y_grid) < scale_mas / 2


    # ## Show the PSF both in [mas] space where it should be circular and in [pixel] space where it should be anamorphic
    # fig, ax = plt.subplots(1, 1)
    # img1 = ax.imshow((np.abs(image_electric))**2, extent=[x_min, x_max, y_min, y_max], cmap='bwr')
    # # plt.colorbar(img1, ax=ax)
    # ax.set_title(r'Airy Pattern | %.1f mas scale | Wavelength: %.3f $\mu$m' % (scale_mas, wavelength))
    # ax.set_xlabel(r'X [mas]')
    # ax.set_ylabel(r'Y [mas]')
    # ax.set_xlim([-10, 10])
    # ax.set_ylim([-10, 10])
    #
    # fig, ax = plt.subplots(1, 1)
    # img1 = ax.imshow((np.abs(image_electric))**2, extent=[-N/2, N/2, -N/2, N/2], cmap='bwr')
    # ax.set_title(r'Airy Pattern | %.1f mas scale | Wavelength: %.3f $\mu$m' % (scale_mas, wavelength))
    # ax.set_xlabel(r'Pixels [ ]')
    # ax.set_ylabel(r'Pixels [ ]')
    # ax.set_xlim([-100, 100])
    # ax.set_ylim([-100, 100])
    #
    # plt.show()

    # (2) Propagate the masked electric field to Pupil Plane
    pup_grating = ifft2(fftshift(slicer_mask * image_electric))
    # (2.1) Add pupil mask, this time without the central obscuration
    aperture_mask = rho < 1.0

    # (3) Propagate back to Focal Plane
    final_focal = fftshift(fft2(aperture_mask * pup_grating))
    final_psf = np.abs(final_focal)**2
    final_psf /= np.max(final_psf)

    # (4) Crop the PSF to fit to the necessary window to ease the convolutions
    min_pix, max_pix = N//2 - N_window//2, N//2 + N_window//2
    crop_psf = final_psf[min_pix:max_pix, min_pix:max_pix]

    # If we want to show the plots for Documentation

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # psf_airy = (np.abs(image_electric))**2
    # img1 = ax1.imshow(psf_airy, extent=[x_min, x_max, y_min, y_max], cmap='bwr')
    # ax1.axhline(y=scale_mas/2, linestyle='--', color='black')
    # ax1.axhline(y=-scale_mas/2, linestyle='--', color='black')
    # ax1.set_xlabel(r'X [mas]')
    # ax1.set_ylabel(r'Y [mas]')
    # ax1.set_xlim([-15, 15])
    # ax1.set_ylim([-15, 15])
    # ax1.set_title(r'Airy Pattern | Slicer Mask %.1f mas' % scale_mas)
    #
    # img2 = ax2.imshow(aperture_mask * (np.abs(pup_grating)**2), extent=[-1, 1, -1, 1], cmap='bwr')
    # ax2.set_title(r'Pupil Plane | Aperture Mask')
    # ax2.set_xlim([-0.25, 0.25])
    # ax2.set_ylim([-0.25, 0.25])
    #
    # img3 = ax3.imshow(final_psf, extent=[x_min, x_max, y_min, y_max], cmap='bwr')
    # ax3.set_xlabel(r'X [mas]')
    # ax3.set_ylabel(r'Y [mas]')
    # ax3.set_xlim([-15, 15])
    # ax3.set_ylim([-15, 15])
    # ax3.set_title(r'Diffraction Effects')
    # plt.show()

    return crop_psf


class DiffractionEffects(object):
    """
    A class to take care of diffraction effects
    for the PSF FWHM calculations
    """
    def __init__(self):

        return

    def add_diffraction(self, surface, psf_geo, PSF_window, scale_mas, wavelength):
        """
        Calculate the diffraction effects which include:
            - ELT pupil diffraction
            - Image Slicer + Grating

        and convolve it with a Geometric PSF to produce an
        estimate of the Diffraction PSF

        :param psf_geo: geometric PSF
        :param PSF_window: length of the geometric PSF window [microns]
        :param scale_mas: spaxel scale to consider [mas]
        :param wavelength: wavelength [microns]
        :return:
        """

        # Calculate the diffraction effects
        N_window = psf_geo.shape[0]
        diffraction = airy_and_slicer(surface=surface, wavelength=wavelength, scale_mas=scale_mas,
                                      PSF_window=PSF_window, N_window=N_window)

        # Convolve the diffraction effects with the geometric PSF
        psf_convol = convolve2d(psf_geo, diffraction, mode='same')
        # Normalize to peak 1.0
        psf_convol /= np.max(psf_convol)

        return psf_convol

    def gaussian2d(self, xy_data, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """
        A 2D Gaussian evaluated the points in xy_data with the following characteristics:
        Centred at (x0, y0), with sigma_x, sigma_y, rotated by theta radians
        with amplitude, and offset
        :param xy_data:
        :param x0:
        :param y0:
        :param sigma_x:
        :param sigma_y:
        :param theta:
        :param amplitude:
        :param offset:
        :return:
        """
        theta = 0.0
        (x, y) = xy_data
        x0 = float(x0)
        y0 = float(y0)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
        return g

    def fit_psf_to_gaussian(self, xx, yy, psf_data, x0, y0, sigmax0, sigmay0):
        """
        Fit a PSF array to a 2D Gaussian
        :param xx: grid of X values to evaluate the Gaussian
        :param yy: grid of Y values to evaluate the Gaussian
        :param psf_data: PSF array to fit
        :param x0: estimate of the X centre (typically geometric PSF centroid)
        :param y0: estimate of the Y centre (typically geometric PSF centroid)
        :return:
        """
        peak0, offset0 = 1.0, 0.0     # Guesses
        rotation0 = 0.0              # Guesses
        # # We guess that the FWHM will be around 2 detector pixels [30 microns]
        # # Using the formula for a Gaussian FWHM = 2 sqrt(2 ln(2)) Sigma
        # sigma_fwhm = 100e-3 / 2 * np.sqrt(2 * np.log(2))
        # sigmax0, sigmay0 = sigma_fwhm, sigma_fwhm

        guess_param = [x0, y0, sigmax0, sigmay0, peak0, offset0]
        bounds = ([-np.inf, -np.inf, 0.0, 0.0, 0, -np.inf],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # we don't let the theta vary more than +-45 degrees because that would
        # flip the values of FWHM_X and FWHM_Y

        while True:

            try:
                pars, cov = curve_fit(self.gaussian2d, (xx.ravel(), yy.ravel()), psf_data.ravel(),
                                      p0=guess_param, bounds=bounds)
                # Results of the fit for the sigmaX, sigmaY and theta
                sigmaX, sigmaY, theta = pars[2], pars[3], pars[4]
                # print("Successful Guassian Fit")
                break

            except RuntimeError:  # If it fails to converge
                # sigmaX, sigmaY, theta = np.nan, np.nan, np.nan
                print("\n[WARNING] Gaussian Fit of the PSF failed!")

                # try with other guesses
                deltax = np.random.uniform(low=-1.0, high=1.0)
                deltay = np.random.uniform(low=-1.0, high=1.0)
                new_sigmax0 = (1 + deltax) * sigmax0
                new_sigmay0 = (1 + deltay) * sigmay0
                print("Trying with new Sigmas: X %.3f & Y %.3f microns" % (1e3 * new_sigmax0, 1e3 * new_sigmay0))
                guess_param = [x0, y0, new_sigmax0, new_sigmay0, peak0, offset0]

            # if sigmaX > 0.0 and sigmaY > 0.0:  # we found a reasonable value
            #     break


        # try:
        #     pars, cov = curve_fit(self.gaussian2d, (xx.ravel(), yy.ravel()), psf_data.ravel(),
        #                           p0=guess_param, bounds=bounds)
        #     # Results of the fit for the sigmaX, sigmaY and theta
        #     sigmaX, sigmaY, theta = pars[2], pars[3], pars[4]
        #
        # except RuntimeError:    # If it fails to converge
        #
        #     # sigmaX, sigmaY, theta = np.nan, np.nan, np.nan
        #     print("\n[WARNING] Gaussian Fit of the PSF failed!")
        #
        #     # try with other guesses
        #     deltax = np.random.uniform(low=-1.0, high=1.0)
        #     deltay = np.random.uniform(low=-1.0, high=1.0)
        #     new_sigmax0 = (1 + deltax) * sigmax0
        #     new_sigmay0 = (1 + deltay) * sigmay0
        #     print("Trying with new Sigmas: X %.3f & Y %.3f microns" % (1e3 * new_sigmax0, 1e3*new_sigmay0))
        #     self.fit_psf_to_gaussian(xx, yy, psf_data, x0, y0, new_sigmax0, new_sigmay0)
        #     # plt.figure()
        #     # plt.imshow(psf_data)
        #     # plt.colorbar()
        #     # plt.show()

        fwhm_x = 2 * np.sqrt(2 * np.log(2)) * sigmaX * 1000
        fwhm_y = 2 * np.sqrt(2 * np.log(2)) * sigmaY * 1000

        return fwhm_x, fwhm_y, theta


diffraction = DiffractionEffects()

# TODO: [***] This should probably go, we no longer use it
def pixelate_crosstalk_psf(raw_psf, PSF_window, N_points, xt=0.02):

    pix_samp = PSF_window / N_points  # microns per PSF point
    group = int(15 / pix_samp)
    N = int(PSF_window / 15)
    pix_psf = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            crop = raw_psf[i * group:(i + 1) * group, j * group:(j + 1) * group]
            pix_psf[i, j] = np.mean(crop)

    # Add Crosstalk
    kernel = np.array([[0, xt, 0],
                       [xt, 1 - 4 * xt, xt],
                       [0, xt, 0]])
    cross_psf = convolve2d(pix_psf, kernel, mode='same')

    return cross_psf


def read_hdf5(path_to_file):
    """
    Read a HDF5 file containing some analysis results

    HDF5 files for the E2E analysis have the following 'Group' structure:

        (1) "Zemax Metadata": this Group is a dictionary containing metadata on the Zemax file used in the analysis
            this includes: file name, system mode, spaxel scale, IFU channel, grating, AO modes
            Plus the Git hash for the E2E Python code [for traceability]

        (2) Analysis Name 1: each analysis type (RMS_WFE, FWHM_PSF) will have an associated group. Within each group,
            there will be an arbitrary number of subgroups.
                - Subgroup #0: each subgroup represents an instance of that analysis type; for example an RMS WFE
                  ran at a specific surface, with a set of parameters. Each subgroup will containing a series of
                  datasets representing the results of the analysis, plus some associated metadata (surface, parameters)
                - Subgroup #N: another analysis ran at a different surface, or with different sampling
        (3) Analysis Name 2: another type of analysis
                - [...]

    TODO: at the moment this function only reads the metadata for Zemax and each Analysis.

    :param path_to_file:
    :return:
    """

    print("\nReading HDF5 file: ", path_to_file)
    file = h5py.File(path_to_file, 'r')

    # List the groups
    groups = list(file.keys())
    print("Groups available: ", groups)

    # Read Zemax Metadata
    zemax_metadata = {}
    print("\nZemax Metadata:")
    for key in file['Zemax Metadata'].attrs.keys():
        print('{} : {}'.format(key, file['Zemax Metadata'].attrs[key]))
        zemax_metadata[key] = file['Zemax Metadata'].attrs[key]

    # Read the analysis groups
    for group_name in groups:
        if group_name != 'Zemax Metadata':
            analysis_group = file[group_name]
            print('\nAnalysis: ', group_name)
            # For each Analysis Group we loop over subgroups
            for subgroup_key in analysis_group.keys():
                subgroup = analysis_group[subgroup_key]
                print('Subgroup #', subgroup_key)
                # List the metadata of the subgroup
                for att_key in subgroup.attrs.keys():
                    print('     {} : {}'.format(att_key, subgroup.attrs[att_key]))

    file.close()

    return zemax_metadata


def get_fields(system, info=False):
    """
    Read the Field Points for a given Optical System
    :param system: a system loaded with the Python Standalone Application
    :param info: boolen, whether to print the information
    :return: field_points, an array of size [N_fields, 2]
    """

    system_data = system.SystemData
    fields = system_data.Fields
    N_fields = fields.NumberOfFields

    if info is True:
        print("\nReading Field Points")
        print("Total Number of Field Points: %d" % N_fields)

    field_points = np.zeros((N_fields, 2))

    for k in np.arange(1, N_fields + 1):
        _field = fields.GetField(k)
        field_points[k - 1, 0] = _field.X
        field_points[k - 1, 1] = _field.Y

        if info is True:
            print("X = %.4f, Y = %.4f" % (_field.X, _field.Y))

    return field_points


def get_wavelengths(system, info=False):
    """
    Read the Wavelengths for a given Optical System
    :param system: a system loaded with the Python Standalone Application
    :param info: boolen, whether to print the information
    :return: field_points, an array of size [N_wavelengths]
    """

    system_data = system.SystemData
    wavelengths = system_data.Wavelengths
    N_wavelengths = wavelengths.NumberOfWavelengths

    if info is True:
        print("\nReading Wavelengths")
        print("Total Number of Wavelengths: %d" % N_wavelengths)

    wavelength_array = np.zeros(N_wavelengths)

    for k in np.arange(1, N_wavelengths + 1):
        _wave = wavelengths.GetWavelength(k).Wavelength
        wavelength_array[k - 1] = _wave

        if info is True:
            print("%.5f microns" % _wave)

    return wavelength_array


def define_pupil_sampling(r_obsc, N_rays, mode='random'):

    if mode == 'random':
        # Ideally we want to define a Uniform Distribution for the (X, Y) points in the unit disk
        # but sampling from the radial coordinates
        # If one defines a uniform distribution for (r, theta) the resulting (x, y) coordinates won't follow
        # a Uniform Distribution! there will be more points concentrated around de centre...

        # We can fix that by using the "Probability Density Under Transformation" as described in:
        # https://www.cs.cornell.edu/courses/cs6630/2015fa/notes/pdf-transform.pdf
        # Sampling as: r -> sqrt(eps_1) and theta -> 2 pi eps_2, with eps_1 and eps_2 uniform distributions [0, 1]
        # results in a (X, Y) which is uniform!

        r = np.sqrt(np.random.uniform(low=r_obsc**2, high=1.0, size=N_rays))
        theta = np.random.uniform(low=0.0, high=2 * np.pi, size=N_rays)
        px, py = r * np.cos(theta), r * np.sin(theta)

    else:
        raise NotImplementedError
        # TODO: Implemented a different pupil sampling, if needed

    return px, py


def create_zemax_file_list(which_system,
                           AO_modes=('NOAO', 'LTAO', 'SCAO', 'HCAO'),
                           scales=('4x4', '10x10', '20x20', '60x30'),
                           IFUs=('AB',),
                           grating=('VIS', 'IZJ', 'HK', 'IZ', 'J', 'H', 'K', 'Z_HIGH', 'H_HIGH', 'K_LONG', 'K_SHORT')):
    """
    Creates a list of Zemax file names using the naming convention that Matthias defined for the E2E model

    ['CRYO_PO4x4_IFU-AB_SPEC-VIS.zmx',
    'CRYO_PO4x4_IFU-AB_SPEC-IZJ.zmx',
    'CRYO_PO4x4_IFU-AB_SPEC-HK.zmx',
    ...,]

    Output: file_list, and settings_list

    settings_list is a list of dictionaries containing the setting (scale, IFU, spectral band) of each file
    we need that in the future to postprocess the results of each Zemax file

    :param which_system: which system do we want to analyze, can be 'ELT' or 'HARMONI' or 'IFS'
    :param AO_modes: a list of AO modes that can be ['NOAO', 'LTAO', 'SCAO', 'HCAO']
    :param scales: a list of Spaxel Scales that can be ['4x4', '10x10', '20x20', '60x30']
    :param IFUs: a list of IFU subfields that can be ['AB', ...]
    :param grating: a list of Gratings that can be ['VIS', 'IZJ', 'HK', 'IZ', 'J', 'H', 'K', 'Z_HIGH', 'H_HIGH', 'K_LONG', 'K_SHORT']
    :return:
    """

    if which_system not in ['ELT', 'HARMONI', 'IFS']:
        raise ValueError("System must be 'ELT' or 'HARMONI' or 'IFS'")

    if which_system == 'IFS':     # Only the IFS, starting at the CRYO

        file_list = []
        settings_list = []          # List of settings dictionaries
        for scale in scales:
            name_scale = 'CRYO_PO' + scale
            for ifu in IFUs:
                if scale != '4x4':
                    name_IFU = '_IFU' + ifu
                else:
                    name_IFU = '_IFU-' + ifu            # For 4x4 the IFU is reversed
                for spec in grating:

                    if ifu == 'AB' or ifu == 'EF':
                        # For 4x4 the SPEC is also reversed, we add a '-' sign
                        name_spec = '_SPEC-' + spec if scale == '4x4' else '_SPEC' + spec

                    if ifu == 'CD' or ifu == 'GH': # _SPEC-K_LONG
                        name_spec = '_SPEC' + spec if scale == '4x4' else '_SPEC-' + spec

                    filename = name_scale + name_IFU + name_spec + '.zmx'
                    file_list.append(filename)

                    # Save settings:
                    sett_dict = {'system': which_system, 'scale': scale, 'ifu': ifu, 'grating': spec}
                    settings_list.append(sett_dict)

        return file_list, settings_list

    if which_system == 'HARMONI':     # The same as 'IFS' but including the FPRS with the AO_modes

        file_list = []
        settings_list = []
        # Create a list of file names from the CRYO onwards
        ifs_file_list, ifs_sett_list = create_zemax_file_list(which_system='IFS', scales=scales, IFUs=IFUs, grating=grating)
        # Add FPRS_ + AO mode as prefix
        for AO_mode in AO_modes:
            name_fprs = 'FPRS_' + AO_mode + '_'
            for ifs_file, ifs_sett in zip(ifs_file_list, ifs_sett_list):
                filename = name_fprs + ifs_file
                file_list.append(filename)

                ifs_sett['AO_mode'] = AO_mode       # Add AO mode to the settings
                ifs_sett['system'] = which_system   # Override the system mode to HARMONI, not IFS
                settings_list.append(ifs_sett)

        return file_list, settings_list

    if which_system == 'ELT':         # The same as 'HARMONI' but including the ELT

        # Create a list of file names from HARMONI
        harmoni_files_list, harmoni_sett_list = create_zemax_file_list(which_system='HARMONI', AO_modes=AO_modes,
                                                                       scales=scales, IFUs=IFUs, grating=grating)
        # Add ELT as prefix
        file_list = ['ELT_' + harmoni_file for harmoni_file in harmoni_files_list]
        settings_list = []
        for harm_sett in harmoni_sett_list:
            harm_sett['system'] = which_system      # Override the system mode to ELT, not HARMONI
            settings_list.append(harm_sett)

        return file_list, settings_list

    return


class AnalysisGeneric(object):

    """
    Object to perform fairly generic analyses on Zemax files created by the E2E model

    If you want to run your own custom analysis the best way is to use Class Inheritance
    Create a child class from this AnalysisGeneric object
    Define a method (function, typically static method) in the child class representing the 'analysis_function'
    that self.run_analysis will call. That method / function should perform all the operations that
    you are interested in for your analysis

    """

    def __init__(self, zosapi, *args, **kwargs):
        """
        We basically need a ZOS API Standalone Application to
        open and close the files we want to analyze
        :param zosapi:
        :param args:
        :param kwargs:
        """

        self.zosapi = zosapi

    def run_analysis(self, analysis_function, results_shapes, results_names, files_dir, zemax_file, results_path,
                     wavelength_idx=None, configuration_idx=None, surface=None,
                     *args, **kwargs):
        """

        Run some analysis on the Optical System over a range of Wavelengths and Configurations
        For this, we need some Python function "analysis_function" that receives as input
        (Optical System, Wavelength Number, Configuration Number, Surface Number, *args, **kwargs)
        *args and **kwargs represent whatever other information our function might need for the
        calculations. This could be something like Number of Fields, Number of Rays, or some other
        Zemax analysis parameters

        The "run_analysis" function will loop over our chosen Wavelengths and Configurations (Slices)
        and apply the specified "analysis_function"

        ~~ Extremely IMPORTANT! ~~
        We are aware that the shape and type of results will vary for each analysis
        In order to accommodate this variability we use the following strategy:

            You can customize the output of 'run_analysis' by using a list 'results_names'
            that contains the names of the variables that your 'analysis_function' will provide
            as well as a list 'results_shapes' containing the shape of such variables

            This code 'run_analysis' will dynamically create such variables using the globals() dictionary
            such that any array of arbitrary shape [N_waves, N_configs, Nx, Ny, ..., Nz] can be
            accommodated.

            For example, if we want to run a Spot Diagram analysis we would call
            'run_analysis' with analaysis_function=SpotDiagram.analysis_function and provide as arguments
            results_names = ['xy'], results_shapes = [(N_fields, N_rays^2, 2)]

            'run_analysis' would then loop over those lists (in this case we only have 1 array)
            and create the global variable xy, with shape [N_waves, N_config, N_fields, N_rays^2, 2]
            and loop over the Wavelengths and Configurations applying the 'analysis_function'

        All of this is done for a given "zemax_file" in the "files_dir". This is a template so if you
        want to run the analysis over multiple files you can create a Child Class and add a method
        that loops over the files (see RMS_ XXX)

        :param analysis_function: arbitrary function that does the calculations needed for our analyses
        :param results_shapes: list of shapes for the arrays that 'analysis_function' produces
        :param results_shape: list of names of the arrays that 'analysis_function' produces
        :param files_dir: path to where you keep the Zemax files
        :param zemax_file: the name of the Zemax file you want to analyze
        :param results_path: path where you will save the results
        :param wavelength_idx: list of Zemax wavelength numbers. If left as None, it will analyze all Wavelengths
        :param configuration_idx: list of Zemax configurations. If left as None, it will analyze all Configurations
        :param surface: zemax surface number at which we will compute the analysis. If left as None, it will use the Image Plane
        :param args: whatever other arguments your "analysis_function" may need
        :param kwargs: whatever other keyword arguments your "analysis_function" may need
        :return:
        """

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(files_dir, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(files_dir, zemax_file), False)
        file_name = zemax_file.split(".")[0]            # Remove the ".zmx" suffix

        # Check if the results directory already exists
        results_dir = os.path.join(results_path, file_name)
        print("Results will be saved in: ", results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)           # If not, create the directory to store results

        # Get some info on the system
        system = self.zosapi.TheSystem                      # The Optical System
        MCE = system.MCE                                    # Multi Configuration Editor
        LDE = system.LDE                                    # Lens Data Editor
        N_surfaces = LDE.NumberOfSurfaces
        N_configs = MCE.NumberOfConfigurations

        # print("\nSystem Data:")
        print("Number of Surfaces: ", N_surfaces)
        # print("Number of Configurations / Slices: ", N_configs)
        _wavelengths = get_wavelengths(system, info=False)
        N_waves = _wavelengths.shape[0]
        # field_points = get_fields(system, info=False)

        # Let's deal with the data format
        if wavelength_idx is None:         # We use all the available wavelengths
            wavelengths = _wavelengths
            wavelength_idx = np.arange(1, N_waves + 1)
        else:     # We have received a list of wavelength indices
            wavelengths = [_wavelengths[idx - 1] for idx in wavelength_idx]
            N_waves = len(wavelength_idx)

        if configuration_idx is None:
            configurations = np.arange(1, N_configs + 1)
        else:       # We have received a list of configurations
            configurations = configuration_idx
            N_configs = len(configuration_idx)
        N_slices = N_configs

        if surface is None:     # we go to the final image plane
            surface = system.LDE.NumberOfSurfaces - 1

        N_surfaces = system.LDE.NumberOfSurfaces
        if surface != (N_surfaces - 1):
            # The Surface is not the final Image Plane. We must force Zemax to IGNORE
            # all surfaces after our surface of interest so that the RWCE operands works
            surfaces_to_ignore = np.arange(surface + 1, N_surfaces)
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = True

            last_surface = system.LDE.GetSurfaceAt(surface)
            thickness = last_surface.Thickness
            print("Thickness: ", last_surface.Thickness)
            # image = system.LDE.GetSurfaceAt(N_surfaces - 1)
            # print("\nImage Name: ", image.Comment)

            # Temporarily set the Thickness to 0
            last_surface.Thickness = 0.0
            print("Thickness: ", last_surface.Thickness)

        # Double check we are using the right surface
        print("\nSurface Name: ", system.LDE.GetSurfaceAt(surface).Comment)

        # Customize the initilization of the results array so that we can vary the shapes
        # some analysis may be [spaxels_per_slice, N_slices, N_waves] such as an RMS WFE map
        # others might be [N_fields, N_rays, N_slices, N_waves] such as a Spot Diagram

        # print("\nDynamically creating Global Variables to store results")
        for _name, _shape in zip(results_names, results_shapes):
            # print("Variable: %s with shape (N_waves, N_configs) + " % _name, _shape)
            globals()[_name] = np.empty((N_waves, N_slices) + _shape)
            # print(globals()[_name].shape)

        # print("\nApplying 'analysis_function': ", analysis_function.__name__)
        print("At Surface #%d | Image Plane is #%d" % (surface, N_surfaces - 1))
        print("For Wavelength Numbers: ", wavelength_idx)
        print("For Configurations %d -> %d" % (configurations[0], configurations[-1]))

        # Main part of the Analysis. Here we loop over the Wavelengths and Configurations
        start = time()
        print("\nAnalysis Started")
        for k, (wave_idx, wavelength) in enumerate(zip(wavelength_idx, wavelengths)):
            print("Wavelength #%d: %.3f" % (wave_idx, wavelength))

            for j, config in enumerate(configurations):

                # print("Wavelength #%d: %.3f | config: %d" %(wave_idx, wavelength, config))

                # Here is where the arbitrary "analysis_function" will be called for each Wavelength and Configuration
                results = analysis_function(system=system, wave_idx=wave_idx, config=config, surface=surface,
                                            *args, **kwargs)

                # Store the result values in the arrays:
                for _name, res in zip(results_names, results):
                    globals()[_name][k, j] = res

        # Unignore the surfaces
        if surface != (N_surfaces - 1):
            last_surface = system.LDE.GetSurfaceAt(surface)
            last_surface.Thickness = thickness
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = False

        self.zosapi.CloseFile(save=False)
        analysis_time = time() - start
        print("\nAnalysis done in %.1f seconds" % analysis_time)
        time_per_wave = analysis_time / len(wavelengths)
        print("%.2f seconds per Wavelength" % time_per_wave)

        # local_keys = [key for key in globals().keys()]
        # print(local_keys)

        # We return a list containing the results, and the value of the wavelengths
        list_results = [globals()[_name] for _name in results_names]
        list_results.append(wavelengths)

        return list_results

    def template_analysis_function(self, system, wave_idx, config, surface, *args, **kwargs):
        """
        Template for the analysis function

        Typically we pass as arguments the 'system' so that any information about it can be accessed
        The Zemax Wavelength number 'wave_idx' and Configuration 'config' at which we will apply the analysis
        And the Zemax surface number 'surface' at which we will apply the analysis

        We also include *args and **kwargs to accommodate other parameters your analysis may need
        For instance, the number of rings to be used in RMS wavefront calculations

        :param system:
        :param wave_idx:
        :param config:
        :param surface:
        :param args:
        :param kwargs:
        :return:
        """

        # Write your function operations here

        # This will typically contain a lot of ZOS API commands such as:
        # - Setting the Current configuration:          system.MCE.SetCurrentConfiguration(config)
        # - Reading the Field Points:                   sysField = system.SystemData.Fields
        #                                               fx = sysField.GetField(1).X
        #                                               fy = sysField.GetField(1).Y
        # - Running some Ray Trace:                     raytrace = system.Tools.OpenBatchRayTrace()
        #                                               normUnPolData = raytrace.CreateNormUnpol(N_rays,
        #                                                                        constants.RaysType_Real, toSurface)


        # You should return a list of arrays

        return

    def save_hdf5(self, analysis_name, analysis_metadata, list_results, results_names, file_name, file_settings, results_dir):

        # First thing is to create a separate folder within the results directory for this analysis
        hdf5_dir = os.path.join(results_dir, 'HDF5')
        print("Analysis Results will be saved in folder: ", hdf5_dir)
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)           # If not, create the directory to store results

        hdf5_file = file_name + '.hdf5'
        # Check whether the file already exists
        if os.path.isfile(os.path.join(hdf5_dir, hdf5_file)):   # Overwrite it
            print("HDF5 file already exists. Adding analysis")
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'r+') as f:

                # Here we have 2 options: either we are adding another analysis type or we are adding the same type
                # but with different settings (such as at a different surface)
                file_keys = list(f.keys())
                if analysis_name in file_keys:
                    print("Analysis type already exists")
                    analysis_group = f[analysis_name]
                    # we need to know how many analyses of the same type already exist
                    subgroup_keys = list(analysis_group.keys())
                    subgroup_number = len(subgroup_keys)        # if [0, 1] already exist we call it '2'
                    subgroup = analysis_group.create_group(str(subgroup_number))
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

                else:       # It's a new analysis type

                    # (2) Create a Group for this analysis
                    analysis_group = f.create_group(analysis_name)
                    # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                    subgroup = analysis_group.create_group('0')
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

        else:       # File does not exist, we create it now
            print("Creating HDF5 file: ", hdf5_file)
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as f:

                # (1) Save Zemax Metadata
                zemax_metadata = f.create_group('Zemax Metadata')
                zemax_metadata.attrs['(1) Zemax File'] = file_name
                zemax_metadata.attrs['(2) System Mode'] = file_settings['system']
                zemax_metadata.attrs['(3) Spaxel Scale'] = file_settings['scale']
                zemax_metadata.attrs['(4) IFU'] = file_settings['ifu']
                zemax_metadata.attrs['(5) Grating'] = file_settings['grating']
                AO = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else 'NA'
                zemax_metadata.attrs['(6) AO Mode'] = AO

                # (2) Create a Group for this analysis
                analysis_group = f.create_group(analysis_name)

                # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                subgroup = analysis_group.create_group('0')
                # Save results datasets
                for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                    data = subgroup.create_dataset(array_name, data=array)
                # Save analysis metadata
                subgroup.attrs['E2E Python Git Hash'] = sha
                subgroup.attrs['Analysis Type'] = analysis_name
                date_created = datetime.datetime.now().strftime("%c")
                subgroup.attrs['Date'] = date_created
                subgroup.attrs['At Surface #'] = file_settings['surface']
                # Add whatever extra metadata we have:
                for key, value in analysis_metadata.items():
                    subgroup.attrs[key] = value

        return

    # TODO: [***] This plot_and_save is deprecated, we shouldn't "post-process" the results in e2e_analysis.py
    def plot_and_save(self, analysis_name, list_results, file_name, file_settings, results_dir, wavelength_idx):
        """
        Semi-"generic" plot and save routine to post-process the analysis results
        It works for analysis types such as RMS Wavefront Error where you have a list of arrays like
        [MAIN_RESULTS, OBJ_XY, FOC_XY, WAVELENGTHS]
        and MAIN_RESULTS is something like the RMS WFE, of shape [N_waves, N_configs, N_fields]
        i.e. a given value for each Field Point, at each Configuration (slice) and Wavelength

        :param analysis_name: string, to add the analysis name to the plots
        :param list_results: list containing the arrays [MAIN_RESULTS, OBJ_XY, FOC_XY, WAVELENGTHS]
        :param file_name: string, name of the Zemax file
        :param file_settings: dictionary of settings, so we can access things like the spaxel scale, the IFU section...
        :param results_dir: path to where the results will be saved
        :param wavelength_idx: list of wavelength numbers used, or None for All Wavelengths
        :return:
        """

        # First thing is to create a separate folder within the results directory for this analysis
        analysis_dir = os.path.join(results_dir, analysis_name)
        print("Analysis Results will be saved in folder: ", analysis_dir)
        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)           # If not, create the directory to store results

        analysis_array = list_results[0]  # The first is always the Analysis results
        object_coord = list_results[1]
        focal_coord = list_results[2]
        wavelengths = list_results[-1]

        N_waves = analysis_array.shape[0]

        if wavelength_idx is None:
            wave_idx = np.arange(1, N_waves + 1)
        else:
            wave_idx = wavelength_idx

        # Read the settings
        system = file_settings['system']
        surface_number = file_settings['surface']
        scale, ifu, grating = file_settings['scale'], file_settings['ifu'], file_settings['grating']
        ao_mode = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else None
        # Find out the NameCode for the surface
        surface_codes = focal_planes[system][scale]
        surface_name = list(surface_codes.keys())[list(surface_codes.values()).index(surface_number)]
        # surface_number = str(surface) if surface is not None else '_IMG'

        print("Metadata Review")
        print("Filename: ", file_name)
        print("System: ", system)
        print("Scale: ", scale)
        print("IFU: ", ifu)
        print("Grating: ", grating)
        print("AO mode: ", ao_mode)
        print("Surface Code: ", surface_name)
        print("Surface Number: ", surface_number)

        ### ---------------------------------- Write Results File -------------------------------------------------- ###
        name_code = file_name + '_' + analysis_name + '_SURF_' + surface_name

        # Save Metadata of the Analysis
        output_name = os.path.join(analysis_dir, name_code + "_METADATA.txt")
        metadata_file = open(output_name, "w")
        metadata_file.write("Analysis: %s \n" % analysis_name)
        metadata_file.write("Zemax File: %s \n" % file_name)
        metadata_file.write("At Surface: #%s, %s \n" % (str(surface_number), surface_name))
        metadata_file.write("System Mode: %s \n" % system)
        metadata_file.write("Spaxel Scale: %s \n" % scale)
        metadata_file.write("IFU section: %s \n" % ifu)
        metadata_file.write("AO mode: %s \n" % ao_mode)
        metadata_file.write("Wavelength Numbers: [")
        for idx in wave_idx:
            metadata_file.write(" %d " % idx)
        metadata_file.write(" ] \n")
        metadata_file.write("Wavelengths [microns] : [")
        for wave in wavelengths:
            metadata_file.write(" %.4f " % wave)
        metadata_file.write(" ] \n")

        # Save the arrays
        np.save(os.path.join(analysis_dir, name_code), analysis_array)
        np.save(os.path.join(analysis_dir, name_code + '_OBJECT_COORD'), object_coord)
        np.save(os.path.join(analysis_dir, name_code + '_FOCAL_COORD'), focal_coord)
        np.save(os.path.join(analysis_dir, name_code + '_WAVELENGTHS'), wavelengths)

        # L = ["This is Delhi \n", "This is Paris \n", "This is London \n"]
        #
        # # \n is placed to indicate EOL (End of Line)
        # metadata_file.write("Hello \n")
        # metadata_file.writelines(L)
        metadata_file.close()


        ### --------------------------------- PLOTS and FIGURES --------------------------------------------------- ###

        # Statistics
        means, stds, mins, maxs = np.zeros(N_waves), np.zeros(N_waves), np.zeros(N_waves), np.zeros(N_waves)
        for i in range(N_waves):

            mu0 = np.mean(analysis_array[i])
            std0 = np.std(analysis_array[i])
            means[i], stds[i] = mu0, std0
            mins[i] = np.min(analysis_array[i])
            maxs[i] = np.max(analysis_array[i])

            if i in [0, N_waves//2, N_waves - 1]:    # Only show Min | Central | Max wavelengths

                # fig = plt.figure()
                # plt.hist(analysis_array[i].flatten(), bins=10, histtype='step')
                # plt.xlabel(analysis_name)
                # plt.ylabel(r'Frequency')
                # plt.xlim([mu0 - 5 * std0, mu0 + 5 * std0])
                # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_WAVE%d' % (wave_idx[i])
                # plt.title(fig_name)
                # if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                #     os.remove(os.path.join(analysis_dir, fig_name))
                # fig.savefig(os.path.join(analysis_dir, fig_name))
                # plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                sns.distplot(analysis_array[i].flatten(), ax=ax, axlabel=analysis_name)
                fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_WAVE%d' % (wave_idx[i])
                # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_WAVE%d' % (wave_idx[i])
                plt.title(fig_name)
                if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                    os.remove(os.path.join(analysis_dir, fig_name))
                fig.savefig(os.path.join(analysis_dir, fig_name))
                plt.close(fig)

        fig = plt.figure()
        plt.plot(wavelengths, means, color='black', label='Mean')
        plt.fill_between(wavelengths, means - stds, means + stds, alpha=0.2, color='salmon', label=r'$\pm \sigma$')
        plt.plot(wavelengths, maxs, color='crimson', label='Max')
        plt.plot(wavelengths, mins, color='blue', label='Min')
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(analysis_name)
        fig_name = file_name + '_' + analysis_name + '_SURF' + surface_name
        plt.title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        fig.savefig(os.path.join(analysis_dir, fig_name))
        plt.close(fig)

        # (1) Object Field Coordinates [wavelengths, configs, N_fields, xy]
        # Independent of wavelength
        plt.figure()
        plt.scatter(object_coord[0, :, :, 0], object_coord[0, :, :, 1], s=3)
        plt.xlabel(r'Object X [mm]')
        plt.ylabel(r'Object Y [mm]')
        plt.axes().set_aspect('equal')
        fig_name = file_name + '_XY_OBJECT'
        plt.title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        plt.savefig(os.path.join(analysis_dir, fig_name))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

        # (2) Focal Plane
        # Positions might depend on Wavelength (at the detector)
        # so we show all wavelengths
        colors = cm.Reds(np.linspace(0.5, 1, N_waves))
        plt.figure()
        for i in range(N_waves):
            plt.scatter(focal_coord[i, :, :, 0], focal_coord[i, :, :, 1], s=3, color=colors[i])
        plt.xlabel(r'Focal X Coordinate [mm]')
        plt.ylabel(r'Focal Y Coordinate [mm]')
        fig_name = file_name + '_XY_FOCAL_SURF' + surface_name
        plt.title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        plt.savefig(os.path.join(analysis_dir, fig_name))
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

        # (3) Analysis Results
        # This might be very tricky
        # Reference: both Object and Focal coordinates

        # (A) Object Coordinates
        # Independent of wavelength
        i_wave = 0
        x, y = object_coord[i_wave, :, :, 0].flatten(), object_coord[i_wave, :, :, 1].flatten()
        triang = tri.Triangulation(x, y)
        fig, ax = plt.subplots(1, 1)
        tpc = ax.tripcolor(triang, analysis_array[i_wave].flatten(), shading='flat', cmap='jet')
        ax.scatter(x, y, s=2, color='black')
        ax.set_xlabel(r'Object X [mm]')
        ax.set_ylabel(r'Object Y [mm]')
        ax.set_aspect('equal')
        plt.colorbar(tpc, ax=ax, orientation='horizontal')
        fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_OBJ_WAVE%d' % (wave_idx[i_wave])
        # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_OBJ_WAVE%d' % (wave_idx[i_wave])
        ax.set_title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        plt.savefig(os.path.join(analysis_dir, fig_name))
        # plt.show(block=False)
        # plt.pause(0.1)
        plt.close()

        # (B) Focal Coordinates

        if surface_number != None:     # Not a Detector Plane -> Focal has the same position for all wavelengths?

            # We have to loop over wavelengths
            for i in [0, N_waves // 2, N_waves - 1]:
                x, y = focal_coord[i, :, :, 0].flatten(), focal_coord[i, :, :, 1].flatten()
                triang = tri.Triangulation(x, y)
                fig, ax = plt.subplots(1, 1)
                tpc = ax.tripcolor(triang, analysis_array[i].flatten(), shading='flat', cmap='jet')
                ax.scatter(x, y, s=2, color='black')
                ax.set_xlabel(r'Focal X [mm]')
                ax.set_ylabel(r'Focal Y [mm]')
                ax.set_aspect('equal')
                plt.colorbar(tpc, ax=ax, orientation='horizontal')
                fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_FOC_WAVE%d' % (wave_idx[i])
                # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_FOC_WAVE%d' % (wave_idx[i])
                ax.set_title(fig_name)
                if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                    os.remove(os.path.join(analysis_dir, fig_name))
                plt.savefig(os.path.join(analysis_dir, fig_name))
                # plt.show(block=False)
                # plt.pause(0.1)
                plt.close()

        elif surface_number == None:  # Detector Plane -> Focal will depend on wavelength

            # Average across spaxel scales?
            x, y = focal_coord[:, :, :, 0], focal_coord[:, :, :, 1]
            x, y = np.mean(x, axis=2).flatten(), np.mean(y, axis=2).flatten()
            analysis_array = np.mean(analysis_array, axis=2)


            # (B) Focal Coordinates
            # All Wavelengths at once
            # x, y = focal_coord[:, :, :, 0].flatten(), focal_coord[:, :, :, 1].flatten()
            triang = tri.Triangulation(x, y)
            fig, ax = plt.subplots(1, 1)
            tpc = ax.tripcolor(triang, analysis_array.flatten(), shading='flat', cmap='jet')
            # ax.scatter(x, y, s=2, color='black')
            ax.set_xlabel(r'Focal Plane X [mm]')
            ax.set_ylabel(r'Focal Plane Y [mm]')
            ax.set_aspect('equal')
            plt.colorbar(tpc, ax=ax, orientation='horizontal')
            fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_FOC'
            # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_FOC'
            ax.set_title(fig_name)
            if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                os.remove(os.path.join(analysis_dir, fig_name))
            plt.savefig(os.path.join(analysis_dir, fig_name))
            # plt.show(block=False)
            # plt.pause(0.1)
            plt.close()

        return


class SpotDiagramAnalysis(AnalysisGeneric):

    # NOTE: we stop using this Analysis very early on (might need maintenance) but feel free to copy / adapt

    """
    Example of a custom analysis - Spot Diagrams

    This is how it works:
        - We inherit the AnalysisGeneric class
        - We define analysis_function_spot_diagram as the 'analysis_function'
        - self.run_analysis will loop over Wavelengths and Configurations calling analysis_function_spot_diagram
        - self.loop_over_files will loop over the E2E model files, calling run_analysis and saving plots

    """

    @staticmethod
    def analysis_function_spot_diagram(system, wave_idx, config, field, surface, N_rays):
        """
        Calculate the Spot Diagram for a given Wavelength number and Configuration
        for a list of Field Points (field), at a specific Surface

        For the Pupil Sampling we do the following
        we sample with a rectangular grid for P_x between [-1, 1] with N_rays
        which gives an XY grid for (P_x, P_y) with a total of N_rays^2
        Obviously, some rays lie outside the pupil (P_x ^ 2 + P_y ^ 2) > 1
        so those rays are ignored in the Ray Tracing. We only trace meaningful rays!

        But in order to avoid the further complication of finding out how many rays
        actually make sense, which will change as a function of the N_rays we choose
        and would force us to adapt the size of the arrays a posteriori
        we always keep the results as [N_rays ^ 2] in size, and fill with NaN
        those rays which will not be traced.
        This is not a problem because Python deals with NaN values very well.
        We have np.nanmax(), np.nanmean() functions, and the Spot Diagram plots
        will automatically ignore any NaN value.

        This function will return a list of arrays, containing a single entry
        an array of shape [N_fields, N_rays^2, 2] containing the (X, Y) coordinates
        of each ray for each field

        :param system: a system loaded with the Python Standalone Application
        :param wave_idx: Zemax wavelength number
        :param config: Zemax configuration number
        :param field: a list of Zemax field point numbers
        :param surface: Zemax surface number
        :param N_rays: Number of rays along the P_x [-1, 1] line. Total rays = N_rays ^ 2
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        N_fields = sysField.NumberOfFields

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # Pupil Rays
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx**2 + pyy**2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        XY = np.empty((len(field), N_rays ** 2, 2))
        XY[:] = np.NaN          # Fill it with Nan so we can discard the rays outside the pupil

        # Loop over each Field computing the Spot Diagram
        for j, field_idx in enumerate(field):

            fx, fy = sysField.GetField(field_idx).X, sysField.GetField(field_idx).Y
            hx, hy = fx / r_max, fy / r_max      # Normalized field coordinates (hx, hy)

            raytrace = system.Tools.OpenBatchRayTrace()
            # remember to specify the surface to which you are tracing!
            normUnPolData = raytrace.CreateNormUnpol(N_rays_inside, constants.RaysType_Real, surface)

            for (p_x, p_y) in zip(px, py):
                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

            # Run the RayTrace for the whole Slice
            CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
            normUnPolData.StartReadingResults()
            for i in range(N_rays_inside):
                output = normUnPolData.ReadNextResult()
                if output[2] == 0 and output[3] == 0:
                    XY[j, i, 0] = output[4]
                    XY[j, i, 1] = output[5]

            normUnPolData.ClearData()
            CastTo(raytrace, 'ISystemTool').Close()

        # It is extremely important to output a LIST of arrays even if you only need 1 array
        # Otherwise, if you only have 1 result array, the reading of the results in run_analysis
        # will loop over the first axis and Python will broadcast incorrectly

        return [XY]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, field=(1,), N_rays=20, plot_mode='Fields'):


        N_fields = len(field)
        results_names = ['XY']
        results_shapes = [(N_fields, N_rays**2, 2)]             # [N_fields, N_rays^2, (x,y)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        for zemax_file in file_list:

            file_name = zemax_file.split('.')[0]

            list_results = self.run_analysis(analysis_function=self.analysis_function_spot_diagram,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, field=field, N_rays=N_rays)

            xy, wavelengths = list_results

            if plot_mode == 'Fields':
                self.plot_fields(xy, wavelengths, wavelength_idx, configuration_idx, field,
                                 surface, file_name, results_path)

            return

    def plot_fields(self, xy, wavelengths, wavelength_idx, configuration_idx, fields, surface, file_name, result_path,
                    scale=5, cmap=cm.Reds):

        file_dir = os.path.join(result_path, file_name)
        spot_dir = os.path.join(file_dir, 'SPOT_DIAGRAM_FIELDS')
        if not os.path.exists(spot_dir):
            os.mkdir(spot_dir)           # If not, create the directory to store results

        # xy has shape [N_waves, N_configs, N_fields, N_rays^2, 2]
        N_waves, N_configs = xy.shape[0], xy.shape[1]
        N_fields = len(fields)
        colors = cmap(np.linspace(0.5, 1.0, N_waves))

        if wavelength_idx is None:
            wavelength_idx = np.arange(1, N_waves + 1)
        if surface is None:
            surface = '_IMG'

        for i in range(N_waves):
            wave = wavelengths[i]
            wave_idx = wavelength_idx[i]
            for j in range(N_configs):
                config = configuration_idx[j]

                fig_name = file_name + '_SPOTDIAG_SURF' + str(surface) + '_WAVE%d' % wave_idx + '_CONFIG%d' % config
                data = xy[i, j]

                deltax = max([np.nanmax(np.abs(data[k, :, 0] - np.nanmean(data[k, :, 0]))) for k in range(N_fields)])
                deltay = max([np.nanmax(np.abs(data[k, :, 1] - np.nanmean(data[k, :, 1]))) for k in range(N_fields)])
                xmax = 1.2 * max(deltax, deltay)        # some oversizing
                fig, axes = plt.subplots(1, N_fields, figsize=(scale*N_fields, scale))
                alpha = np.linspace(0.5, 1.0, N_fields)
                for k in range(N_fields):
                    ax = axes[k]
                    ax.set_aspect('equal')
                    x, y = data[k, :, 0], data[k, :, 1]
                    mx, my = np.nanmean(x), np.nanmean(y)
                    dx, dy = x - mx, y - my
                    ax.scatter(dx, dy, s=3, color=colors[i], alpha=alpha[k])
                    ax.set_title('Field #%d ($\lambda$: %.4f $\mu$m, cfg: %d)' % (fields[k], wave, config))
                    ax.set_xlabel(r'X [$\mu$m]')
                    ax.set_ylabel(r'Y [$\mu$m]')
                    ax.set_xlim([-xmax, xmax])
                    ax.set_ylim([-xmax, xmax])
                    if k > 0:
                        ax.get_yaxis().set_visible(False)

                if os.path.isfile(os.path.join(spot_dir, fig_name)):
                    os.remove(os.path.join(spot_dir, fig_name))
                fig.savefig(os.path.join(spot_dir, fig_name))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close('all')

        return


class SpotDiagramDetector(AnalysisGeneric):

    def analysis_functions_spots(self, system, wave_idx, config, surface, N_rays, reference='ChiefRay'):

        # colors = cm.Reds(np.linspace(0.5, 1, 23))
        # color = colors[wave_idx - 1]


        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # Pupil Rays
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx**2 + pyy**2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        XY = np.empty((1, N_rays_inside + 1, 2))     # Rays inside plus 1 for the chief ray extra

        obj_xy = np.zeros((1, 2))        # (fx, fy) coordinates
        foc_xy = np.empty((1, 2))        # raytrace results of the Centroid

        if config == 1:
            print("\nTracing %d rays to calculate FWHM PSF" % (N_rays_inside))

        # fig, axes = plt.subplots(1, N_fields)

        # Loop over each Field computing the Spot Diagram


        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / r_max, fy / r_max      # Normalized field coordinates (hx, hy)

        j = 0
        obj_xy[j, :] = [fx, fy]

        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        normUnPolData = raytrace.CreateNormUnpol(N_rays_inside + 1, constants.RaysType_Real, surface)

        # Add the Chief Ray
        normUnPolData.AddRay(wave_idx, hx, hy, 0.0, 0.0, constants.OPDMode_None)

        for (p_x, p_y) in zip(px, py):
            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(N_rays_inside + 1):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                XY[j, i, 0] = output[4]
                XY[j, i, 1] = output[5]

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        x, y = XY[j, :, 0], XY[j, :, 1]
        chief_x, chief_y = x[0], y[0]           # We added the Chief Ray first to the BatchTrace
        cent_x, cent_y = np.mean(x), np.mean(y)

        # Select what point will be used as Reference for the FWHM
        if reference == "Centroid":
            ref_x, ref_y = cent_x, cent_y
        elif reference == "ChiefRay":
            ref_x, ref_y = chief_x, chief_y
        else:
            raise ValueError("reference should be 'ChiefRay' or 'Centroid'")

        # Add the Reference Point to the Focal Plane Raytrace results
        foc_xy[j, :] = [ref_x, ref_y]

        # # Calculate the contours
        # x, y = XY[j, :, 0], XY[j, :, 1]
        #
        # mx, my = np.mean(x), np.mean(y)
        # dx, dy = 1000*(x - mx), 1000*(y - my)       # in microns
        # ax = axes[j]
        # ax.scatter(dx, dy, s=3, color=color)
        # # ax.set_xlabel(r'X [mm]')
        # # ax.set_ylabel(r'X [mm]')
        # ax.set_xlim([-25, 25])
        # ax.set_ylim([-25, 25])
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_aspect('equal')
        # ax.xaxis.set_visible('False')
        # ax.yaxis.set_visible('False')

        return [XY, obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, N_rays=40, reference='ChiefRay'):
        """

        """

        # We need to know how many rays are inside the pupil
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx ** 2 + pyy ** 2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['XY', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(1, N_rays_inside + 1, 2), (1, 2), (1, 2)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        for zemax_file in file_list:

            list_results = self.run_analysis(analysis_function=self.analysis_functions_spots,
                                             files_dir=files_dir,zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, N_rays=N_rays, reference=reference)

            return list_results



class RMSSpotAnalysis(AnalysisGeneric):

    @staticmethod
    def analysis_function_rms_spot(system, wave_idx, config, surface, field,
                                   reference='Centroid', mode='RMS', ray_density=10):

        N_fields = len(field)
        spots = np.zeros(N_fields)

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Create Spot Diagram analysis
        spot = system.Analyses.New_Analysis(constants.AnalysisIDM_StandardSpot)
        spot_setting = spot.GetSettings()
        baseSetting = CastTo(spot_setting, 'IAS_Spot')
        baseSetting.Field.SetFieldNumber(0)                     # 0 means All Field Points
        baseSetting.Wavelength.SetWavelengthNumber(wave_idx)    # set the Wavelength
        baseSetting.RayDensity = ray_density                    # set the Ray Density

        if reference == 'Centroid':
            baseSetting.ReferTo = constants.ReferTo_Centroid
        elif reference == 'ChiefRay':
            baseSetting.ReferTo = constants.ReferTo_ChiefRay

        base = CastTo(spot, 'IA_')
        base.ApplyAndWaitForCompletion()

        # Get Results
        spot_results = base.GetResults()
        for i, f in enumerate(field):
            if mode == 'RMS':
                spots[i] = spot_results.SpotData.GetRMSSpotSizeFor(f, wave_idx)
            elif mode == 'GEO':
                spots[i] = spot_results.SpotData.GetGeoSpotSizeFor(f, wave_idx)

        spot.Close()

        return [spots]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, field=(1,),
                        reference='Centroid', mode='RMS', ray_density=6, save_txt=False):

        N_fields = len(field)
        results_names = ['RMS_SPOT']
        results_shapes = [(N_fields,)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        list_spot = []
        list_waves = []
        for zemax_file in file_list:

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_spot,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, field=field, reference=reference, mode=mode, ray_density=ray_density)

            rms_spot, wavelengths = list_results
            list_spot.append(rms_spot)
            list_waves.append(wavelengths)

            # mean_field_spot = np.mean(spots, axis=0)
            # plt.figure()
            # plt.imshow(mean_field_spot, origin='lower', cmap='Reds')
            # plt.colorbar()
            # plt.ylabel(r'Slice #')
            # plt.xlabel(r'Wavelength [$\mu$m]')
            #
            # mean_spot_wave = np.mean(spots, axis=(0, 1))
            #
            # plt.figure()
            # plt.plot(wavelengths, mean_spot_wave)
            # plt.xlabel(r'Wavelength [$\mu$m]')
            # plt.show()
        self.plot_results(list_spot, list_waves, file_list, results_path, gratings=files_opt['grating'])

        return list_spot

    def plot_results(self, list_spot, list_waves, file_list, results_path, gratings):


        # WARNING: this will only work if you have ran the code for a single SCALE and IFU
        file_no_spec = file_list[0].split('_SPEC')[0]

        file_dir = os.path.join(results_path, file_no_spec)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        spot_dir = os.path.join(file_dir, 'RMS_SPOT_WAVELENGTH')
        if not os.path.exists(spot_dir):
            os.mkdir(spot_dir)           # If not, create the directory to store results

        N_gratings = len(gratings)
        colors = cm.jet(np.linspace(0, 1, N_gratings))
        plt.figure()
        for k in range(N_gratings):
            spots, waves = list_spot[k], list_waves[k]
            mean_spots = np.mean(spots, axis=(1, 2))
            plt.plot(waves, mean_spots, color=colors[k], label=gratings[k])
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'RMS Spot [$\mu$m]')
        plt.ylim(bottom=0)
        plt.legend(title=r'Grating', loc=3, ncol=N_gratings//3)
        figname = file_no_spec + '_RMS_SPOT_WAVELENGTH'
        plt.title(figname)
        plt.savefig(os.path.join(spot_dir, figname))

        return

### This are the relevant analyses for the E2E model:
#   - Ensquared Energy
#   - FWHM_PSF
#   - Raytrace
#   - RMS WFE

class EnsquaredEnergyAnalysis(AnalysisGeneric):
    """
    Geometric Ensquared Energy analysis for the HARMONI E2E files
    For the calculation of EE we use a custom approach based on the
    specific HARMONI requirements, which only cares about SPATIAL direction
    and ignores the SPECTRAL direction
    """

    @staticmethod
    def analysis_function_ensquared_energy(system, wave_idx, config, surface, slicer_surface, px, py, alpha=2):
        """
        Analysis function to calculate the Geometric Ensquared Energy

        We return a list containing the following data:
            - Ensquared Energy value
            - Object Coordinates (field point at the centre of the slice)
            - Raytrace results both at the Image Slicer and Detector plane in the form of centroid coordinates

        :param system: the Zemax system as given by the API
        :param wave_idx: Zemax wavelength index
        :param config: Zemax configuration / slice
        :param surface: Not used, we always use the last surface (Detector Plane)
        :param slicer_surface: Zemax surface number for the Image Slicer
        :param px: pupil X coordinates of the rays to be traced
        :param py: pupil Y coordinates of the rays to be traced
        :param alpha: how many detector pixels and slices to consider
        :return:
        """

        det_pix = 15e-3  # Detector pixel 15 microns

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Detector is always the last surface
        detector_surface = system.LDE.NumberOfSurfaces - 1

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # Use the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / r_max, fy / r_max  # Normalized field coordinates (hx, hy)
        obj_xy = np.array([fx, fy])

        N_rays = px.shape[0]

        # We need to trace rays up to the Image Slicer and then up to the Detector
        slicer_xy = np.empty((N_rays, 2))
        detector_xy = np.empty((N_rays, 2))

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace_slicer = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace_slicer.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)
        for (p_x, p_y) in zip(px, py):      # Add the ray to the RayTrace
            rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace_slicer, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        for i in range(N_rays):     # Get Raytrace results at the Image Slicer
            output = rays_slicer.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                slicer_xy[i, 0] = output[4]
                slicer_xy[i, 1] = output[5]
                checksum_slicer += 1
        if checksum_slicer < N_rays:
            raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()
        # CastTo(raytrace_slicer, 'ISystemTool').Close()

        # Count how many rays fall inside a +- 1 mm window in Y, wrt the centroid
        scx, scy = np.mean(slicer_xy[:, 0]), np.mean(slicer_xy[:, 1])
        below_slicer = slicer_xy[:, 1] < scy + 1.0 * alpha / 2
        above_slicer = slicer_xy[:, 1] > scy - 1.0 * alpha / 2
        inside_slicer = (np.logical_and(below_slicer, above_slicer))
        total_slicer = np.sum(inside_slicer)
        index_valid_slicer = np.argwhere(inside_slicer == True)
        # print(index_slicer)

        # (2) Run the raytrace up to the DETECTOR
        # For speed, we re-use the same Raytrace, just define new rays!
        # raytrace_det = system.Tools.OpenBatchRayTrace()
        rays_detector = raytrace_slicer.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)
        for (p_x, p_y) in zip(px, py):
            rays_detector.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)
        CastTo(raytrace_slicer, 'ISystemTool').RunAndWaitForCompletion()

        rays_detector.StartReadingResults()
        checksum_detector = 0
        index_valid_detector = []       # Valid means they make it to the detector even if vignetted at the Slicer
        vignetted = []
        index_vignetted = []
        for i in range(N_rays):
            output = rays_detector.ReadNextResult()
            if output[2] == 0 and output[3] == 0:       # ErrorCode & VignetteCode
                detector_xy[i, 0] = output[4]
                detector_xy[i, 1] = output[5]
                checksum_detector += 1
                index_valid_detector.append(i)
            elif output[2] == 0 and output[3] != 0:
                # Some rays are vignetted
                error_code = output[2]
                vignette_code = output[3]
                vignetted.append([output[4],  output[5]])
                detector_xy[i, 0] = output[4]
                detector_xy[i, 1] = output[5]
                checksum_detector += 1
                index_valid_detector.append(i)
                index_vignetted.append(i)

                # print("\nError Code: %d | Vignette Code: %d" % (error_code, vignette_code))
        vignetted = np.array(vignetted)
        # if checksum_detector < N_rays:
        #     print('Some rays were lost after the Image Slicer: ', checksum_detector)
        #

        rays_detector.ClearData()
        CastTo(raytrace_slicer, 'ISystemTool').Close()

        # (3) Calculate the ENSQUARED ENERGY

        # We only count the rays that where inside the slicer to begin with and the ones that make it to the detector
        valid_both = []
        for i in range(N_rays):
            if i in index_valid_slicer and i in index_valid_detector:
                valid_both.append(i)

        valid_det_x = detector_xy[:, 0][valid_both]
        valid_det_y = detector_xy[:, 1][valid_both]

        # Now, out of the VALID rays, we calculate which detector rays fall inside a 2x pixel box along X
        dcx = np.mean(valid_det_x)      # Detector Centroid X
        dcy = np.mean(valid_det_y)

        left_detector = valid_det_x < dcx + det_pix * alpha / 2
        right_detector = valid_det_x > dcx - det_pix * alpha / 2
        inside_detector = (np.logical_and(left_detector, right_detector))
        total_detector = np.sum(inside_detector)
        EE = total_detector / N_rays

        # Add detector cross-talk 98%
        EE *= 0.98

        if EE > 1.00:
            print(config, wave_idx)
            raise ValueError("Ensquared Energy is larger than 1.0")

        #
        # print("\nRays Traced: ", N_rays)
        # print("Make it to the Image Slicer: %d / %d" % (checksum_slicer, N_rays))
        # print("Inside 2 slice box: %d / %d" % (total_slicer, N_rays))
        # print("Make it to the Detector: %d / %d" % (checksum_detector, N_rays))
        # print("Inside 2 pixel box: %d / %d" % (total_detector, checksum_detector))
        # print("Ensquared Energy: %.3f" % EE)
        #

        # if config in [1, 2, 3, 4, 5, 6] and wave_idx in [1, 2, 3, 4, 5]:
        #
        #     print("\nTracing %d rays to calculate Ensquared Energy" % (N_rays))
        #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #
        #     ax1.scatter(px[valid_both], py[valid_both], s=5, color='lime')
        #     ax1.scatter(px[index_vignetted], py[index_vignetted], s=5, color='red')
        #     ax1.set_xlabel(r'$P_x$')
        #     ax1.set_ylabel(r'$P_y$')
        #     ax1.set_xlim([-1, 1])
        #     ax1.set_ylim([-1, 1])
        #     ax1.set_title(r'Pupil Plane | %s rays' % N_rays)
        #     ax1.set_aspect('equal')
        #
        #     sx, sy = slicer_xy[:, 0], slicer_xy[:, 1]
        #     scx, scy = np.mean(sx), np.mean(sy)
        #     ax2.scatter(sx[valid_both], sy[valid_both], s=3, color='blue')
        #     ax2.scatter(sx[index_vignetted], sy[index_vignetted], s=3, color='red')
        #     # ax2.scatter(scx, scy, s=8, color='red')
        #     ax2.axhline(y=scy + 1.0, color='black', linestyle='--')
        #     ax2.axhline(y=scy - 1.0, color='black', linestyle='--')
        #     ax2.set_xlabel(r'Slicer X [mm]')
        #     ax2.set_ylabel(r'Slicer Y [mm]')
        #     ax2.set_xlim([scx - 2, scx + 2])
        #     ax2.set_ylim([scy - 2, scy + 2])
        #     ax2.set_title(r'Image Slicer | $\pm$1.0 mm wrt Centroid')
        #     ax2.set_aspect('equal')
        #
        #     ax3.scatter(valid_det_x, valid_det_y, s=3, color='green')
        #     ax3.scatter(valid_det_x[index_vignetted], valid_det_y[index_vignetted], s=3, color='red')
        #     # ax3.scatter(vignetted[:, 0], vignetted[:, 1], s=3, color='orange')
        #     ax3.axvline(x=dcx + det_pix, color='black', linestyle='--')
        #     ax3.axvline(x=dcx - det_pix, color='black', linestyle='--')
        #     ax3.set_xlabel(r'Detector X [mm]')
        #     ax3.set_ylabel(r'Detector Y [mm]')
        #     ax3.set_xlim([dcx - 2*det_pix, dcx + 2*det_pix])
        #     ax3.set_ylim([dcy - 2*det_pix, dcy + 2*det_pix])
        #     ax3.set_title(r'Detector Plane | $\pm$15 $\mu$m wrt Centroid')
        #     ax3.set_aspect('equal')
        # plt.show()

        return [EE, obj_xy, [scx, scy], [dcx, dcy]]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, N_rays=500, alpha=2):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_ensquared_energy


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param N_rays: how many rays to use to calculate the Geometric Ensquared Energy

        :return:
        """

        # We want the result to produce as output: Ensquared Energy, Object coords, Image Slicer and Detector centroids
        results_names = ['EE', 'OBJ_XY', 'SLI_XY', 'DET_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(1,), (2,), (2,), (2,)]

        # read the file options
        file_list, file_settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        list_results = []
        for zemax_file, settings in zip(file_list, file_settings):

            system = settings['system']
            spaxel_scale = settings['scale']
            ifu = settings['ifu']
            # We need to know what is the Surface Number for the Image Slicer in the Zemax file
            slicer_surface = focal_planes[system][spaxel_scale][ifu]['IS']

            # Generate a set of random pupil rays
            px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
            print("Using %d rays" % N_rays)

            results = self.run_analysis(analysis_function=self.analysis_function_ensquared_energy,
                                        files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                        results_shapes=results_shapes, results_names=results_names,
                                        wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                        slicer_surface=slicer_surface, px=px, py=py, alpha=alpha)

            list_results.append(results)

        return list_results


class FWHM_PSF_Analysis(AnalysisGeneric):
    """
    Calculate the Geometric FWHM of the PSF
    by computing the spot diagrams, getting the contour plots
    and estimating the diameter of the 50% contour line
    """

    def calculate_fwhm(self, surface, xy_data, PSF_window, N_points, spaxel_scale, wavelength):

        start = time()
        # Calculate the Geometric PSF
        x, y = xy_data[:, 0], xy_data[:, 1]
        cent_x, cent_y = np.mean(x), np.mean(y)

        std_x, std_y = np.std(x), np.std(y)
        bandwidth = min(std_x, std_y)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5*bandwidth).fit(xy_data)

        # define a grid to compute the PSF
        xmin, xmax = cent_x - PSF_window/2/1000, cent_x + PSF_window/2/1000
        ymin, ymax = cent_y - PSF_window/2/1000, cent_y + PSF_window/2/1000
        x_grid = np.linspace(xmin, xmax, N_points)
        y_grid = np.linspace(ymin, ymax, N_points)
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        xy_grid = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T
        log_scores = kde.score_samples(xy_grid)

        psf_geo = np.exp(log_scores)
        psf_geo /= np.max(psf_geo)
        psf_geo = psf_geo.reshape(xx_grid.shape)

        # time_geopsf = time() - start

        # plt.figure()
        # img = plt.imshow(psf_geo, extent=[xmin, xmax, ymin, ymax], cmap='bwr', origin='lower')
        # plt.scatter(x, y, s=2, color='black')
        # plt.xlabel(r'X [mm]')
        # plt.ylabel(r'Y [mm]')
        # plt.title('PSF | %d pixels | %.1f $\mu$m / pixel' % (N_points, PSF_window / N_points))

        # start = time()

        psf_diffr = diffraction.add_diffraction(surface=surface, psf_geo=psf_geo, PSF_window=PSF_window,
                                                scale_mas=spaxel_scale, wavelength=wavelength)
        # time_diffpsf = time() - start
        #
        # start = time()

        # guesses = {'IS': {4.0: }}

        # Fit the PSF to a 2D Gaussian
        fwhm_x, fwhm_y, theta = diffraction.fit_psf_to_gaussian(xx=xx_grid, yy=yy_grid, psf_data=psf_diffr, x0=cent_x, y0=cent_y)

        # time_gauss = time() - start
        #
        # print('FWHM time: %.3f sec for GeoPSF estimate:' % time_geopsf)
        # print('FWHM time: %.3f sec for DiffPSF convolution:' % time_diffpsf)
        # print('FWHM time: %.3f sec for Gaussian fit:' % time_gauss)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # img1 = ax1.imshow(psf_geo, extent=[xmin, xmax, ymin, ymax], cmap='bwr', origin='lower')
        # plt.colorbar(img1, ax=ax1, orientation='horizontal')
        # ax1.set_xlabel(r'X [mm]')
        # ax1.set_ylabel(r'Y [mm]')
        # ax1.set_title(r'Geometric PSF estimate | Surface: %s' % surface)
        #
        # img2 = ax2.imshow(psf_diffr, extent=[xmin, xmax, ymin, ymax], cmap='bwr', origin='lower')
        # plt.colorbar(img2, ax=ax2, orientation='horizontal')
        # ax2.set_xlabel(r'X [mm]')
        # ax2.set_ylabel(r'Y [mm]')
        # if surface == 'DET':
        #     ax2.set_title(r'Diffr. PSF | %.3f microns | %.1f mas | FWHM_x: %.1f $\mu$m' % (wavelength, spaxel_scale, fwhm_x))
        # elif surface == 'IS':
        #     ax2.set_title(r'Diffr. PSF | %.3f microns | %.1f mas | FWHM_y: %.1f $\mu$m' % (wavelength, spaxel_scale, fwhm_y))

        return fwhm_x, fwhm_y

    def analysis_function_geofwhm_psf(self, system, wave_idx, config, surface, slicer_surface, px, py, spaxel_scale,
                                      N_points, PSF_window):


        # print("Config: ", config)
        # Set Current Configuration
        # start = time()
        system.MCE.SetCurrentConfiguration(config)

        wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields

        N_rays = px.shape[0]

        XY = np.empty((N_rays, 2))
        FWHM = np.zeros(2)          # [mean_radius, m_radiusX, m_radiusY]
        obj_xy = np.zeros(2)        # (fx, fy) coordinates for the chief ray
        foc_xy = np.empty(2)        # raytrace results of the Centroid
        N = int(PSF_window / 15)
        PSF_cube = np.empty((N, N))

        if config == 1:
            print("\nTracing %d rays to calculate FWHM PSF" % (N_rays))

        # (0) Define the Field coordinates for the Chief Ray, at the centre of the Slice
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / X_MAX, fy / Y_MAX      # Normalized field coordinates (hx, hy)
        obj_xy[:] = [fx, fy]

        # We need to trace rays up to the Image Slicer and then up to the Detector
        slicer_xy = np.empty((N_rays, 2))
        detector_xy = np.empty((N_rays, 2))

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)
        for (p_x, p_y) in zip(px, py):      # Add the ray to the RayTrace
            rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        valid_index_slicer = []
        for i in range(N_rays):     # Get Raytrace results at the Image Slicer
            output = rays_slicer.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                slicer_xy[i, 0] = output[4]
                slicer_xy[i, 1] = output[5]
                valid_index_slicer.append(i)
                checksum_slicer += 1
        # if checksum_slicer < N_rays:
        #     print(checksum_slicer)
            # raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()

        # time_ray_slicer = time() - start
        # start = time()

        # (2) Run the raytrace up to the DETECTOR
        # Detector is always the last surface
        detector_surface = system.LDE.NumberOfSurfaces - 1
        # For speed, we re-use the same Raytrace, just define new rays!
        # raytrace_det = system.Tools.OpenBatchRayTrace()
        rays_detector = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)
        for (p_x, p_y) in zip(px, py):
            rays_detector.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_detector.StartReadingResults()
        checksum_detector = 0
        valid_index_detector = []       # Valid means they make it to the detector even if vignetted at the Slicer
        for i in range(N_rays):
            output = rays_detector.ReadNextResult()
            if output[2] == 0:      # ErrorCode & VignetteCode
                detector_xy[i, 0] = output[4]
                detector_xy[i, 1] = output[5]
                checksum_detector += 1
                valid_index_detector.append(i)
            elif output[2] == 0 and output[3] != 0:
                vignette_code = output[3]
                vignette_surf = system.LDE.GetSurfaceAt(vignette_code).Comment
                print("Vignetting at surface #%d: %s" % (vignette_code, vignette_surf))

        # if checksum_detector < N_rays:
        #     print(checksum_detector)
        #     print('Some rays were lost at the Detector')
        # print("\nValid Rays: Slicer %d | Detector %d" % (checksum_slicer, checksum_detector))

        rays_detector.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # time_ray_det = time() - start
        # print("Raytrace time: %.3f sec for Image Slicer | %.3f sec for Detector" % (time_ray_slicer, time_ray_det))



        # ============================================================================================================ #
        #                                            PSF Calculations
        # ============================================================================================================ #

        xy_data_slicer = np.stack([slicer_xy[:, 0][valid_index_slicer], slicer_xy[:, 1][valid_index_slicer]]).T
        xy_data_detector = np.stack([detector_xy[:, 0][valid_index_detector], detector_xy[:, 1][valid_index_detector]]).T
        start = time()
        fwhm_x_slicer, fwhm_y_slicer = self.calculate_fwhm(surface='IS', xy_data=xy_data_slicer, PSF_window=1500,
                                                           N_points=N_points, spaxel_scale=spaxel_scale, wavelength=wavelength)
        time_fwhm_slicer = time() - start
        start = time()
        fwhm_x_det, fwhm_y_det = self.calculate_fwhm(surface='DET', xy_data=xy_data_detector, PSF_window=PSF_window,
                                                     N_points=N_points, spaxel_scale=spaxel_scale, wavelength=wavelength)
        time_fwhm_det = time() - start
        # print("FWHM time: %.3f sec for Image Slicer | %.3f sec for Detector" % (time_fwhm_slicer, time_fwhm_det))


        # print("\nWave #%d, config #%d" % (wave_idx, config))
        # print("Fx: %.4f Fy: %.4f" % (fx, fy))
        # print("Hx: %.4f Hy: %.4f" % (hx, hy))
        #
        if config == 1:
            print("FWHM in X [Detector]: %.1f microns | in Y [Image Slicer]: %.2f microns " % (fwhm_x_det, fwhm_y_slicer))

        FWHM[:] = [fwhm_x_det, fwhm_y_slicer]        # Store the results

        plt.show()

        return [FWHM, PSF_cube, obj_xy, foc_xy]


    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, N_rays=500, N_points=50, PSF_window=50):
        """
        Loop over the Zemax files and calculate the FWHM

        Relevant parameters:
            - N_rays: How many pupil rays rays to trace to estimate the Geometric PSF
            - N_points: how many points 'pixels' to crop the PSF to before fitting to Gaussian
            - PSF_window: how many microns does that crop window span. important to ensure a sampling of 1 micron / point

        :param files_dir:
        :param files_opt:
        :param results_path:
        :param wavelength_idx:
        :param configuration_idx:
        :param surface:
        :param N_rays:
        :param N_points:
        :param PSF_window:
        :return:
        """


        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['GEO_FWHM', 'PSF_CUBE', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(2,), (PSF_window//15, PSF_window//15), (2,), (2,)]

        px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
        print("Using %d rays" % N_rays)

        metadata = {}
        metadata['N_rays Pupil'] = N_rays
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx
        metadata['N_points PSF'] = N_points
        metadata['PSF window [microns]'] = PSF_window

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            # Clear the Cache for the Airy Pattern function
            airy_and_slicer.cache_clear()

            mas_dict = {'4x4': 4.0, '10x10': 10.0, '20x20': 20.0, '60x30': 60.0}
            spaxel_scale = mas_dict[settings['scale']]

            # We need to know what is the Surface Number for the Image Slicer in the Zemax file
            slicer_surface = focal_planes[settings['system']][settings['scale']][settings['ifu']]['IS']

            list_results = self.run_analysis(analysis_function=self.analysis_function_geofwhm_psf,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, slicer_surface=slicer_surface, px=px, py=py,
                                             spaxel_scale=spaxel_scale, N_points=N_points, PSF_window=PSF_window)
            results.append(list_results)


            geo_fwhm, psf_cube, obj_xy, foc_xy, wavelengths = list_results

            settings['surface'] = 'IMG' if surface is None else surface

            file_name = zemax_file.split('.')[0]
            # self.save_hdf5(analysis_name='FWHM_PSF', analysis_metadata=metadata, list_results=list_results,
            #                results_names=results_names, file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


class RaytraceAnalysis(AnalysisGeneric):
    """
    Raytrace analysis at any surface
    """

    @staticmethod
    def analysis_function_raytrace(system, wave_idx, config, rays_per_slice, surface):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial
        N_fields = sysField.NumberOfFields

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])
        # TODO: make this robust
        # Watch Out! This assumes that Field #1 is at the edge of the slice and that Field #3 is at the other edge
        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(3).X, sysField.GetField(3).Y

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / r_max, fx_max / r_max
        hy_min, hy_max = fy_min / r_max, fy_max / r_max

        hx = np.linspace(hx_min, hx_max, rays_per_slice)
        hy = np.linspace(hy_min, hy_max, rays_per_slice)

        # The Field coordinates for the Object
        obj_xy = r_max * np.array([hx, hy]).T
        foc_xy = np.empty((rays_per_slice, 2))
        global_xy = np.empty((rays_per_slice, 2))
        local_xyz = np.empty((rays_per_slice, 3))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(rays_per_slice, constants.RaysType_Real, surface)

        # Loop over all Rays in the Slice
        for i, (h_x, h_y) in enumerate(zip(hx, hy)):
            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(rays_per_slice):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                local_xyz[i, 0] = output[4]
                local_xyz[i, 1] = output[5]
                local_xyz[i, 2] = output[6]

                # Local Focal X Y
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # Get the transformation from Local to Global Coordinates
        global_mat = system.LDE.GetGlobalMatrix(surface)
        R11, R12, R13, R21, R22, R23, R31, R32, R33, X0, Y0, Z0 = global_mat[1:]
        global_matrix = np.array([[R11, R12, R13],
                                  [R21, R22, R23],
                                  [R31, R32, R33]])
        offset = np.array([X0, Y0, Z0])

        # Transform from Local to Global and only save X and Y
        global_xyz = (np.dot(global_matrix, local_xyz.T)).T + offset
        global_xy = global_xyz[:, :2]

        return [obj_xy, foc_xy, global_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, rays_per_slice=3):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_raytrace


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param rays_per_slice: number rays to trace per slice
        :return:
        """

        results_names = ['OBJ_XY', 'FOC_XY', 'GLOB_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(rays_per_slice, 2), (rays_per_slice, 2), (rays_per_slice, 2)]

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_raytrace,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, rays_per_slice=rays_per_slice)

            obj_xy, foc_xy, global_xy, wavelengths = list_results

            # For each file we save the list of results
            # we could use that in a Monte Carlo analysis for example
            results.append(list_results)

        return results



class RMS_WFE_Analysis(AnalysisGeneric):
    """
    Example of how to use Class Inheritance with the AnalysisGeneric object
    to calculate the RMS WFE error

    In this case the "analysis_function" is "analysis_function_rms_wfe"
    """

    @staticmethod
    def analysis_function_rms_wfe(system, wave_idx, config, spaxels_per_slice, surface):
        """
        Analysis function that calculates the RMS Wavefront Error [nm]
        for an Optical System, at a given Wavelength and Configuration / Slice
        at a specified Surface

        This function will return a list of arrays:
            - RMS_WFE: array of size [N_spaxels_per_slice] containing the RMS WFE for each spaxel of the given config
            - OBJ_XY: array of size [N_spaxels_per_slice, 2] containing the (XY) coordinates at the Object plane
            - FOC_XY: same as OBJ_XY but at the Focal plane defined by Surface

        :param system: Optical System that the ZOS API has loaded
        :param wave_idx: Zemax Wavelength Number to be used in the RMS calculation
        :param config: Zemax Configuration Number, the current Slice
        :param spaxels_per_slice: how many spaxels per slice to use for sampling the slice
        :param surface: Zemax surface number at which to evaluate the RMS WFE
        :param rings: Zemax parameter for the RMS WFE calculation
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        # check that the field is normalized correctly
        # if sysField.Normalization != constants.FieldNormalizationType_Radial:
        #     sysField.Normalization = constants.FieldNormalizationType_Radial

        N_fields = sysField.NumberOfFields
        wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength

        # # Loop over the fields to get the Normalization Radius
        # r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
        #                         sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])
        # print(r_max)

        # TODO: make this robust
        # Watch Out! This assumes that Field #1 is at the edge of the slice and that Field #3 is at the other edge
        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(3).X, sysField.GetField(3).Y

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(3)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(3)])

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / X_MAX, fx_max / X_MAX
        hy_min, hy_max = fy_min / Y_MAX, fy_max / Y_MAX

        # print("h_x : (%.3f, %.3f)" % (hx_min, hx_max))
        # print("h_y : (%.3f, %.3f)" % (hy_min, hy_max))

        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)

        # The Field coordinates for the Object
        obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        RMS_WFE = np.empty(spaxels_per_slice)
        foc_xy = np.empty((spaxels_per_slice, 2))
        global_xy = np.empty((spaxels_per_slice, 2))
        local_xyz = np.empty((spaxels_per_slice, 3))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(spaxels_per_slice, constants.RaysType_Real, surface)

        theMFE = system.MFE
        nops = theMFE.NumberOfOperands
        theMFE.RemoveOperandsAt(1, nops)
        # build merit function
        op = theMFE.GetOperandAt(1)
        op.ChangeType(constants.MeritOperandType_CONF)
        op.GetOperandCell(constants.MeritColumn_Param1).Value = config

        # Pupil Sampling
        samp = 4
        wfe_op = constants.MeritOperandType_RWRE

        # Loop over all Spaxels in the Slice
        for i, (h_x, h_y) in enumerate(zip(hx, hy)):

            # operand = constants.MeritOperandType_RWRE
            # rms = system.MFE.GetOperandValue(operand, surface, wave_idx, h_x, h_y, 0.0, 0.0, 0.0, 0.0)
            # RMS_WFE[i] = wavelength * 1e3 * rms         # We assume the Wavelength comes in Microns

            op = theMFE.AddOperand()
            op.ChangeType(wfe_op)
            op.GetOperandCell(constants.MeritColumn_Param1).Value = int(samp)
            op.GetOperandCell(constants.MeritColumn_Param2).Value = int(wave_idx)
            op.GetOperandCell(constants.MeritColumn_Param3).Value = float(h_x)
            op.GetOperandCell(constants.MeritColumn_Param4).Value = float(h_y)
            op.GetOperandCell(constants.MeritColumn_Weight).Value = 0

            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # update merit function
        theMFE.CalculateMeritFunction()
        # retrieve value of each RWRE operand
        # theMCE.SetCurrentConfiguration(i)
        system.MCE.SetCurrentConfiguration(config)
        # print("N operands:", nops)
        for irow in range(2, theMFE.NumberOfOperands + 1):
            op = theMFE.GetOperandAt(irow)
            rms = op.Value
            # print(irow)
            # print("RMS: %.2f nm" % (wavelength * 1e3 * rms))
            RMS_WFE[irow - 2] = wavelength * 1e3 * rms  # We assume the Wavelength comes in Microns

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(spaxels_per_slice):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0:
                local_xyz[i, 0] = output[4]
                local_xyz[i, 1] = output[5]
                local_xyz[i, 2] = output[6]

                # Local Focal X Y
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]

                # print("\nConfiguration #%d" % config)
                # print("fx: %.5f fy: %.5f" % (X_MAX * hx[i], Y_MAX * hy[i]))
                # print("hx: %.5f hy: %.5f" % (hx[i], hy[i]))
                # x_det, y_det = output[4], output[5]
                # print("x_det: %.3f y_det: %.3f" % (x_det, y_det))

            # elif output[2] == 0 and output[3] != 0:
            #     vignetting = output[3]
            #     surf_name = system.LDE.GetSurfaceAt(vignetting).Comment
            #     print("\nWARNING: Vignetting at Surface #%d, Name %s" % (vignetting, surf_name))
            #     print("Wavelength: %.3f | Configuration #%d" % (wavelength, config))
            #     print("Field #%d | fx: %.5f fy: %.5f" % (i + 1, X_MAX * hx[i], Y_MAX * hy[i]))
            #     print("hx: %.5f hy: %.5f" % (hx[i], hy[i]))
            #     x_det, y_det = output[4], output[5]
            #     print("x_det: %.3f y_det: %.3f" % (x_det, y_det))
            #
            #     # Add the local coordinates despite the vignetting
            #     local_xyz[i, 0] = output[4]
            #     local_xyz[i, 1] = output[5]
            #     local_xyz[i, 2] = output[6]


        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # # Get the transformation from Local to Global Coordinates
        # global_mat = system.LDE.GetGlobalMatrix(surface)
        # R11, R12, R13, R21, R22, R23, R31, R32, R33, X0, Y0, Z0 = global_mat[1:]
        # global_matrix = np.array([[R11, R12, R13],
        #                           [R21, R22, R23],
        #                           [R31, R32, R33]])
        # offset = np.array([X0, Y0, Z0])
        #
        # # Transform from Local to Global and only save X and Y
        # global_xyz = (np.dot(global_matrix, local_xyz.T)).T + offset
        # global_xy = global_xyz[:, :2]

        return [RMS_WFE, obj_xy, foc_xy, global_xy]

    # The following functions depend on how you want to post-process your analysis results
    # For example, "loop_over_files" automatically runs the RMS WFE analysis across all Zemax files
    # and uses "flip_rms" to reorder the Configurations -> Slice #,
    # and "plot_RMS_WFE_maps" to save the figures

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=51):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_rms_wfe


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param spaxels_per_slice: number spaxels to sample each slice with
        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['RMS_WFE', 'OBJ_XY', 'FOC_XY', 'GLOB_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(spaxels_per_slice,), (spaxels_per_slice, 2), (spaxels_per_slice, 2), (spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx


        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_wfe,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice)

            results.append(list_results)
            rms_wfe, obj_xy, foc_xy, global_xy, wavelengths = list_results

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            results_dir = os.path.join(results_path, file_name)
            settings['surface'] = 'IMG' if surface is None else surface

            self.save_hdf5(analysis_name='RMS_WFE', analysis_metadata=metadata, list_results=list_results, results_names=results_names,
                           file_name=file_name, file_settings=settings, results_dir=results_path)


        return results

# TODO: probably don't need this anymore
# Define the Spaxel Sizes [um] for each Spaxel Scale at each Focal Plane
spaxel_sizes = {}
spaxel_sizes["60x30"] = {"FPRS": [97.656, 195.313],
                              "PO": [195.313, 390.625],
                              "IFU": [65, 130],
                              "SPEC": [15, 30]}
spaxel_sizes["20x20"] = {"FPRS": [65.104, 65.104],
                              "PO": [195.313, 390.625],
                              "IFU": [65, 130],
                              "SPEC": [15, 30]}
spaxel_sizes["10x10"] = {"FPRS": [35.552, 35.552],
                              "PO": [195.313, 390.625],
                              "IFU": [65, 130],
                              "SPEC": [15, 30]}
spaxel_sizes["4x4"] = {"FPRS": [13.021, 13.021],
                            "PO": [195.313, 390.625],
                            "IFU": [65, 130],
                            "SPEC": [15, 30]}

# {***} Rationale for the AnalysisFast {***}
# AnalysisGeneric works fine, but it's not very efficient / fast. It is based on looping through Wavelength index
# and Configuration and running a set of operations for just 1 wavelength / 1 config.
# At some point, we realized that it could be faster to only loop through configurations and run a set of operations
# for all available Wavelength indices. For example, calculating the RMS WFE, it's much faster to just add operands
# for all wavelengths and field points of a given Configuration, update the Merit Function and then read the operands

# For that reason, we created AnalysisFast, a faster version of AnalysisGeneric that only loops over Configurations
# a whole suite of inherited analysis classes is available. They are very similar to their AnalysisGeneric
# counterparts, except for the fact that AnalysisGeneric uses array shapes of [N_waves, N_configs, (shape)]
# while AnalysisFast uses shapes of [N_configs, (shape)], with (shape) typically starting with N_waves,
# so there's a "flip" one should be aware of


class AnalysisFast(object):
    """
    Faster version that doesn't look over the wavelengths
    """

    def __init__(self, zosapi, *args, **kwargs):
        """
        We basically need a ZOS API Standalone Application to
        open and close the files we want to analyze
        :param zosapi:
        :param args:
        :param kwargs:
        """

        self.zosapi = zosapi

    def run_analysis(self, analysis_function, results_shapes, results_names, files_dir, zemax_file, results_path,
                     wavelength_idx=None, configuration_idx=None, surface=None,
                     *args, **kwargs):

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(files_dir, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(files_dir, zemax_file), False)
        file_name = zemax_file.split(".")[0]  # Remove the ".zmx" suffix

        # Check if the results directory already exists
        results_dir = os.path.join(results_path, file_name)
        print("Results will be saved in: ", results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)  # If not, create the directory to store results

        # Get some info on the system
        system = self.zosapi.TheSystem  # The Optical System
        MCE = system.MCE  # Multi Configuration Editor
        LDE = system.LDE  # Lens Data Editor
        N_surfaces = LDE.NumberOfSurfaces
        N_configs = MCE.NumberOfConfigurations

        # print("\nSystem Data:")
        print("Number of Surfaces: ", N_surfaces)
        # print("Number of Configurations / Slices: ", N_configs)
        _wavelengths = get_wavelengths(system, info=False)
        N_waves = _wavelengths.shape[0]
        # field_points = get_fields(system, info=False)

        # Let's deal with the data format
        if wavelength_idx is None:  # We use all the available wavelengths
            wavelengths = _wavelengths
            wavelength_idx = np.arange(1, N_waves + 1)
        else:  # We have received a list of wavelength indices
            wavelengths = [_wavelengths[idx - 1] for idx in wavelength_idx]
            N_waves = len(wavelength_idx)

        if configuration_idx is None:
            configurations = np.arange(1, N_configs + 1)
        else:  # We have received a list of configurations
            configurations = configuration_idx
            N_configs = len(configuration_idx)
        N_slices = N_configs

        if surface is None:  # we go to the final image plane
            surface = system.LDE.NumberOfSurfaces - 1

        N_surfaces = system.LDE.NumberOfSurfaces
        if surface != (N_surfaces - 1):
            # The Surface is not the final Image Plane. We must force Zemax to IGNORE
            # all surfaces after our surface of interest so that the RWCE operands works
            surfaces_to_ignore = np.arange(surface + 1, N_surfaces)
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = True

            last_surface = system.LDE.GetSurfaceAt(surface)
            thickness = last_surface.Thickness
            print("Thickness: ", last_surface.Thickness)
            # image = system.LDE.GetSurfaceAt(N_surfaces - 1)
            # print("\nImage Name: ", image.Comment)

            # Temporarily set the Thickness to 0
            last_surface.Thickness = 0.0
            print("Thickness: ", last_surface.Thickness)

        # Double check we are using the right surface
        print("\nSurface Name: ", system.LDE.GetSurfaceAt(surface).Comment)

        # Customize the initilization of the results array so that we can vary the shapes
        # some analysis may be [spaxels_per_slice, N_slices, N_waves] such as an RMS WFE map
        # others might be [N_fields, N_rays, N_slices, N_waves] such as a Spot Diagram

        # print("\nDynamically creating Global Variables to store results")
        for _name, _shape in zip(results_names, results_shapes):
            # Notice the change with respect to AnalysisGeneric, here we have N_configs + (shape)
            globals()[_name] = np.empty((N_slices, ) + _shape)
            # print(globals()[_name].shape)

        # print("\nApplying 'analysis_function': ", analysis_function.__name__)
        print("At Surface #%d | Image Plane is #%d" % (surface, N_surfaces - 1))
        print("For Wavelength Numbers: ", wavelength_idx)
        print("For Configurations %d -> %d" % (configurations[0], configurations[-1]))

        start = time()
        print("\nAnalysis Started: [FAST MODE]")
        # We only loop through configurations, which is much faster!
        for j, config in enumerate(configurations):

            # print("Config #%d: " % (config))

            # Here is where the arbitrary "analysis_function" will be called for each Wavelength and Configuration
            results = analysis_function(system=system, wavelength_idx=wavelength_idx, config=config, surface=surface,
                                        *args, **kwargs)

            # Store the result values in the arrays:
            for _name, res in zip(results_names, results):
                globals()[_name][j] = res

        # Unignore the surfaces
        if surface != (N_surfaces - 1):
            last_surface = system.LDE.GetSurfaceAt(surface)
            last_surface.Thickness = thickness
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = False

        self.zosapi.CloseFile(save=False)
        analysis_time = time() - start
        print("\nAnalysis done in %.1f seconds" % analysis_time)
        time_per_wave = analysis_time / len(wavelengths)
        print("%.2f seconds per Wavelength" % time_per_wave)

        # local_keys = [key for key in globals().keys()]
        # print(local_keys)

        # We return a list containing the results, and the value of the wavelengths
        list_results = [globals()[_name] for _name in results_names]
        list_results.append(wavelengths)

        return list_results

    def save_hdf5(self, analysis_name, analysis_metadata, list_results, results_names, file_name, file_settings, results_dir):

        # First thing is to create a separate folder within the results directory for this analysis
        hdf5_dir = os.path.join(results_dir, 'HDF5')
        print("Analysis Results will be saved in folder: ", hdf5_dir)
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)           # If not, create the directory to store results

        hdf5_file = file_name + '.hdf5'
        # Check whether the file already exists
        if os.path.isfile(os.path.join(hdf5_dir, hdf5_file)):   # Overwrite it
            print("HDF5 file already exists. Adding analysis")
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'r+') as f:

                # Here we have 2 options: either we are adding another analysis type or we are adding the same type
                # but with different settings (such as at a different surface)
                file_keys = list(f.keys())
                if analysis_name in file_keys:
                    print("Analysis type already exists")
                    analysis_group = f[analysis_name]
                    # we need to know how many analyses of the same type already exist
                    subgroup_keys = list(analysis_group.keys())
                    subgroup_number = len(subgroup_keys)        # if [0, 1] already exist we call it '2'
                    subgroup = analysis_group.create_group(str(subgroup_number))
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

                else:       # It's a new analysis type

                    # (2) Create a Group for this analysis
                    analysis_group = f.create_group(analysis_name)
                    # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                    subgroup = analysis_group.create_group('0')
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

        else:       # File does not exist, we create it now
            print("Creating HDF5 file: ", hdf5_file)
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as f:

                # (1) Save Zemax Metadata
                zemax_metadata = f.create_group('Zemax Metadata')
                zemax_metadata.attrs['(1) Zemax File'] = file_name
                zemax_metadata.attrs['(2) System Mode'] = file_settings['system']
                zemax_metadata.attrs['(3) Spaxel Scale'] = file_settings['scale']
                zemax_metadata.attrs['(4) IFU'] = file_settings['ifu']
                zemax_metadata.attrs['(5) Grating'] = file_settings['grating']
                AO = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else 'NA'
                zemax_metadata.attrs['(6) AO Mode'] = AO

                # (2) Create a Group for this analysis
                analysis_group = f.create_group(analysis_name)

                # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                subgroup = analysis_group.create_group('0')
                # Save results datasets
                for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                    data = subgroup.create_dataset(array_name, data=array)
                # Save analysis metadata
                subgroup.attrs['E2E Python Git Hash'] = sha
                subgroup.attrs['Analysis Type'] = analysis_name
                date_created = datetime.datetime.now().strftime("%c")
                subgroup.attrs['Date'] = date_created
                subgroup.attrs['At Surface #'] = file_settings['surface']
                # Add whatever extra metadata we have:
                for key, value in analysis_metadata.items():
                    subgroup.attrs[key] = value

        return


class RMS_WFE_FastAnalysis(AnalysisFast):
    """
    [Fast Version] of the RMS WFE calculation
    """

    @staticmethod
    def analysis_function_rms_wfe(system, wavelength_idx, config, spaxels_per_slice, surface, pupil_sampling,
                                  slicer_surface=None):
        """
        Calculate the RMS WFE for a given Configuration, for an arbitrary number of Field Points and Wavelengths
        This is how we do it:

        (1) Get the Min and Max field points to normalize the coordinates
        (2) Sample the Slice from the min and max field points, using spaxels_per_slice points
        (3) Define the Merit Function. Loop over the Wavelengths, add 1 operand per field points to calculate RMS WFE
        (4) At the same time, add Chief Rays to the raytrace to get the detector coordinates
        (5) Read the results both for the Merit Function operands and the Raytrace results

        :param system: the Zemax system
        :param wavelength_idx: list of the Wavelength indices we want to use
        :param config: current configuration number
        :param spaxels_per_slice: how many points per slice to use to calculate the RMS WFE (typically 3)
        :param surface: the surface number at which we will calculate the RMS WFE
        :param pupil_sampling: how many points N to use on an N x N grid per pupil quadrant. RWRE operand parameter
        :return:
        """
        start0 = time()

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        ### [TEMPORARY PATCH]
        if slicer_surface is not None:

            # First of all, we need to find the Surface Number for the IMAGE SLICER
            N_surfaces = system.LDE.NumberOfSurfaces
            surface_names = {}      # A dictionary of surface number -> surface comment
            for k in np.arange(1, N_surfaces):
                surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
            # find the Slicer surface number
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Slicer Mirror')]
            slicer = system.LDE.GetSurfaceAt(slicer_num)

            # Read Current Aperture Settings
            apt_type = slicer.ApertureData.CurrentType
            # print("Aperture type: ", apt_type)
            if apt_type == 4:  # 4 is Rectangular aperture
                current_apt_sett = slicer.ApertureData.CurrentTypeSettings
                # print("Current Settings:")
                x0 = current_apt_sett._S_RectangularAperture.XHalfWidth
                y0 = current_apt_sett._S_RectangularAperture.YHalfWidth
                if y0 != 999:
                    # Change Settings
                    aperture_settings = slicer.ApertureData.CreateApertureTypeSettings(
                        constants.SurfaceApertureTypes_RectangularAperture)
                    aperture_settings._S_RectangularAperture.XHalfWidth = x0
                    aperture_settings._S_RectangularAperture.YHalfWidth = 999
                    slicer.ApertureData.ChangeApertureTypeSettings(aperture_settings)

                    current_apt_sett = slicer.ApertureData.CurrentTypeSettings
                    print("Changing aperture of surface: ", slicer.Comment)
                    print("New Settings:")
                    print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
                    print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)

        ### [TEMPORARY PATCH]

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)
        N_rays = N_waves * spaxels_per_slice

        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(N_fields).X, sysField.GetField(N_fields).Y

        # Note that this assumes Rectangular Normalization, the default in the E2E files.
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / X_MAX, fx_max / X_MAX
        hy_min, hy_max = fy_min / Y_MAX, fy_max / Y_MAX

        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)
        # hx = np.array([sysField.GetField(i + 1).X / X_MAX for i in range(N_fields)])
        # hy = np.array([sysField.GetField(i + 1).Y / Y_MAX for i in range(N_fields)])

        # The Field coordinates for the Object
        obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        RMS_WFE = np.empty((N_waves, spaxels_per_slice))
        foc_xy = np.empty((N_waves, spaxels_per_slice, 2))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, surface)

        # Start creating the Merit Function
        theMFE = system.MFE

        # Clear any operands left
        nops = theMFE.NumberOfOperands
        theMFE.RemoveOperandsAt(1, nops)

        # Build merit function
        # Set first operand to current configuration
        op = theMFE.GetOperandAt(1)
        op.ChangeType(constants.MeritOperandType_CONF)
        op.GetOperandCell(constants.MeritColumn_Param1).Value = config

        # Pupil Sampling
        samp = pupil_sampling
        wfe_op = constants.MeritOperandType_RWRE

        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                op = theMFE.AddOperand()
                op.ChangeType(wfe_op)
                op.GetOperandCell(constants.MeritColumn_Param1).Value = int(samp)
                op.GetOperandCell(constants.MeritColumn_Param2).Value = int(wave_idx)
                op.GetOperandCell(constants.MeritColumn_Param3).Value = float(h_x)
                op.GetOperandCell(constants.MeritColumn_Param4).Value = float(h_y)
                op.GetOperandCell(constants.MeritColumn_Weight).Value = 0

                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # time_1 = time() - start0
        # print("\nTime spent setting up MF and Raytrace: %.3f sec" % time_1)
        # start = time()

        # update merit function
        theMFE.CalculateMeritFunction()
        # time_mf = time() - start
        # print("Time spent updating MF: %.3f sec" % time_mf)

        # start = time()
        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # time_ray = time() - start
        # print("Time spent running Raytrace: %.3f sec" % time_ray)

        # start = time()
        normUnPolData.StartReadingResults()

        # Retrieve the results for the operands and raytrace
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                irow = 2 + i_wave * N_fields + j_field

                op = theMFE.GetOperandAt(irow)

                # print(op.GetOperandCell(constants.MeritColumn_Param1).Value)
                # print(op.GetOperandCell(constants.MeritColumn_Param2).Value)
                # print(op.GetOperandCell(constants.MeritColumn_Param3).Value)
                rms = op.Value
                wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength

                RMS_WFE[i_wave, j_field] = wavelength * 1e3 * rms  # We assume the Wavelength comes in Microns
                # print("Row #%d: Wave %.3f micron | Field #%d -> RMS %.3f" % (irow, wavelength, j_field + 1, rms))

                # If for one the AO modes we get an RMS value of 0.0, print the data so we can double check the Zemax file
                if RMS_WFE[i_wave, j_field] == 0.0:
                    print("\nConfig #%d | Wave #%d | Field #%d" % (config, wave_idx, j_field + 1))

                output = normUnPolData.ReadNextResult()
                if output[2] == 0:
                    x, y = output[4], output[5]
                    foc_xy[i_wave, j_field, 0] = x
                    foc_xy[i_wave, j_field, 1] = y

                    vignet_code = output[3]
                    if vignet_code != 0:

                        vignetting_surface = system.LDE.GetSurfaceAt(vignet_code).Comment
                        # print("\nConfig #%d" % (config))
                        # print("Vignetting at surface #%d: %s" % (vignet_code, vignetting_surface))

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()
        # time_res = time() - start
        # print("Time spent reading results: %.3f sec" % time_res)

        # time_total = time() - start0
        # print("TOTAL Time: %.3f sec" % time_total)
        # sec_per_wave = time_total / N_waves * 1000
        # print("%3.f millisec per Wavelength" % sec_per_wave)
        #
        #
        # plt.scatter(foc_xy[:, :, 0].flatten(), foc_xy[:, :, 1].flatten(), s=2, color='black')

        return [RMS_WFE, obj_xy, foc_xy]

    # The following functions depend on how you want to post-process your analysis results
    # For example, "loop_over_files" automatically runs the RMS WFE analysis across all Zemax files
    # and uses "flip_rms" to reorder the Configurations -> Slice #,
    # and "plot_RMS_WFE_maps" to save the figures

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=3, pupil_sampling=4,
                        save_hdf5=True, remove_slicer_aperture=False):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_rms_wfe


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param spaxels_per_slice: number spaxels to sample each slice with
        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['RMS_WFE', 'OBJ_XY', 'FOC_XY']
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves, spaxels_per_slice,), (spaxels_per_slice, 2), (N_waves, spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx


        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            if remove_slicer_aperture == True:
                system = settings['system']
                spaxel_scale = settings['scale']
                ifu = settings['ifu']
                # We need to know what is the Surface Number for the Image Slicer in the Zemax file
                slicer_surface = focal_planes[system][spaxel_scale][ifu]['IS']
            else:
                slicer_surface = None

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_wfe,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice,
                                             pupil_sampling=pupil_sampling, slicer_surface=slicer_surface)

            results.append(list_results)

            if save_hdf5 == True:
                # Post-Processing the results
                file_name = zemax_file.split('.')[0]
                settings['surface'] = 'DETECTOR' if surface is None else surface
                self.save_hdf5(analysis_name='RMS_WFE', analysis_metadata=metadata, list_results=list_results, results_names=results_names,
                               file_name=file_name, file_settings=settings, results_dir=results_path)


        return results

class EnsquaredEnergyFastAnalysis(AnalysisFast):
    """
    Faster version of the Ensquared Energy calculation
    The outside loop is only for the configurations
    For each configuration, we construct a single Raytrace object
    that covers all the specified wavelengths
    """

    @staticmethod
    def analysis_function_ensquared(system, wavelength_idx, config, surface, slicer_surface, px, py, box_size):
        """
        Calculation of the Ensquared Energy




        :param system: the Zemax optical system
        :param wavelength_idx: list of the Zemax Wavelength indices we want to use
        :param config: the Zemax Configuration (Slice) we are going to use
        :param surface: [not used] we always do the calculation at the last surface (Detector)
        :param slicer_surface: Zemax surface number for the Image Slicer
        :param px: Pupil X coordinates of the rays we will trace
        :param py: Pupil Y coordinates of the rays we will trace
        :param box_size: size of the box in [detector pixels] that we'll use for the calculation
        :return:
        """

        det_pix = 15e-3         # Size of the detector pixel [mm]

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # First of all, we need to find the Surface Number for the IMAGE SLICER "Image Plane"
        N_surfaces = system.LDE.NumberOfSurfaces
        surface_names = {}  # A dictionary of surface number -> surface comment
        for k in np.arange(1, N_surfaces):
            surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
        # find the Slicer surface number
        try:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image Plane')]
        except ValueError:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image plane')]
        slicer_surface = slicer_num
        # slicer = system.LDE.GetSurfaceAt(slicer_num)

        # Double check that the Image Slicer surface number is correct
        slicer = system.LDE.GetSurfaceAt(slicer_surface).Comment
        if slicer != 'Image Plane' and slicer != 'Image plane':
            print(slicer)
            raise ValueError("Surface #%d is not the Image Slicer. Please check the Zemax file" % slicer_surface)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Use the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / X_MAX, fy / Y_MAX  # Normalized field coordinates (hx, hy)
        obj_xy = np.array([fx, fy])

        N_pupil = px.shape[0]   # Number of rays in the Pupil for a given field point and wavelength
        N_rays = N_waves * N_pupil

        EE = np.empty(N_waves)
        sli_foc_xy = np.empty((N_waves, 2))
        det_foc_xy = np.empty((N_waves, 2))

        slicer_xy = np.empty((N_waves, N_pupil, 2))
        slicer_xy[:] = np.nan
        detector_xy = np.empty((N_waves, N_pupil, 2))
        detector_xy[:] = np.nan

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)

        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            for (p_x, p_y) in zip(px, py):  # Add the ray to the RayTrace
                rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        for k in range(N_rays):  # Get Raytrace results at the Image Slicer
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            # print(i_wave, j_pupil)
            output = rays_slicer.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                slicer_xy[i_wave, j_pupil, 0] = output[4]
                slicer_xy[i_wave, j_pupil, 1] = output[5]
                checksum_slicer += 1
        if checksum_slicer < N_rays:
            raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()

        # Count how many rays fall inside a +- 1 mm window in Y, wrt the centroid
        slicer_cent_x = np.nanmean(slicer_xy[:, :, 0], axis=1)
        slicer_cent_y = np.nanmean(slicer_xy[:, :, 1], axis=1)
        sli_foc_xy[:, 0] = slicer_cent_x
        sli_foc_xy[:, 1] = slicer_cent_y

        # print(slicer_cent_y)
        below_slicer = slicer_xy[:, :, 1] < slicer_cent_y[:, np.newaxis] + 1.0 * box_size / 2
        above_slicer = slicer_xy[:, :, 1] > slicer_cent_y[:, np.newaxis] - 1.0 * box_size / 2
        inside_slicer = (np.logical_and(below_slicer, above_slicer))
        # print(inside_slicer[0, :10])

        # Now, for each wavelength, we calculate which rays fulfil the Image Slicer conditions
        index_valid_slicer = [np.argwhere(inside_slicer[i, :] == True)[:, 0] for i in range(N_waves)]
        # print(index_valid_slicer[1][:10])
        # print(index_valid_slicer[2][:10])

        # (2) Run the raytrace up to the DETECTOR
        # For speed, we re-use the same Raytrace, just define new rays!
        # raytrace_det = system.Tools.OpenBatchRayTrace()
        # Detector is always the last surface
        detector_surface = system.LDE.NumberOfSurfaces - 1
        rays_detector = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)
        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            for (p_x, p_y) in zip(px, py):
                rays_detector.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()

        rays_detector.StartReadingResults()
        checksum_detector = 0
        # index_valid_detector = []       # Valid means they make it to the detector even if vignetted at the Slicer
        vignetted = []
        index_vignetted = []
        index_valid_detector = np.empty((N_waves, N_pupil))
        index_valid_detector[:] = np.nan
        for k in range(N_rays):  # Get Raytrace results at the Detector
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            output = rays_detector.ReadNextResult()
            if output[2] == 0 and output[3] == 0:       # ErrorCode & VignetteCode
                detector_xy[i_wave, j_pupil, 0] = output[4]
                detector_xy[i_wave, j_pupil, 1] = output[5]
                checksum_detector += 1
                index_valid_detector[i_wave, j_pupil] = j_pupil

            elif output[2] == 0 and output[3] != 0:
                # Some rays are vignetted
                vignetted.append([output[4],  output[5]])
                detector_xy[i_wave, j_pupil, 0] = output[4]
                detector_xy[i_wave, j_pupil, 1] = output[5]
                checksum_detector += 1
                index_valid_detector[i_wave, j_pupil] = j_pupil
                index_vignetted.append(k)

        # index_valid_detector = np.array(index_valid_detector)
        # # print(index_valid_detector.shape)
        # # print(index_valid_detector)
        # index_valid_detector = index_valid_detector.reshape((N_waves, N_pupil))
        # # print(index_valid_detector.shape)

        rays_detector.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # (3) Calculate the ENSQUARED ENERGY
        # We only count the rays that where inside the slicer to begin with and the ones that make it to the detector
        for i_wave in range(N_waves):
            valid_both = []
            for k in range(N_pupil):
                # print(index_valid_detector[i_wave])
                if k in index_valid_slicer[i_wave] and k in index_valid_detector[i_wave]:
                    valid_both.append(k)

            valid_det_x = detector_xy[i_wave, :, 0][valid_both]
            valid_det_y = detector_xy[i_wave, :, 1][valid_both]

            # Now, out of the VALID rays, we calculate which detector rays fall inside a 2x pixel box along X
            dcx = np.mean(valid_det_x)  # Detector Centroid X
            dcy = np.mean(valid_det_y)
            det_foc_xy[i_wave] = [dcx, dcy]

            left_detector = valid_det_x < dcx + det_pix * box_size / 2
            right_detector = valid_det_x > dcx - det_pix * box_size / 2
            inside_detector = (np.logical_and(left_detector, right_detector))
            total_detector = np.sum(inside_detector)
            ensq = total_detector / N_pupil
            # print(ensq)
            EE[i_wave] = ensq * 0.98

        # fig, axes = plt.subplots(2, N_waves)
        # colors = cm.Reds(np.linspace(0.5, 1, N_waves))
        # for j in range(N_waves):
        #     ax1 = axes[0][j]
        #     ax1.scatter(slicer_xy[j, :, 0], slicer_xy[j, :, 1], s=3, color=colors[j])
        #     ax1.scatter(sli_foc_xy[j, 0], sli_foc_xy[j, 1], s=3, color='black')
        #     scy = sli_foc_xy[j, 1]
        #     ax1.axhline(y=scy + 1.0 * box_size / 2, color='black', linestyle='--')
        #     ax1.axhline(y=scy - 1.0 * box_size / 2, color='black', linestyle='--')
        #     wavelength = system.SystemData.Wavelengths.GetWavelength(wavelength_idx[j]).Wavelength
        #     ax1.set_title("%.3f $\mu$m" % wavelength)
        #
        #     ax2 = axes[1][j]
        #     dcx = det_foc_xy[j, 0]
        #     ax2.scatter(detector_xy[j, :, 0], detector_xy[j, :, 1], s=3, color=colors[j])
        #     ax2.scatter(det_foc_xy[j, 0], det_foc_xy[j, 1], s=3, color='black')
        #     ax2.axvline(x=dcx + det_pix * box_size / 2, color='black', linestyle='--')
        #     ax2.axvline(x=dcx - det_pix * box_size / 2, color='black', linestyle='--')
        #
        # plt.show()

        return EE, obj_xy, sli_foc_xy, det_foc_xy

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, N_rays=500, box_size=2):
        """

        """

        # We want the result to produce as output: Ensquared Energy, Object coords, Image Slicer and Detector centroids
        results_names = ['EE', 'OBJ_XY', 'SLI_XY', 'DET_XY']
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves,), (2,), (N_waves, 2,), (N_waves, 2,)]

        metadata = {}
        metadata['N_rays'] = N_rays
        metadata['Box Size Spaxels'] = box_size
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options
        file_list, file_settings = create_zemax_file_list(which_system=files_opt['which_system'],
                                                          AO_modes=files_opt['AO_modes'],
                                                          scales=files_opt['scales'], IFUs=files_opt['IFUs'],
                                                          grating=files_opt['grating'])

        results = []
        for zemax_file, settings in zip(file_list, file_settings):
            system = settings['system']
            spaxel_scale = settings['scale']
            ifu = settings['ifu']
            # We need to know what is the Surface Number for the Image Slicer in the Zemax file
            slicer_surface = focal_planes[system][spaxel_scale][ifu]['IS']
            # print(slicer_surface)

            # Generate a set of random pupil rays
            px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
            print("Using %d rays" % N_rays)

            list_results = self.run_analysis(analysis_function=self.analysis_function_ensquared,
                                        files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                        results_shapes=results_shapes, results_names=results_names,
                                        wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                        slicer_surface=slicer_surface, px=px, py=py, box_size=box_size)

            results.append(list_results)

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            settings['surface'] = 'DETECTOR'
            self.save_hdf5(analysis_name='ENSQ_ENERG', analysis_metadata=metadata, list_results=list_results, results_names=results_names,
                           file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


class FWHM_PSF_FastAnalysis(AnalysisFast):
    """

    """

    def calculate_fwhm(self, surface, xy_data, PSF_window, N_points, spaxel_scale, wavelength):

        start = time()
        # Calculate the Geometric PSF
        x, y = xy_data[:, 0], xy_data[:, 1]
        cent_x, cent_y = np.mean(x), np.mean(y)

        std_x, std_y = np.std(x), np.std(y)
        bandwidth = min(std_x, std_y)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy_data)

        # define a grid to compute the PSF
        xmin, xmax = cent_x - PSF_window/2/1000, cent_x + PSF_window/2/1000
        ymin, ymax = cent_y - PSF_window/2/1000, cent_y + PSF_window/2/1000
        x_grid = np.linspace(xmin, xmax, N_points)
        y_grid = np.linspace(ymin, ymax, N_points)
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        xy_grid = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T
        log_scores = kde.score_samples(xy_grid)

        psf_geo = np.exp(log_scores)
        psf_geo /= np.max(psf_geo)
        psf_geo = psf_geo.reshape(xx_grid.shape)

        time_geopsf = time() - start
        # print("Time to estimate GeoPSF: %.3f sec" % time_geo)
        start = time()

        psf_diffr = diffraction.add_diffraction(surface=surface, psf_geo=psf_geo, PSF_window=PSF_window,
                                                scale_mas=spaxel_scale, wavelength=wavelength)
        time_diffpsf = time() - start
        # print("Time to add Diffraction: %.3f sec" % time_diffpsf)
        start = time()

        # guesses = {'IS': {4.0: }}

        # Fit the PSF to a 2D Gaussian
        guess_x = PSF_window / 2 / 1000
        fwhm_x, fwhm_y, theta = diffraction.fit_psf_to_gaussian(xx=xx_grid, yy=yy_grid, psf_data=psf_diffr,
                                                                x0=cent_x, y0=cent_y, sigmax0=guess_x, sigmay0=guess_x)

        time_gauss = time() - start

        # print('FWHM time: %.3f sec for GeoPSF estimate:' % time_geopsf)
        # print('FWHM time: %.3f sec for DiffPSF convolution:' % time_diffpsf)
        # print('FWHM time: %.3f sec for Gaussian fit:' % time_gauss)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # img1 = ax1.imshow(psf_geo, extent=[xmin, xmax, ymin, ymax], cmap='bwr', origin='lower')
        # plt.colorbar(img1, ax=ax1, orientation='horizontal')
        # ax1.set_xlabel(r'X [mm]')
        # ax1.set_ylabel(r'Y [mm]')
        # ax1.set_title(r'Geometric PSF estimate | Surface: %s' % surface)
        #
        # img2 = ax2.imshow(psf_diffr, extent=[xmin, xmax, ymin, ymax], cmap='bwr', origin='lower')
        # plt.colorbar(img2, ax=ax2, orientation='horizontal')
        # ax2.set_xlabel(r'X [mm]')
        # ax2.set_ylabel(r'Y [mm]')
        # if surface == 'DET':
        #     ax2.set_title(r'Diffr. PSF | %.3f microns | %.1f mas | FWHM_x: %.1f $\mu$m' % (wavelength, spaxel_scale, fwhm_x))
        # elif surface == 'IS':
        #     ax2.set_title(r'Diffr. PSF | %.3f microns | %.1f mas | FWHM_y: %.1f $\mu$m' % (wavelength, spaxel_scale, fwhm_y))

        return fwhm_x, fwhm_y

    def analysis_function_fwhm_psf(self, system, wavelength_idx, config, surface, slicer_surface, px, py, spaxel_scale,
                                      N_points, PSF_window):

        start0 = time()
        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # First of all, we need to find the Surface Number for the IMAGE SLICER "Image Plane"
        N_surfaces = system.LDE.NumberOfSurfaces
        surface_names = {}  # A dictionary of surface number -> surface comment
        for k in np.arange(1, N_surfaces):
            surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
        # find the Slicer surface number
        try:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image Plane')]
        except ValueError:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image plane')]
        slicer_surface = slicer_num

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Use the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / X_MAX, fy / Y_MAX  # Normalized field coordinates (hx, hy)
        obj_xy = np.array([fx, fy])

        N_pupil = px.shape[0]   # Number of rays in the Pupil for a given field point and wavelength
        N_rays = N_waves * N_pupil

        FWHM = np.zeros((N_waves, 2))
        foc_xy = np.zeros((N_waves, 2))

        slicer_xy = np.empty((N_waves, N_pupil, 2))
        slicer_xy[:] = np.nan
        detector_xy = np.empty((N_waves, N_pupil, 2))
        detector_xy[:] = np.nan

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)

        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            for (p_x, p_y) in zip(px, py):  # Add the ray to the RayTrace
                rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        for k in range(N_rays):  # Get Raytrace results at the Image Slicer
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            # print(i_wave, j_pupil)
            output = rays_slicer.ReadNextResult()
            if output[2] == 0:
                slicer_xy[i_wave, j_pupil, 0] = output[4]
                slicer_xy[i_wave, j_pupil, 1] = output[5]
                checksum_slicer += 1
        if checksum_slicer < N_rays:
            raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()

        # (2) Run the raytrace up to the DETECTOR
        # For speed, we re-use the same Raytrace, just define new rays!
        # raytrace_det = system.Tools.OpenBatchRayTrace()
        # Detector is always the last surface
        detector_surface = system.LDE.NumberOfSurfaces - 1
        rays_detector = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)
        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            for (p_x, p_y) in zip(px, py):
                rays_detector.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()

        rays_detector.StartReadingResults()
        checksum_detector = 0
        index_valid_detector = np.empty((N_waves, N_pupil))
        index_valid_detector[:] = np.nan
        for k in range(N_rays):  # Get Raytrace results at the Detector
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            output = rays_detector.ReadNextResult()
            if output[2] == 0:       # ErrorCode & VignetteCode
                detector_xy[i_wave, j_pupil, 0] = output[4]
                detector_xy[i_wave, j_pupil, 1] = output[5]
                checksum_detector += 1
                index_valid_detector[i_wave, j_pupil] = j_pupil
            #
            # elif output[2] == 0 and output[3] != 0:
            #     # Some rays are vignetted
            #     code = output[3]
            #     vignetting_surface = system.LDE.GetSurfaceAt(code).Comment
            #     # print("Config #%d | Wave #%d || Vignetting at surface #%d: %s" % (config, wavelength_idx[i_wave], code, vignetting_surface))

        rays_detector.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # time_rays = time() - start0
        # print("Time for Raytrace Slicer and Detector: %.3f sec" % time_rays)

        # fig, axes = plt.subplots(2, N_waves)
        # colors = cm.Reds(np.linspace(0.5, 1, N_waves))
        # for j in range(N_waves):
        #     ax1 = axes[0][j]
        #     ax1.scatter(slicer_xy[j, :, 0], slicer_xy[j, :, 1], s=3, color=colors[j])
        #     # scx =
        #     # ax1.scatter(sli_foc_xy[j, 0], sli_foc_xy[j, 1], s=3, color='black')
        #     # scy = sli_foc_xy[j, 1]
        #     # ax1.axhline(y=scy + 1.0 * box_size / 2, color='black', linestyle='--')
        #     # ax1.axhline(y=scy - 1.0 * box_size / 2, color='black', linestyle='--')
        #     wavelength = system.SystemData.Wavelengths.GetWavelength(wavelength_idx[j]).Wavelength
        #     ax1.set_title("%.3f $\mu$m" % wavelength)
        #
        #     ax2 = axes[1][j]
        #     # dcx = det_foc_xy[j, 0]
        #     ax2.scatter(detector_xy[j, :, 0], detector_xy[j, :, 1], s=3, color=colors[j])
        #     # ax2.scatter(det_foc_xy[j, 0], det_foc_xy[j, 1], s=3, color='black')
        #     # ax2.axvline(x=dcx + det_pix * box_size / 2, color='black', linestyle='--')
        #     # ax2.axvline(x=dcx - det_pix * box_size / 2, color='black', linestyle='--')
        #

        # FWHM

        windows = {4.0: [5000, 200], 60.0: [500, 50]}
        win_slicer = windows[spaxel_scale][0]
        win_detect = windows[spaxel_scale][1]
        if config == 1:
            print("Sampling the Image Slicer plane with %d points: %.3f microns / point" % (N_points, (win_slicer / N_points)))
            print("Sampling the Detector plane with %d points: %.3f microns / point" % (N_points, (win_detect / N_points)))

        for i_wave, wave_idx in enumerate(wavelength_idx):

            xy_data_slicer = slicer_xy[i_wave]
            wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength
            fwhm_x_slicer, fwhm_y_slicer = self.calculate_fwhm(surface='IS', xy_data=xy_data_slicer, PSF_window=win_slicer,
                                                               N_points=N_points, spaxel_scale=spaxel_scale,
                                                               wavelength=wavelength)

            xy_data_detector = detector_xy[i_wave]
            fwhm_x_det, fwhm_y_det = self.calculate_fwhm(surface='DET', xy_data=xy_data_detector, PSF_window=win_detect,
                                                         N_points=N_points, spaxel_scale=spaxel_scale,
                                                         wavelength=wavelength)

            # plt.show()

            foc_xy[i_wave] = [np.mean(xy_data_detector[:, 0]), np.mean(xy_data_detector[:, 1])]

            FWHM[i_wave] = [fwhm_x_det, fwhm_y_slicer]

            if config == 1:
                print("%.3f microns" % wavelength)
                print("FWHM in X [Detector]: %.1f microns | in Y [Image Slicer]: %.2f microns " % (fwhm_x_det, fwhm_y_slicer))

        # plt.show()

        return FWHM, obj_xy, foc_xy


    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, N_rays=500, N_points=50, PSF_window=150):
        """

        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['FWHM', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        results_shapes = [(N_waves, 2,), (2,), (N_waves, 2,)]

        px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
        print("Using %d rays" % N_rays)

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            # Clear the Cache for the Airy Pattern function
            airy_and_slicer.cache_clear()

            mas_dict = {'4x4': 4.0, '10x10': 10.0, '20x20': 20.0, '60x30': 60.0}
            spaxel_scale = mas_dict[settings['scale']]

            # We need to know what is the Surface Number for the Image Slicer in the Zemax file
            slicer_surface = focal_planes[settings['system']][settings['scale']][settings['ifu']]['IS']

            list_results = self.run_analysis(analysis_function=self.analysis_function_fwhm_psf,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, slicer_surface=slicer_surface, px=px, py=py,
                                             spaxel_scale=spaxel_scale, N_points=N_points, PSF_window=PSF_window)
            results.append(list_results)

        return results


class CommonWavelengthRange(AnalysisFast):
    """
    Calculate the Common Wavelength Range

    At some point we saw that, in some cases, the rays fall outside the active area
    of the detector (specially for short wavelengths). This means that the wavelengths
    defined in Zemax are "optimistic" and that we have to re-calculate the
    Common Wavelength Range
    """

    @staticmethod
    def analysis_function_cwr(system, wavelength_idx, config, surface):
        """
        To calculate the Common Wavelength Range we will perform a raytrace for
        all configurations and wavelengths in the E2E Model Zemax file

        We will use the results of the raytrace at the detector plane to interpolate
        and find the value of the shortest and longest wavelengths for which all slices (configurations)
        fall within the active area of the detector.

        Here we only run the Raytrace and store the values, we will do the post-processing
        in another script e2e_common_wave_range.py

        :param system: the Zemax optical system
        :param wavelength_idx: list of Zemax wavelength indices
        :param config: current configuration
        :param surface: Zemax surface number
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        # We will trace all 3 available Field Points (the centre and both edges of the slice)
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        hx = np.array([sysField.GetField(i + 1).X / X_MAX for i in range(N_fields)])
        hy = np.array([sysField.GetField(i + 1).Y / Y_MAX for i in range(N_fields)])

        # The Field coordinates for the Object
        # obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        foc_xy = np.empty((N_waves, N_fields, 2))

        N_rays = N_waves * N_fields

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, surface)

        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # time_ray = time() - start
        # print("Time spent running Raytrace: %.3f sec" % time_ray)

        # start = time()
        normUnPolData.StartReadingResults()
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                output = normUnPolData.ReadNextResult()
                if output[2] == 0:      # ignore the vignetting
                    x, y = output[4], output[5]
                    foc_xy[i_wave, j_field, 0] = x
                    foc_xy[i_wave, j_field, 1] = y

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        return [foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None):

        results_names = ['FOC_XY', ]
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves, 3, 2), ]

        metadata = {}
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_cwr,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface)

            results.append(list_results)

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            settings['surface'] = 'DETECTOR' if surface is None else surface
            self.save_hdf5(analysis_name='COMMON_WAVELENGTH_RANGE', analysis_metadata=metadata,
                           list_results=list_results, results_names=results_names,
                           file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


class Raytrace_FastAnalysis(AnalysisFast):
    """
    Raytrace Analysis [Fast]
    Traces a set of Chief Rays to a given surface (typically the Detector plane)
    """

    @staticmethod
    def analysis_function_raytrace(system, wavelength_idx, config, spaxels_per_slice, surface, ignore_vignetting):
        """
        For a given configuration, trace chief rays for chosen wavelengths and field points (spaxels_per_slice)
        using a single Raytrace

        :param system: Zemax optical system
        :param wavelength_idx: list of Zemax wavelength indices we want
        :param config: current Zemax configuration number
        :param spaxels_per_slice: how many field points per slice we want to trace
        :param surface: Zemax surface number we want to trace the rays to
        :param ignore_vignetting: boolean, whether or not to ignore vignetting
        :return:
        """
        start0 = time()

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)
        N_rays = N_waves * spaxels_per_slice

        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(N_fields).X, sysField.GetField(N_fields).Y

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / X_MAX, fx_max / X_MAX
        hy_min, hy_max = fy_min / Y_MAX, fy_max / Y_MAX

        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)
        # hx = np.array([sysField.GetField(i + 1).X / X_MAX for i in range(N_fields)])
        # hy = np.array([sysField.GetField(i + 1).Y / Y_MAX for i in range(N_fields)])

        # The Field coordinates for the Object
        obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        foc_xy = np.empty((N_waves, spaxels_per_slice, 2))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, surface)

        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # time_ray = time() - start
        # print("Time spent running Raytrace: %.3f sec" % time_ray)

        # start = time()
        normUnPolData.StartReadingResults()

        # Retrieve the results for the operands and raytrace
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                output = normUnPolData.ReadNextResult()
                if ignore_vignetting == False:
                    # We do care about vignetting
                    if output[2] == 0 and output[3] == 0:
                        x, y = output[4], output[5]
                        foc_xy[i_wave, j_field, 0] = x
                        foc_xy[i_wave, j_field, 1] = y

                    elif output[2] == 0 and output[3] != 0:
                        vignet_code = output[3]
                        vignetting_surface = system.LDE.GetSurfaceAt(vignet_code).Comment
                        print("\nConfig #%d | Wavelength idx #%d" % (config, wave_idx))
                        fx, fy = h_x * X_MAX, h_y * Y_MAX
                        print("Field point #%d : hx=%.4f hy=%.4f | fx=%.4f, fy=%.4f" % (j_field + 1, h_x, h_y, fx, fy))
                        print("Vignetting at surface #%d: %s" % (vignet_code, vignetting_surface))
                else:
                    # If we don't care about vignetting (rays falling outside the active area of the detector, for example)
                    # we add the Raytrace results to the focal coordinates array no matter what
                    if output[2] == 0:
                        x, y = output[4], output[5]
                        foc_xy[i_wave, j_field, 0] = x
                        foc_xy[i_wave, j_field, 1] = y

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()
        # time_res = time() - start
        # print("Time spent reading results: %.3f sec" % time_res)

        return [obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=3, ignore_vignetting=False,
                        save_hdf5=False):
        """
        We loop over the Zemax files, calculating the Raytrace

        The results will be (1) an OBJECT XY coordinates array of size [N_configs, spaxels_per_slice, 2], and
        (2) a FOCAL XY coordinates array of size [N_configs, N_waves, spaxels_per_slice, 2], notice the wavelengths!

        """

        results_names = ['OBJ_XY', 'FOC_XY']
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(spaxels_per_slice, 2), (N_waves, spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_raytrace,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             ignore_vignetting=ignore_vignetting,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice)

            results.append(list_results)

            if save_hdf5 == True:
                # We save the arrays in HDF5 format alongside the analysis metadata in case we want to look at it later
                file_name = zemax_file.split('.')[0]
                settings['surface'] = 'DETECTOR' if surface is None else surface
                self.save_hdf5(analysis_name='RAYTRACE', analysis_metadata=metadata, list_results=list_results,
                               results_names=results_names, file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


if __name__ == """__main__""":

    pass



