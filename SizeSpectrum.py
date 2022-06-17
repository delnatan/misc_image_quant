from skimage.morphology import area_opening
from skimage.restoration import rolling_ball
from skimage.transform import rescale
import numpy as np


def remove_background(img, d=80):
    """remove background by scikit-image's `rolling-ball` algorithm

    Args:
        img (2-D array): input image. This is converted to `float` type before processing
        d (float): diameter of rolling ball

    Returns:
        2-D array: background-removed image

    """
    r = d / 2.0
    bgimg = rolling_ball(img.astype(float), radius=r)
    return img - bgimg


def calc_size_spectrum(img, scale=0.25, step=5, end=150, bg_diam=50):
    """compute granular spectrum using area opening instead of reconstruction

    The image is first resized, background-corrected, then measured.
    Instead of using a specific structuring element, the image "opening" is done
    via `area_opening`, which is does not use a structuring element, but uses
    an area instead. Each length scale that is specified is therefore interpre-
    ted as an equivalent area of a circle.

    Briefly, the algorithm works as such:

        loop through area sizes to be measured:
            do area_opening on input image with specified area value
            sum all intensities of opened image
            compute difference in total summed intensities from previous step
            store this value in a vector

    Background correction is done by the "rolling-ball" reconstruction. See
    the `skimage.reconstruction` module for more details.


    Args:
        scale (float):
            input image will be scaled by this amount. Mostly used to speed up
            calculations by reducing computational burden of working with large
            images.
        step (integer):
            the size spectrum spacing
        end (integer):
            the biggest dimension to calculate in the spectrum

    Returns:
        (1-D array):
            the x-axis for the size spectrum
        (1-D array):
            the y-axis for the size spectrum. the output is scaled by the mean
            intensity of the original image. The intent is to also inform
            about how much signal exists in the image, so a comparison between
            a dense/bright and sparse/dim is possible.

    """

    # a list of length scale (diameters)
    diams = [
        1,
    ] + list(range(step, end, step))
    diams = np.array(diams, dtype=float)

    # adjust to scaling
    diams *= scale

    # compute image mean and rescale image to reduce computation burden
    img = img.astype(float)
    origmean = img.mean()

    if scale != 1.0:
        img = rescale(img, scale)

    if bg_diam > 0:
        img = remove_background(img, d=bg_diam * scale)

    # begin algorithm,
    Nsteps = len(diams)
    # allocate an array for the spectrum y-axis
    gs = np.zeros(Nsteps)
    # save the total intensity before image is processed
    sum0 = img.sum()
    # variable for sum of current image intensity
    curint = sum0

    for i, d in enumerate(diams):
        prevint = curint
        area = np.pi * (d / 2) ** 2
        wrk = area_opening(img, area)
        curint = wrk.sum()
        gs[i] = prevint - curint

    # normalize to initial intensity
    gs = gs / sum0 * 100.0

    # scale back diameters
    diams /= scale

    # compute distribution normalizer
    z = np.trapz(gs, x=diams)
    # compute the distribution scaled by original image mean intensity
    norm_gs = origmean * (gs / z)

    return diams, norm_gs
