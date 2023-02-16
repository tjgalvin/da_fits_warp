#! /usr/bin/env python
"""
Tool to derive positional shifts of sources (presumably from the ionosphere) and, if requested, correct for. 

After cross-referencing the input component caatalogue against a base catalogue, images may be reinterpolated onto a corrected pixel grid. Interpolation is performed using a radial-basis function. 

Thanks for using fits_warp! To cite this package, please use the following BibTeX:

        @ARTICLE{2018A&C....25...94H,
            author = {{Hurley-Walker}, N. and {Hancock}, P.~J.},
            title = \"{De-distorting ionospheric effects in the image plane}\",
            journal = {Astronomy and Computing},
        archivePrefix = \"arXiv\",
            elogger.info = {1808.08017},
            primaryClass = \"astro-ph.IM\",
            keywords = {Astrometry, Radio astronomy, Algorithms, Ionosphere},
                year = 2018,
            month = oct,
            volume = 25,
            pages = {94-102},
                doi = {10.1016/j.ascom.2018.08.006},
            adsurl = {http://adsabs.harvard.edu/abs/2018A%26C....25...94H},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

Other formats can be found here: http://adsabs.harvard.edu/abs/2018A%26C....25...94H
"""

import argparse
import glob
import logging
import sys
import warnings
from collections import namedtuple
from pathlib import Path
from time import sleep, time
from typing import Any, Optional, Union

import astropy.units as u
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.stats.circstats import circmean
from astropy.table import Table, hstack
from astropy.utils.exceptions import AstropyWarning
from dask.distributed import Client, LocalCluster
from matplotlib import gridspec
from scipy import interpolate
from scipy.interpolate import CloughTocher2DInterpolator

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(levelname)s:%(lineno)d %(message)s")
logger.setLevel(logging.INFO)

Array = Union[da.array, np.array]

OffsetModel = namedtuple("OffsetModel", ("dx", "dy"))


def make_source_plot(
    fname: Path,
    hdr: dict[str, Any],
    cat_xy: np.ndarray,
    diff_xy: np.ndarray,
    offset_models: OffsetModel,
    cmap: str = "hsv",
) -> None:
    """Create a figure showing the cross-matched sources and their offsets, as 
    well as the interpolated screen of offsets

    Args:
        fname (Path): Output file name of the figure
        hdr (dict[str, Any]): Header of the fits file this figure represents the offset screen for
        cat_xy (np.ndarray): Source positions in pixel space
        diff_xy (np.ndarray): Offset of source positions in pixels
        offset_models (OffsetModel): RBF-interpolation functions used to generate the screen
        cmap (str, optional): Alternative colour map to use. Defaults to "hsv".
    """
    xmin, xmax = 0, hdr["NAXIS1"]
    ymin, ymax = 0, hdr["NAXIS2"]

    gx, gy = np.mgrid[
        xmin : xmax : (xmax - xmin) / 50.0, ymin : ymax : (ymax - ymin) / 50.0
    ]
    mdx = offset_models.dx(np.ravel(gx), np.ravel(gy))
    mdy = offset_models.dy(np.ravel(gx), np.ravel(gy))
    x = cat_xy[:, 0]
    y = cat_xy[:, 1]

    # plot w.r.t. centre of image, in degrees
    try:
        delX = abs(hdr["CD1_1"])
        delY = abs(hdr["CD2_2"])
    except:
        delX = abs(hdr["CDELT1"])
        delY = abs(hdr["CDELT2"])

    # shift all co-ordinates and put them in degrees
    x -= hdr["NAXIS1"] / 2
    gx -= hdr["NAXIS1"] / 2
    xmin -= hdr["NAXIS1"] / 2
    xmax -= hdr["NAXIS1"] / 2
    x *= delX
    gx *= delX
    xmin *= delX
    xmax *= delX
    y -= hdr["NAXIS2"] / 2
    gy -= hdr["NAXIS2"] / 2
    ymin -= hdr["NAXIS2"] / 2
    ymax -= hdr["NAXIS2"] / 2
    y *= delY
    gy *= delY
    ymin *= delY
    ymax *= delY
    scale = 1

    dx = diff_xy[:, 0]
    dy = diff_xy[:, 1]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(100, 100)
    gs.update(hspace=0, wspace=0)
    kwargs = {
        "angles": "xy",
        "scale_units": "xy",
        "scale": scale,
        "cmap": cmap,
        "clim": [-180, 180],
    }
    angles = np.degrees(np.arctan2(dy, dx))
    ax = fig.add_subplot(gs[0:100, 0:48])
    cax = ax.quiver(x, y, dx, dy, angles, **kwargs)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_xlabel("Distance from pointing centre / degrees")
    ax.set_ylabel("Distance from pointing centre / degrees")
    ax.set_title("Source position offsets / arcsec")

    ax = fig.add_subplot(gs[0:100, 49:97])
    cax = ax.quiver(gx, gy, mdx, mdy, np.degrees(np.arctan2(mdy, mdx)), **kwargs)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_xlabel("Distance from pointing centre / degrees")
    ax.tick_params(axis="y", labelleft="off")
    ax.set_title("Model position offsets / arcsec")

    # Color bar
    ax2 = fig.add_subplot(gs[0:100, 98:100])
    cbar3 = plt.colorbar(cax, cax=ax2, use_gridspec=True)
    cbar3.set_label("Angle CCW from West / degrees")  # ,labelpad=-75)
    cbar3.ax.yaxis.set_ticks_position("right")

    outname = fname.with_suffix(".png")

    fig.savefig(outname, dpi=200)


def create_divergence_map(
    fnames: list[Path], xy: Array, x: Array, y: Array, nx: int, ny: int
) -> None:
    """Generate a divergence map of source offsets. This (appears) to be the 
    sum of the gradients of offsets in both X and Y directions. 

    Args:
        fnames (list[Path]): Collection of fits images. The header of the first is used to generate the map. 
        xy (Array): Pixel positions of sources in x and y. 
        x (Array): Offset source positions in x direction
        y (Array): Offset source positions in y direction
        nx (int): Number of x pixels to reshape to
        ny (int): Number of y pixels to reshape to
    """
    logger.info("Creating divergence maps")
    start = time()
    # Save the divergence as a fits image
    im = fits.open(fnames[0])
    outputname = fnames[0].replace(".fits", "")
    div = (
        np.gradient((x - np.array(xy[1, :])).reshape((nx, ny)))[0]
        + np.gradient((y - np.array(xy[0, :])).reshape((nx, ny)))[1]
    )
    im[0].data = div
    im.writeto(outputname + "_div.fits", overwrite=True)
    im[0].data = (x - np.array(xy[1, :])).reshape((nx, ny))
    im.writeto(outputname + "_delx.fits", overwrite=True)
    im[0].data = (y - np.array(xy[0, :])).reshape((nx, ny))
    im.writeto(outputname + "_dely.fits", overwrite=True)
    logger.info("finished divergence map in {0} seconds".format(time() - start))


def make_pix_models(
    fname: Path,
    ra1: str = "ra",
    dec1: str = "dec",
    ra2="RAJ2000",
    dec2: str = "DEJ2000",
    fitsname: Optional[Path] = None,
    plots: bool = False,
    smooth: float = 300.0,
    sigcol: Optional[str] = None,
    noisecol: Optional[str] = None,
    SNR: float = 10,
    max_sources: Optional[int] = None,
) -> OffsetModel:
    """Identified offset sources and create pair of RBF interpolator models (in pixel units)
    to shift positions by to effectively 'dewarp.;

    Args:
        fname (Path): Name of the crossed matched source table 
        ra1 (str, optional): RA column of input catalogue 1. Defaults to "ra".
        dec1 (str, optional): Dec column of input catalogue 1. Defaults to "dec".
        ra2 (str, optional): RA column of reference catalogue. Defaults to "RAJ2000".
        dec2 (str, optional): Dec column of reference catalogue. Defaults to "DEJ2000".
        fitsname (Optional[Path], optional): Name of fits image to get WCS header from, which is used to convert to pixel coordinates. Defaults to None.
        plots (bool, optional): Create the source offset plot. Defaults to False.
        smooth (float, optional): Smoothing factor used by RBF-interpolator. Defaults to 300.0.
        sigcol (Optional[str], optional): Name of column containing source brightness. Defaults to None.
        noisecol (Optional[str], optional): Name of column containing noise level at source position. SNR may be derived if sigcol is also not None. Defaults to None.
        SNR (float, optional): SNR cut applied to all sources. Defaults to 10.
        max_sources (Optional[int], optional): Maximum number of sources to include in model. Reducing this improves performance at cost of screen accuracy. Defaults to None.

    Raises:
        ValueError: Issued when table of sources can not be opened

    Returns:
        OffsetModel: Pair of RBF interpolator models for x and y directions. 
    """
    file_extension = fname.suffix
    logger.debug(f"File extension of {fname} is {file_extension}.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyWarning)
        logger.debug(f"Opening {fname}")
        logger.debug(f"Extension is {file_extension}")
        if file_extension == ".fits":
            logger.info(f"Opening fits file {fname=}")
            raw_data = fits.open(fname)[1].data
        elif file_extension == ".vot":
            logger.info(f"Opening VOTable {fname=}.")
            raw_data = parse_single_table(fname).array
        else:
            raise ValueError(f"File format of {fname} is not supported. ")

    # get the wcs
    logger.debug("Getting header")
    hdr = fits.getheader(fitsname)
    imwcs = wcs.WCS(hdr, naxis=2)

    raw_nsrcs = len(raw_data)

    # filter the data to only include SNR>10 sources
    if None not in (sigcol, noisecol):
        logger.debug(
            f"Filtering {SNR=} sources using {sigcol=} and {noisecol} columns. "
        )
        flux_mask = np.where(raw_data[sigcol] / raw_data[noisecol] > SNR)
        data = raw_data[flux_mask]
    else:
        data = raw_data

    if max_sources is not None and max_sources < len(data) - 1:
        if sigcol is None:
            # Will help to ensure sources are uniformly sampled from
            # across the field
            sort_idx = np.random.choice(
                np.arange(len(data)), size=max_sources, replace=False
            )
            logger.info(f"Randomly selecting {max_sources} sources...")
        else:
            # argsort goes in ascending order, so select from the end
            sort_idx = np.squeeze(np.argsort(data[sigcol]))[-max_sources:]
            logger.info(
                f"Selecting the {max_sources} brightest of {raw_nsrcs} sources..."
            )

        data = data[sort_idx]

    logger.info(
        f"Selected {len(data)} of {raw_nsrcs} available sources to construct the pixel offset model."
    )

    start = time()

    cat_xy = imwcs.all_world2pix(np.asarray(list(zip(data[ra1], data[dec1]))), 1)
    logger.debug(f"Formed cat_xy array: {cat_xy.shape=}")

    ref_xy = imwcs.all_world2pix(np.asarray(list(zip(data[ra2], data[dec2]))), 1)
    logger.debug(f"Formed ref_xy array: {ref_xy.shape=}")

    diff_xy = ref_xy - cat_xy
    logger.debug(f"Formed diff_xy array: {diff_xy.shape}")

    offset_x_model = interpolate.Rbf(
        cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 0], function="linear", smooth=smooth
    )
    offset_y_model = interpolate.Rbf(
        cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 1], function="linear", smooth=smooth
    )

    offset_models = OffsetModel(dx=offset_x_model, dy=offset_y_model)

    logger.info(f"Model created in {time() - start} seconds.")

    if plots:
        make_source_plot(fname, hdr, cat_xy, diff_xy, offset_models)

    return offset_models


def derive_apply_Clough(
    offset_x: Array,
    offset_y: Array,
    data: Array,
    reference_x: Array,
    reference_y: Array,
) -> Array:
    """Create and evaluate a CloughTocher2DInterpolator function to shift image
    pixels with. All inputs need to be the same length. 

    Args:
        offset_x (Array): The offset pixel x-coordinate frame (i.e. true + warped offsets)
        offset_y (Array): The offset pixel y-coordinate frame (i.e. true + warped offsets)
        data (Array): Pixel intensities corresponding to the warped pixels coordinates
        reference_x (Array): Pixel x-coordinate to evaluate to obtain dewarped pixel values
        reference_y (Array): Pixel y-coordinate to evaluate to obtain dewarped pixel values

    Returns:
        Array: True pixel values of the image after dewarping observed positions
    """
    offset_xy = np.array([offset_y, offset_x])

    model = CloughTocher2DInterpolator(offset_xy.T, data)

    return model(np.array([reference_y, reference_x]).T).astype(np.float32)


def correct_images(
    offset_models: OffsetModel, fnames: list[str], suffix: str, testimage: bool=False, overlap_factor: float = 2.
):
    """Given RBF-interpolator models and target images, dewarp them to remove positional
    shifts of sources

    Args:
        offset_models (OffsetModel): (dx, dy) rbf-interpolator models to shift pixels by
        fnames (list[str]): Collection of fits images to correct. These should all have the same pixel coordinates and WCS
        suffix (str): String attached to the end of the file name (before extension)
        testimage (bool, optional): Whether to create a divergence map. Defaults to False.
        overlap_factor (float, optional): Factor to determine number of pixels for the interpolator to share between adjacent blocks. Factor is multiplied against the length of the fastest moving axis. Too low and artefacts will be introduced. Defaults to 1.0.
    """
    # Get co-ordinate system from first image
    # Do not open images at this stage, to save memory
    hdr = fits.getheader(fnames[0])
    nx = hdr["NAXIS1"]
    ny = hdr["NAXIS2"]

    xy = da.indices((ny, nx), dtype=np.float32, chunks=(500, 500))

    logger.debug(f"{xy.shape=}, type: {type(xy)}")

    delta_x = da.map_blocks(
        offset_models.dx, xy[1, :, :], xy[0, :, :], dtype=np.float32
    )

    delta_y = da.map_blocks(
        offset_models.dy, xy[1, :, :], xy[0, :, :], dtype=np.float32
    )

    x = xy[1, :, :] + delta_x
    offset_x = da.reshape(x, (-1,))

    y = xy[0, :, :] + delta_y
    offset_y = da.reshape(y, (-1,))

    if testimage is True:
        logger.info(f"Creating divergence map.")
        logger.info(f"Evaluating x- and y-coordinates for divergence map.")
        create_divergence_map(fnames, xy, x.compute(), y.compute(), nx, ny)

    for fname in fnames:
        fout = fname.replace(".fits", "_" + suffix + ".fits")
        logger.info(f"Interpolating {fname}")
        im = fits.open(fname)
        im.writeto(fout, overwrite=True, output_verify="fix+warn")

        data = im[0].data
        overlap_depth = int(data.shape[-1] * overlap_factor)
        logger.info(
            f"Overlapping fastest moving axis with {overlap_factor=}, setting {overlap_depth=}"
        )
        logger.info(f"Deriving pixel mask")
        finite_mask = np.isfinite(data)
        data[~finite_mask] = 0.0

        data = da.squeeze(da.array(data))
        ravel_data = da.ravel(data)
        reference_x = da.reshape(xy[1, :, :], (-1,))
        reference_y = da.reshape(xy[0, :, :], (-1,))

        logger.debug(f"{offset_x=}")
        logger.debug(f"{offset_y=}")
        logger.debug(f"{ravel_data=}")
        logger.debug(f"{reference_x=}")
        logger.debug(f"{reference_y=}")

        # appears as though map_overlap would prefer equal dimensions
        # to inputs. Should be broadcast-able is arrays are different
        # but yet to master
        warped_data = da.map_overlap(
            derive_apply_Clough,
            offset_x,
            offset_y,
            ravel_data,
            reference_x,
            reference_y,
            dtype=np.float32,
            align_arrays=True,
            allow_rechunk=True,
            depth=overlap_depth,
            # trim=True,
            boundary="nearest",
            meta=np.array((), dtype=np.float32),
        )

        logger.debug(f"{warped_data=}")
        logger.info(f"Dewarping image...")
        warped_data = warped_data.compute()
        logger.info(f"Dewarp finished!")

        logger.info("Reshaping and reapplying pixel mask")
        warped_data = warped_data.reshape(im[0].data.shape)
        warped_data[~finite_mask] = np.nan

        im[0].data = warped_data

        logger.info("saving...")
        im.writeto(fout, overwrite=True, output_verify="fix+warn")
        logger.info(f"wrote {fout}")

    return


def warped_xmatch(
    incat: str,
    refcat: str,
    ra1: str="ra",
    dec1: str="dec",
    ra2: str="RAJ2000",
    dec2: str="DEJ2000",
    radius: float=2 / 60.0,
    enforce_min_srcs: Optional[int]=None,
) -> tuple[Table, Table]:
    """Given two source catalogues, perform an iterative serch to cross-match sources. 
    Successive rounds of cross-matching will refine an initial warp screen, which will
    improve the reliability of the cross match. 

    Args:
        incat (str): Input source catalogue to cross-match with. Should be for fits image supplied to de-warp. 
        refcat (str): Reference source catalogue whose positions are considered correct. 
        ra1 (str, optional): RA column of source in the incat table. Defaults to "ra".
        dec1 (str, optional): Dec column of source in the incat table. Defaults to "dec".
        ra2 (str, optional): RA column of sources in the refcat table. Defaults to "RAJ2000".
        dec2 (str, optional): Dec column of source in the refcat table. Defaults to "DEJ2000".
        radius (float, optional): Cross matching radius to use between catalogues. Defaults to 2/60.0.
        enforce_min_srcs (Optional[int], optional): The minimum number of cross-matched sources that have to be present. Below this the match is considered bad. Defaults to None.

    Raises:
        ValueError: Thrown when the number of cross-matched sources is below enfore_min_srcs

    Returns:
        tuple[Table, Table]: The incat table with corrected RA/Dec positions (based on warped screen) and the cross-matched sources used to derive warped screen
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyWarning)
        # check for incat/refcat as as strings, and load the file if it is
        incat = Table.read(incat)
        refcat = Table.read(refcat)

    # Casting to an array is used to avoid some behaviour when MackedColumns would fill invalid
    # values with a default. Seems a astropy / numpy change upstream made this an issue.
    mask = (~np.isfinite(np.array(incat[ra1]))) | (~np.isfinite(np.array(incat[dec1])))
    if np.sum(mask) > 0:
        logger.warning("NaN position detected in the target catalogue. Excluding. ")
        logger.warning(incat[mask])
        incat = incat[~mask]

    # The data attribute is needed in case either table carries with it a unit metavalue. If
    # it can not be parsed then the below will fail without the data, as SkyCoord ignores the
    # specified unit
    target_cat = SkyCoord(
        incat[ra1].data, incat[dec1].data, unit=(u.degree, u.degree), frame="icrs"
    )
    ref_cat = SkyCoord(
        refcat[ra2].data, refcat[dec2].data, unit=(u.degree, u.degree), frame="icrs"
    )

    center_pos = SkyCoord(
        circmean(target_cat.ra), np.mean(target_cat.dec), frame="icrs"
    )
    center = SkyOffsetFrame(origin=center_pos)

    logger.debug("Mean center posion {0}, {1}".format(center_pos.ra, center_pos.dec))

    tcat_offset = target_cat.transform_to(center)
    rcat_offset = ref_cat.transform_to(center)

    # crossmatch the two catalogs
    idx, dist, _ = tcat_offset.match_to_catalog_sky(rcat_offset)

    # accept only matches within radius
    distance_mask = np.where(dist.degree < radius)  # this mask is into tcat_offset
    match_mask = idx[distance_mask]  # this mask is into rcat_offset
    logger.info(f"Initial match found {len(match_mask)} matches")

    # calculate the ra/dec shifts
    dlon = rcat_offset.lon[match_mask] - tcat_offset.lon[distance_mask]
    dlat = rcat_offset.lat[match_mask] - tcat_offset.lat[distance_mask]

    # remake the offset catalogue with the bulk shift included
    tcat_offset = SkyCoord(
        tcat_offset.lon + np.mean(dlon), tcat_offset.lat + np.mean(dlat), frame=center
    )

    logger.info("Beginning iterative warped cross-match procedure...")
    # now do this again 3 more times but using the Rbf
    for i in range(3):
        logger.debug("Cross match iteration {0}".format(i))
        # crossmatch the two catalogs
        idx, dist, _ = tcat_offset.match_to_catalog_sky(rcat_offset)
        # accept only matches within radius
        distance_mask = np.where(dist.degree < radius)  # this mask is into cat
        match_mask = idx[distance_mask]  # this mask is into tcat_offset
        if len(match_mask) < 1:
            break

        # calculate the ra/dec shifts
        dlon = (
            rcat_offset.lon.degree[match_mask] - tcat_offset.lon.degree[distance_mask]
        )
        dlat = (
            rcat_offset.lat.degree[match_mask] - tcat_offset.lat.degree[distance_mask]
        )

        # use the following to make some models of the offsets
        dlonmodel = interpolate.Rbf(
            tcat_offset.lon.degree[distance_mask],
            tcat_offset.lat.degree[distance_mask],
            dlon,
            function="linear",
            smooth=3,
        )
        dlatmodel = interpolate.Rbf(
            tcat_offset.lon.degree[distance_mask],
            tcat_offset.lat.degree[distance_mask],
            dlat,
            function="linear",
            smooth=3,
        )

        # remake/update the tcat_offset with this new model.
        tcat_offset = SkyCoord(
            tcat_offset.lon
            + dlonmodel(tcat_offset.lon.degree, tcat_offset.lat.degree) * u.degree,
            tcat_offset.lat
            + dlatmodel(tcat_offset.lon.degree, tcat_offset.lat.degree) * u.degree,
            frame=center,
        )

    # final crossmatch to make the xmatch file
    idx, dist, _ = tcat_offset.match_to_catalog_sky(rcat_offset)
    # accept only matches within radius
    distance_mask = np.where(dist.degree < radius)  # this mask is into cat
    match_mask = idx[distance_mask]  # this mask is into tcat_offset
    # logger.info("Final mask {0}".format(len(match_mask)))
    xmatch = hstack([incat[distance_mask], refcat[match_mask]])

    logger.info(f"Final cross-match found {len(match_mask)} matches")
    if enforce_min_srcs is not None and len(match_mask) < enforce_min_srcs:
        logger.info(
            f"Fewer than {enforce_min_srcs} matches found, not creating warped file..."
        )
        raise ValueError(f"Fewer than {enforce_min_srcs} matches found. ")

    # return a warped version of the target catalogue and the final cross matched table
    tcat_corrected = tcat_offset.transform_to(target_cat)
    incat[ra1] = tcat_corrected.ra.degree
    incat[dec1] = tcat_corrected.dec.degree
    
    return incat, xmatch


def cli():
    parser = argparse.ArgumentParser()
    group1 = parser.add_argument_group("Warping input/output files")
    group1.add_argument(
        "--xm",
        dest="xm",
        type=Path,
        default=None,
        help="A .fits binary or VO table. The crossmatch between the reference and source catalogue.",
    )
    group1.add_argument(
        "--infits",
        dest="infits",
        type=str,
        default=None,
        help="The fits image(s) to be corrected; enclose in quotes for globbing.",
    )
    group1.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default=None,
        help='The suffix to append to rename the output (corrected) fits image(s); e.g., specifying "warp" will result in an image like image_warp.fits (no default; if not supplied, no correction will be performed).',
    )
    group2 = parser.add_argument_group("catalog column names")
    group2.add_argument(
        "--ra1",
        dest="ra1",
        type=str,
        default="ra",
        help="The column name for ra  (degrees) for source catalogue.",
    )
    group2.add_argument(
        "--dec1",
        dest="dec1",
        type=str,
        default="dec",
        help="The column name for dec (degrees) for source catalogue.",
    )
    group2.add_argument(
        "--ra2",
        dest="ra2",
        type=str,
        default="RAJ2000",
        help="The column name for ra  (degrees) for reference catalogue.",
    )
    group2.add_argument(
        "--dec2",
        dest="dec2",
        type=str,
        default="DEJ2000",
        help="The column name for dec (degrees) for reference catalogue.",
    )
    group3 = parser.add_argument_group("Other")
    group3.add_argument(
        "--plot",
        dest="plot",
        default=False,
        action="store_true",
        help="Plot the offsets and models (default = False)",
    )
    group3.add_argument(
        "-c",
        "--cores",
        default=1,
        type=int,
        help="Number of cores to instruct dask to use throughout processing",
    )
    group3.add_argument(
        '--overlap-factor',
        type=float,
        default=2,
        help="Factor mulptipled against size of fastest moving image axis to determine the number of pixels from neighbouring sub-blocks to include in image interpolation. To small and artefacts can be introduced. "
    )
    group3.add_argument(
        "--testimage",
        dest="testimage",
        default=False,
        action="store_true",
        help="Generate pixel-by-pixel delta_x, delta_y, and divergence maps (default = False)",
    )
    group3.add_argument(
        "--smooth",
        dest="smooth",
        default=300.0,
        type=float,
        help="Smoothness parameter to give to the radial basis function (default = 300 pix)",
    )
    group3.add_argument(
        "--signal",
        dest="sigcol",
        default=None,
        type=str,
        help="Column from which to get the signal for a signal-to-noise cut (e.g. peak_flux) (no default; if not supplied, cut will not be performed",
    )
    group3.add_argument(
        "--noise",
        dest="noisecol",
        default=None,
        type=str,
        help="Column from which to get the noise for a signal-to-noise cut (e.g. local_rms) (no default; if not supplied, cut will not be performed",
    )
    group3.add_argument(
        "--SNR",
        dest="SNR",
        default=10,
        type=float,
        help="Signal-to-noise ratio for a signal-to-noise cut (default = 10)",
    )
    group3.add_argument(
        "--nsrcs",
        default=None,
        type=int,
        help="Maximum number of sources used when constructing the distortion model. Default behaviour will use all available matches. ",
    )
    group3.add_argument(
        "--enforce-min-srcs",
        default=None,
        type=int,
        help="An exception is raised if there are fewer than this many cross-matched sources located in the internal cross-match procedure. ",
    )
    group3.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Provide extra logging throughout the procedure",
    )
    group4 = parser.add_argument_group("Crossmatching input/output files")
    group4.add_argument(
        "--incat",
        dest="incat",
        type=str,
        default=None,
        help="Input catalogue to be warped.",
    )
    group4.add_argument(
        "--refcat",
        dest="refcat",
        type=str,
        default=None,
        help="Input catalogue to be warped.",
    )
    group4.add_argument(
        "--xmcat",
        dest="xm",
        type=str,
        default=None,
        help="Output cross match catalogue",
    )
    group4.add_argument(
        "--corrected",
        dest="corrected",
        type=str,
        default=None,
        help="Output corrected version of input catalogue",
    )
    group5 = parser.add_argument_group("Information")
    group5.add_argument(
        "--cite",
        dest="cite",
        default=False,
        action="store_true",
        help="Print citation in BibTeX format.",
    )

    results = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    if results.verbose:
        logger.setLevel(logging.DEBUG)

    if results.cite is True:
        logger.info(__doc__)
        sys.exit()

    if results.incat is not None:
        if results.refcat is not None:
            corrected, xmcat = warped_xmatch(
                incat=results.incat,
                refcat=results.refcat,
                ra1=results.ra1,
                dec1=results.dec1,
                ra2=results.ra2,
                dec2=results.dec2,
                enforce_min_srcs=results.enforce_min_srcs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyWarning)
                xmcat.write(results.xm, overwrite=True)
                logger.info("Wrote {0}".format(results.xm))

            if results.corrected is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    corrected.write(results.corrected, overwrite=True)
                    logger.info("Wrote {0}".format(results.corrected))

    if results.infits is not None:
        if results.xm is not None:
            logger.info(f"Using {results.cores} throughout computation. ")
            cluster = LocalCluster(
                n_workers=results.cores, threads_per_worker=1
            )  # Launches a scheduler and workers locally

            with Client(
                cluster
            ) as client:  # Connect to distributed cluster and override default
                fnames = glob.glob(results.infits)
                # Use the first image to define the model
                offset_models = make_pix_models(
                    results.xm,
                    results.ra1,
                    results.dec1,
                    results.ra2,
                    results.dec2,
                    fnames[0],
                    results.plot,
                    results.smooth,
                    results.sigcol,
                    results.noisecol,
                    results.SNR,
                    max_sources=results.nsrcs,
                )
                if results.suffix is not None:
                    correct_images(
                        offset_models,
                        fnames,
                        results.suffix,
                        results.testimage,
                        overlap_factor=results.overlap_factor
                    )
                else:
                    logger.info(
                        "No output fits file specified via --suffix; not doing warping"
                    )
        else:
            logger.info(
                "Must specify a cross-matched catalogue via --xm to perform the warping."
            )


if __name__ == "__main__":
    cli()