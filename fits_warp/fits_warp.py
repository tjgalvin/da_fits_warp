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
import multiprocessing
import os
import sys
import warnings
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
from matplotlib import gridspec, pyplot
from scipy import interpolate
from scipy.interpolate import CloughTocher2DInterpolator
from tqdm import tqdm

from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(levelname)s:%(lineno)d %(message)s")
logger.setLevel(logging.INFO)

Array = Union[da.array, np.array]

def make_source_plot(
    fname: Path,
    hdr: dict[str, Any],
    cat_xy: np.ndarray,
    diff_xy: np.ndarray,
    cmap: str = "hsv",
) -> None:
    xmin, xmax = 0, hdr["NAXIS1"]
    ymin, ymax = 0, hdr["NAXIS2"]

    gx, gy = np.mgrid[
        xmin : xmax : (xmax - xmin) / 50.0, ymin : ymax : (ymax - ymin) / 50.0
    ]
    mdx = dxmodel(np.ravel(gx), np.ravel(gy))
    mdy = dymodel(np.ravel(gx), np.ravel(gy))
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

def create_divergence_map(fnames: list[Path], xy: Array, x: Array, y: Array, nx: int, ny: int) -> None:
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
):
    """
    Read a fits file which contains the crossmatching results for two catalogues.
    Catalogue 1 is the source catalogue (positions that need to be corrected)
    Catalogue 2 is the reference catalogue (correct positions)
    return rbf models for the ra/dec corrections
    :param fname: filename for the crossmatched catalogue
    :param ra1: column name for the ra degrees in catalogue 1 (source)
    :param dec1: column name for the dec degrees in catalogue 1 (source)
    :param ra2: column name for the ra degrees in catalogue 2 (reference)
    :param dec2: column name for the dec degrees in catalogue 2 (reference)
    :param fitsname: fitsimage upon which the pixel models will be based
    :param plots: True = Make plots
    :param smooth: smoothing radius (in pixels) for the RBF function
    :param max_sources: Maximum number of sources to include in the construction of the warping model (defaults to None, use all sources)
    :return: (dxmodel, dymodel)
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

    # cat_xy = da.from_array(
    #     imwcs.all_world2pix(list(zip(data[ra1], data[dec1])), 1),
    #     chunks=(100, 2)
    # )
    cat_xy = imwcs.all_world2pix(np.asarray(list(zip(data[ra1], data[dec1]))), 1)
    logger.debug(f"Formed cat_xy array: {cat_xy.shape=}")

    # ref_xy = da.from_array(
    #     imwcs.all_world2pix(list(zip(data[ra2], data[dec2])), 1),
    #     chunks=(100, 2)
    # )
    ref_xy = imwcs.all_world2pix(np.asarray(list(zip(data[ra2], data[dec2]))), 1)
    logger.debug(f"Formed ref_xy array: {ref_xy.shape=}")

    diff_xy = ref_xy - cat_xy
    logger.debug(f"Formed diff_xy array: {diff_xy.shape}")
    
    global dxmodel
    dxmodel = interpolate.Rbf(
        cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 0], function="linear", smooth=smooth
    )
    global dymodel
    dymodel = interpolate.Rbf(
        cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 1], function="linear", smooth=smooth
    )

    logger.info(f"Model created in {time() - start} seconds.")

    if plots:
        make_source_plot(fname, hdr, cat_xy, diff_xy)


def apply_interp(index, x1, y1, x2, y2, data):
    model = CloughTocher2DInterpolator(
        np.transpose([x1, y1]), np.ravel(data), fill_value=-1
    )
    # evaluate the model over this range
    newdata = model(x2, y2)
    return index, newdata


def correct_images(fnames, suffix, testimage=False, progress=False):
    """
    Read a list of fits image, and apply pixel-by-pixel corrections based on the
    given x/y models, which are global variables defined earlier.
    Interpolate back to a regular grid, and then write output files.
    :param fname: input fits file
    :param fout: output fits file
    :param progress: use tqdm to provide a progress bar
    :return: None
    """
    # Get co-ordinate system from first image
    # Do not open images at this stage, to save memory
    hdr = fits.getheader(fnames[0])
    nx = hdr["NAXIS1"]
    ny = hdr["NAXIS2"]

    xy = da.indices((ny, nx), dtype=np.float32, chunks=(500, 500))
    
    logger.debug(f"{xy.shape=}, type: {type(xy)}")
    
    logger.info(xy)

    x = da.blockwise(
        dxmodel,
        'jk', 
        xy[1,:,:],
        'jk',
        xy[0,:,:],
        'jk',
        dtype=np.float32
    )

    logger.info(x)
    
    y = da.blockwise(
        dymodel,
        'jk', 
        xy[1,:,:],
        'jk',
        xy[0,:,:],
        'jk',
        dtype=np.float32
    )
    
    logger.info(y)
    
    logger.info(f"Computing new x-coordinates")
    x = x.compute()
    logger.info(f"Finished x coordinates")
    
    logger.info(f"Computing new y-coordinates")
    y = y.compute()
    logger.info("Finished y coordinates")
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.imshow(
        x
    )
    ax2.imshow(
        y
    )
    fig.tight_layout()
    fig.savefig('example_x.pdf')
    
    if testimage is True:
        create_divergence_map(fnames, xy, x, y, nx, ny)
    
    # return
    
    for fname in fnames:
        fout = fname.replace(".fits", "_" + suffix + ".fits")
        im = fits.open(fname)
        im.writeto(fout, overwrite=True, output_verify="fix+warn")
        oldshape = im[0].data.shape
        data = da.array(im[0].data)
        unsqueezedshape = data.shape
        data = da.squeeze(data)
        squeezedshape = data.shape
        
        # Replace NaNs with zeroes because otherwise it breaks the interpolation
        nandices = da.isnan(data)
        data[nandices] = 0.0
        logger.info(f"interpolating {fname}")
        logger.debug(f"data shape {data.shape=}")
        logger.info(f"{data}")
    
        logger.debug(f"Rechunking positions")
        t_xy = da.reshape(
            da.transpose(
                da.array([x,y])
            ),
            (-1, 2)
        )
        logger.debug(f"Transpose xy: {t_xy.shape=}")
        logger.debug(f"{t_xy}")
        
        logger.debug(f"Rechunking ravel data")
        ravel_data = da.rechunk(
            np.ravel(data),
            chunks='auto'
        )
        logger.debug(f"{ravel_data=}")
        
        logger.info("all at once")
        # model = CloughTocher2DInterpolator(t_xy, ravel_data)
        models = da.map_overlap(
            CloughTocher2DInterpolator,
            t_xy,
            ravel_data, 
            dtype=np.float32,
            allow_rechunk=True,
            align_arrays=True            
        )
        logger.info(f"{models}")
        
        models = models.compute()
        return
        
        logger.info("Model createdm evaluating")
        newdata = model(xy[1, :], xy[0, :])
        
        # Float32 instead of Float64 since the precision is meaningless
        logger.info("int64 -> int32")
        data = newdata.astype(np.float32)
    
        data = newdata.reshape(squeezedshape)
        # NaN the edges by 10 pixels to avoid weird edge effects
        logger.info("blanking edges")
        data[0:10, :] = np.nan
        data[:, 0:10] = np.nan
        data[:, -10 : data.shape[0]] = np.nan
        data[-10 : data.shape[1], :] = np.nan
        
        # Re-apply any previous NaN mask to the data
        data[nandices] = np.nan
        im[0].data = data.reshape(oldshape)
        logger.info("saving...")
        im.writeto(fout, overwrite=True, output_verify="fix+warn")
        logger.info("wrote {0}".format(fout))
        # Explicitly delete potential memory hogs
        del im, data
    
    return


def warped_xmatch(
    incat=None,
    refcat=None,
    ra1="ra",
    dec1="dec",
    ra2="RAJ2000",
    dec2="DEJ2000",
    radius=2 / 60.0,
    enforce_min_srcs=None,
):
    """
    Create a cross match solution between two catalogues that accounts for bulk shifts and image warping.
    The warping is done in pixel coordinates, not sky coordinates.

    :param image: Fits image containing the WCS info for sky->pix conversion (Ideally the image which was used
                  to create incat.
    :param incat: The input catalogue which is to be warped during the cross matching process.
    :param ref_cat: The reference image which will remain unwarped during the cross matching process
    :param ra1, dec1: column names for ra/dec in the input catalogue
    :param ra2, dec2: column names for ra/dec in the reference catalogue
    :param radius: initial matching radius in degrees
    :param enforce_min_sources: If not None, an exception is raised if there are fewer than this many sources found
    :return:
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


if __name__ == "__main__":
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
        "--progress",
        default=False,
        action="store_true",
        help="Provide a progress bar for stages that are distributed into work-units",
    )
    group3.add_argument(
        "-v", "--verbose",
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
        help="logger.info citation in BibTeX format.",
    )

    results = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    if results.verbose:
        logger.setLevel(logging.DEBUG)

    if results.cite is True:
        logger.info(
            __doc__
        )
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
            cluster = LocalCluster(n_workers=8, threads_per_worker=1)  # Launches a scheduler and workers locally
            with Client(cluster) as client:  # Connect to distributed cluster and override default
                fnames = glob.glob(results.infits)
                # Use the first image to define the model
                make_pix_models(
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
                        fnames,
                        results.suffix,
                        results.testimage,
                        progress=results.progress,
                    )
                else:
                    logger.info(
                        "No output fits file specified via --suffix; not doing warping"
                    )
        else:
            logger.info(
                "Must specify a cross-matched catalogue via --xm to perform the warping."
            )
