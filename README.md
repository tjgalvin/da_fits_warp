# da_fits_warp
Aim: Warp catalogues and images to remove the distorting affect of the ionosphere. This version uses Dask arrays to handle code parallelisation. 

Authors: Natasha Hurley-Walker, Paul Hancock, Tim Galvin

## Usage
```
usage: da_fits_warp [-h] [--xm XM] [--infits INFITS] [--suffix SUFFIX] [--ra1 RA1] [--dec1 DEC1] [--ra2 RA2] [--dec2 DEC2] [--plot] [-c CORES] [--overlap-factor OVERLAP_FACTOR]
                    [--testimage] [--smooth SMOOTH] [--signal SIGCOL] [--noise NOISECOL] [--SNR SNR] [--nsrcs NSRCS] [--enforce-min-srcs ENFORCE_MIN_SRCS] [-v] [--incat INCAT]
                    [--refcat REFCAT] [--xmcat XM] [--corrected CORRECTED] [--cite]

options:
  -h, --help            show this help message and exit

Warping input/output files:
  --xm XM               A .fits binary or VO table. The crossmatch between the reference and source catalogue.
  --infits INFITS       The fits image(s) to be corrected; enclose in quotes for globbing.
  --suffix SUFFIX       The suffix to append to rename the output (corrected) fits image(s); e.g., specifying "warp" will result in an image like image_warp.fits (no default; if not
                        supplied, no correction will be performed).

catalog column names:
  --ra1 RA1             The column name for ra (degrees) for source catalogue.
  --dec1 DEC1           The column name for dec (degrees) for source catalogue.
  --ra2 RA2             The column name for ra (degrees) for reference catalogue.
  --dec2 DEC2           The column name for dec (degrees) for reference catalogue.

Other:
  --plot                Plot the offsets and models (default = False)
  -c CORES, --cores CORES
                        Number of cores to instruct dask to use throughout processing
  --overlap-factor OVERLAP_FACTOR
                        Factor mulptipled against size of fastest moving image axis to determine the number of pixels from neighbouring sub-blocks to include in image interpolation. To
                        small and artefacts can be introduced.
  --testimage           Generate pixel-by-pixel delta_x, delta_y, and divergence maps (default = False)
  --smooth SMOOTH       Smoothness parameter to give to the radial basis function (default = 300 pix)
  --signal SIGCOL       Column from which to get the signal for a signal-to-noise cut (e.g. peak_flux) (no default; if not supplied, cut will not be performed
  --noise NOISECOL      Column from which to get the noise for a signal-to-noise cut (e.g. local_rms) (no default; if not supplied, cut will not be performed
  --SNR SNR             Signal-to-noise ratio for a signal-to-noise cut (default = 10)
  --nsrcs NSRCS         Maximum number of sources used when constructing the distortion model. Default behaviour will use all available matches.
  --enforce-min-srcs ENFORCE_MIN_SRCS
                        An exception is raised if there are fewer than this many cross-matched sources located in the internal cross-match procedure.
  -v, --verbose         Provide extra logging throughout the procedure

Crossmatching input/output files:
  --incat INCAT         Input catalogue to be warped.
  --refcat REFCAT       Input catalogue to be warped.
  --xmcat XM            Output cross match catalogue
  --corrected CORRECTED
                        Output corrected version of input catalogue

Information:
  --cite                Print citation in BibTeX format.
```

## Bugs/Questions
Please use the GitHub issue tracker to submit bug reports, feature requests, or questions.

## Credit
If you use fits_warp in your work please Cite [Hurley-Walker and Hancock 2018](http://adsabs.harvard.edu/abs/2018A%26C....25...94H).

This `da_fits_warp` is an update to the original [`fits_warp`](https://github.com/nhurleywalker/fits_warp) implementation, with the most major change being the use of `dask` to handle parallelism. 