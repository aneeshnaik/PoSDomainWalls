# PoSDomainWalls

## Summary

This code was used to generate the results in the article [Naik &amp; Burrage (2022)](https://arxiv.org/abs/2205.00712): "Dark sector domain walls could explain the observed planes of satellites".

The code was used to run two restricted N-body simulations of a large number of massless point particles representing Milky Way satellites, one simulation with the additional presence of a scalar domain wall (and accompanying fifth force), one without. Please see the paper for more details.

## Citation

Our code is freely available for use under the MIT License. For details, see LICENSE.

If using our code, please cite our paper: [Naik &amp; Burrage (2022)](https://arxiv.org/abs/2205.00712).


## Structure

This code is structured as follows:
- `/src` contains all of the 'source code', primarily the simulation code.
- `/simulations` contains runscripts for the simulations.
- `/figures` contains the plotting scripts used for the paper.

The various subsections below describe these components in further detail.

Note that the simulation outputs themselves are not saved in this directory. Instead, the runscripts expect a system environment variable called `POSDWDDIR` containing a path to an existing directory, and the simulation outputs are saved in this directory. The simulations are very quick to run and so anyone interested in analysing the outputs is invited to recreate out simulations via the runscripts, having first set the environment variable `POSDWDDIR`.

### `/src`

This directory contains all of the 'source code' underlying the project. There are 5 code files within this directory:

- `simulation.py`: contains the class Simulation, which sets up and runs the simulation.
- `constants.py`: definitions of various useful physical constants and unit conversions
- `nfw.py`: various functions associated with the NFW halo profile (used to represent the host galaxy)
- `satellites.py`: contains a function to sample initial satellite positions and velocities
- `wall.py`: fifth force acceleration due to the domain wall


### `/simulations`

This directory contains the runscripts for the two simulations (as well as a few extra simulations in the subdirectory `/extra_params`. 

### `/figures`

This directory contains the four figures from our paper (in .pdf format), along with the scripts used to generate them. The general file pattern is:
- `fign_x.py`
- `fign_x.pdf`

Here, `x` is some brief descriptor describing the figure. Each python script generates the corresponding figure.

These scripts give different example use cases for how to read and analyse the simulation outputs.


## Prerequisites

This section lists the various dependencies of our code. The version number in parenthesis is not (necessarily) a requirement, but simply the version of a given library we used at the time of publication. Earlier/later versions of a library could work equally well.

- `numpy` (1.20.3)
- `tqdm` (4.62.3)
- `matplotlib` (3.4.3)
- `scipy` (1.7.1)
