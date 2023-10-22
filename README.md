# Code for "Is this model reliable for everyone? Testing for strong calibration"

This folder contains all the code for reproducing experiments from the manuscript.

## General organization of the code:
* `detector.py`: File containing the definitions of the testing procedures.
* `subroup_testing.py`: File that actually tests for strong calibration on a provided dataset.
* `plot_*.py`: Plotting code
* `run_script.*`: Code for runnig jobs locally or on the cluster
* `aggregate_*.py` and `concat_files.py`: Code for aggregating results
* `*data*.py`: Data generation and preparation
* `common.py`: General helper functions
* Folders specify how experiments are run

## Reproducing experiments
We run experiments using the `nestly` and `scons` framework.

## Running unittests:
Code is in `test.py`. Run `python -m unittest`.

### Random notes:
Make an `_output` folder when running commands, since that is the default folder we output results in.
