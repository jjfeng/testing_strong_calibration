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
We run experiments using the `nestly` and `scons` framework. In particular,
* `scons simulation_power_mini` will run simulations from Section 3.
* `scons exp_zsfg` will perform the real-world data analysis in Section 4. Because we cannot share the electronic health record data used in this experiment, the code will not run. Nevertheless, one can see the code itself.
* `scons simulation_null` will run the simulations from Section 4 of the Appendix.

## Running unittests:
Code is in `test.py`. Run `python -m unittest`.
