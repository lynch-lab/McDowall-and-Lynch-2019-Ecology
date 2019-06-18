### DOI for repo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3247945.svg)](https://doi.org/10.5281/zenodo.3247945)

### Installation
Follow the instructions at https://www.scipy.org/install.html to install numpy.

git clone https://github.com/phil-mcdowall/hexibm.git

### Running
Model takes 3 command line arguments.

To see help on arguments run `python model.py -h`

 Arguments passed to script from stdin are unnamed arguments.
 The first argument is the variance on reproductive rates, the second argument is a subdirectory of the current directory in which
 to store output, and the final argument is the number of cores to use. The number of cores should be at most the total number of cores available -1.

`model.py 0.5 output 1`
 
 runs the model with variance 0.5 in repro rates, outputs to the subdirectory 'output' and uses 1 core.
