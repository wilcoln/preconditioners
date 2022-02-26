# Preconditioners

The goal of this research project is to design preconditioners with good generalization properties.

## Folder structure

* `docs/`
Auto-Generated & manual code documentation.

* `data/`
Contains data original & intermediate synthetic data.

* `notebooks/`
All the notebooks, avoid defining functions here.

* `preconditioners/`
Python package containing the main code for this research.

* `results/`
For results, e.g. tables (csv files), and plots (images)

* `scripts/`
Contains bash scripts, this scripts might just be launchers for python scripts defined in the main package.
Useful for running long experiments for example.

## Installation

### Create the conda environment by running
```
conda env create -f env.yml
conda activate preconditioners
python -m ipykernel install --user --name preconditioners
```

After installing, every time you want to work with this project run `conda activate preconditioners` and after you 
finish, run `conda deactivate`.

### Package installation
To install the package go to the root of this directory and run
```
conda develop preconditioners/
```
Now every time you want to run some piece of code you can import it from `preconditioners`.
Alternatively, you can add the `preconditioners` package to your python path, by adding these line at the top of 
your python script or jupyter notebook.
```python
import sys
sys.path.insert(0, '/path/to/preconditioners')
```

###
To run tests, run
```
pytest --pyargs preconditioners/tests/*
```
