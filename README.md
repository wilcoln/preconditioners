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


## Experiments
Most experiments are in `src/preconditioners/`:

* The `generalization/` folder is for experiments studying the generalization capabilities of different optimization algorithms.

* The `linearization/` folder is for experiments studying how well an MLP is approximated by its linearization onced trained using NGD.

* The `eigenvalues/` folder is for experiments studying the eigenvalues of the Fisher information and preconditioned neural tangent kernel.

For example, an experiment can be run with the following command
```
python src/preconditioners/generalization/optimizer_benchmark.py --num_layers 3 --width 64
```

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
python -m pip install -e src/
```
or (depends if you prefer using pip or conda as your package manager)

```
conda develop src/
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
To generate documentation, run
```
scripts/makedoc.sh
```
The documentation entrypoint will be generated at `docs/_build/html/index.html`
