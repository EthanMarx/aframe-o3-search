# Aframe O3 Search 
Repository for analysis of Aframe's search over the LVK's O3 observing run 

If you use or refer to these data products in your work, please cite https://doi.org/10.1103/1v7r-bkzs


## Repository

- `aframe` : submodule containing the aframe software repository
- `aframe_o3_search` : core scripts for creating plots and tables
- `data` : Aframe data and other pipeline (IAS, ArewGW, OGC) data used to create plots
- `paper` : Figures and tables from the paper

## Generating Figures and Tables
First, clone the repository, ensuring the also clone the aframe submodule

```
git clone git@github.com:EthanMarx/aframe-o3-search.git --recurse-submodules
```

Next, create a venv and install the software into it using your favorite tool, e.g.

```
python -m venv ./venv
source activate ./venv/bin/activate
pip install .
```

Finally, a bash script is provided that will run each of the scripts in serial.
The bash script will first download aframe and veto data products from zenodo,
apply vetos to the aframe triggers, and then create plots and tables from the paper.

Note that some scripts rely on data products from previous scripts, so
running them in order is important.

```
bash run.sh
```
