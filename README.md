# Aframe O3 Search 
Repository for analysis of Aframe's search over the LVK's O3 observing run 

## Repository

- `aframe` : submodule containing the aframe software repository
- `aframe_o3_search` : core scripts for creating plots and tables
- `data` : Aframe data and other pipeline (IAS, ArewGW, OGC) used to create plots
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

Finally, a bash script is provided that will run each of the scripts in serial

```
bash run.sh
```