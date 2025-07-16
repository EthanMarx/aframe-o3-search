# Aframe O3 Search 
Repository for analysis of Aframe's search over the LVK's O3 observing run 

If you use or refer to these data products in your work, please cite https://doi.org/10.1103/1v7r-bkzs

See also https://zenodo.org/records/15922679, where the data is made publicly available

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

## Reading Data

The trigger files are stored in hdf5 format. If you have installed this project as described above, 
you'll have access to python classes for easily working with the data:

```python
from ledger.events import EventSet, RecoveredInjectionSet

# triggers from the actual search
zero_lag = EventSet.read("./data/aframe/post_veto/0lag.hdf5")

# have access to properties of events, e.g.
zero_lag.detection_time, zero_lag.detection_statistic 

# set of injections used to evaluate aframe sensitivity
foreground = RecoveredInjectionSet.read("./data/aframe/post_veto/foreground.hdf5")

# noise events accumulated through timeslide,s and used to evaluate FAR of events
background = EventSet.read("./data/aframe/post_veto/background.hdf5")
```

Otherwise, these files can be opened with e.g. `h5py`
