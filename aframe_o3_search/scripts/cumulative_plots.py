import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal

import gwpy
import numpy as np
import pandas as pd

from aframe_o3_offline.catalogs import validate_catalog, build_gwtc3_catalog, recover
import aframe_o3_offline.constants as c
from aframe_o3_offline.utils import build_table, time_to_gwtc_event, gwtc_event_to_time
from ledger.injections import InjectionParameterSet
from ledger.events import EventSet, RecoveredInjectionSet
from aframe_o3_offline.p_astro import fit_or_load_pastro
from astropy.table import Table
from datetime import datetime
import matplotlib.pyplot as plt
from ledger.events import SECONDS_IN_YEAR
from pathlib import Path


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 16
})

def main(
    background_path: Path,
    rejected_path: Path,
    foreground_path: Path,
    outdir: Path,
):
    
    rejected = InjectionParameterSet.read(rejected_path)
    background = EventSet.read(background_path)
    foreground = RecoveredInjectionSet.read(foreground_path)
    p_astro = fit_or_load_pastro(background, foreground, rejected, cache_dir=background_path.parent)
    detection_statistics = np.linspace(-2, 14, 50)
    p_astros = p_astro(detection_statistics)

    data_dir = Path("/home/ethan.marx/projects/aframe-o3-offline/production_data/figures/")
    data_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    hist, bins = np.histogram(
        background.detection_statistic, bins=1000, density=True
    )
    hist = np.cumsum(hist[::-1])[::-1]

    ax[0].step((bins[:-1] + bins[1:]) / 2, hist, label="Background event density", where="mid")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Background event density")
    ax[0].set_xlabel("Detection statistic")
    ax[0].set_xlim([-2, 9.5])
    ax[0].set_ylim([1e-7, 1e2])
    # Create a secondary x-axis
    ax2 = ax[0].twiny()
    x_lim = ax[0].get_xlim()
    ax2.set_xlim(x_lim)


    log_ifar_ticks = list(range(-6, 2))[::2] + [1]
    ifar_ticks = [10**i for i in log_ifar_ticks]

    far_ticks = [1.0 / ifar for ifar in ifar_ticks]
    # Use background.threshold_at_far() to get detection statistics for each FAR
    det_stat_ticks = [background.threshold_at_far(far / SECONDS_IN_YEAR) for far in far_ticks]


    # Set the ticks for the secondary axis
    ax2.set_xticks(det_stat_ticks)

    # Create labels for the ticks (showing the iFAR values)
    tick_labels = [f"10$^{{{i}}}$" for i in log_ifar_ticks]
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel("Inverse false alarm rate (yrs)", labelpad=15, fontsize="16")

    # Add a grid for readability
    ax[0].grid(True, which='both', linestyle='--', alpha=0.5)

    ax[1].plot(detection_statistics, p_astros, label="p$_{astro}$")
    ax[1].set_xlabel("Detection statistic")
    ax[1].set_ylabel("p$_{astro}$")
    # Create a secondary x-axis
    ax2 = ax[1].twiny()
    x_lim = ax[1].get_xlim()
    ax2.set_xlim(x_lim)
    ax[1].set_ylim([0, 1.0])
    ax[1].set_xlim([2, 12])

    log_ifar_ticks = list(range(-6, 2))[::2] + [1]
    ifar_ticks = [10**i for i in log_ifar_ticks]

    far_ticks = [1.0 / ifar for ifar in ifar_ticks]
    # Use background.threshold_at_far() to get detection statistics for each FAR
    det_stat_ticks = [background.threshold_at_far(far / SECONDS_IN_YEAR) for far in far_ticks]

    # Set the ticks for the secondary axis
    ax2.set_xticks(det_stat_ticks)

    # Create labels for the ticks (showing the iFAR values)
    tick_labels = [f"10$^{{{i}}}$" for i in log_ifar_ticks]
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel("Inverse false alarm rate (yrs)", labelpad=15, fontsize="16")

    # Add a grid for readability
    ax[1].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(outdir / "pastro_cum_background.pdf", bbox_inches="tight")


if __name__ == "__main__":
    data_dir = Path("/home/ethan.marx/projects/aframe-o3-offline/data/aframe/vetoed/")
    background_path = data_dir / "background.hdf5"
    rejected_path = data_dir / "rejected-parameters.hdf5"
    foreground_path = data_dir / "foreground.hdf5" 
    outdir = Path("/home/ethan.marx/projects/aframe-o3-offline/production_data/figures/")
    main(background_path, rejected_path, foreground_path, outdir)