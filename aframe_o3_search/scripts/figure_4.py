import numpy as np

from ledger.injections import InjectionParameterSet
from ledger.events import EventSet, RecoveredInjectionSet
from aframe_o3_search.p_astro import fit_or_load_pastro
from aframe_o3_search.utils import filter_lal_warnings 
import matplotlib.pyplot as plt
from ledger.events import SECONDS_IN_YEAR
from pathlib import Path
from jsonargparse import auto_cli
import logging

filter_lal_warnings()

plt.rcParams.update(
    {"text.usetex": True, "font.family": "Computer Modern", "font.size": 16}
)


def main(
    background_path: Path,
    rejected_path: Path,
    foreground_path: Path,
    outdir: Path,
):
    log_format = "%(levelname)s - %(message)s"
    logging.basicConfig(format=log_format)
    logger = logging.getLogger("apply_vetos")
    logger.setLevel(logging.INFO)

    logger.info("Reading in background foreground and 0lag triggers") 
    rejected = InjectionParameterSet.read(rejected_path)
    background = EventSet.read(background_path)
    foreground = RecoveredInjectionSet.read(foreground_path)
    logger.info("Fitting or loading Pastro model")

    p_astro = fit_or_load_pastro(
        background, foreground, rejected, cache_dir=background_path.parent
    )
    logger.info("Creating plots")
    detection_statistics = np.linspace(-2, 14, 50)
    p_astros = p_astro(detection_statistics)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    hist, bins = np.histogram(
        background.detection_statistic, bins=1000, density=True
    )
    hist = np.cumsum(hist[::-1])[::-1]

    ax[0].step(
        (bins[:-1] + bins[1:]) / 2,
        hist,
        label="Background event density",
        where="mid",
    )
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
    det_stat_ticks = [
        background.threshold_at_far(far / SECONDS_IN_YEAR) for far in far_ticks
    ]

    # Set the ticks for the secondary axis
    ax2.set_xticks(det_stat_ticks)

    # Create labels for the ticks (showing the iFAR values)
    tick_labels = [f"10$^{{{i}}}$" for i in log_ifar_ticks]
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel(
        "Inverse false alarm rate (yrs)", labelpad=15, fontsize="16"
    )

    # Add a grid for readability
    ax[0].grid(True, which="both", linestyle="--", alpha=0.5)

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
    det_stat_ticks = [
        background.threshold_at_far(far / SECONDS_IN_YEAR) for far in far_ticks
    ]

    # Set the ticks for the secondary axis
    ax2.set_xticks(det_stat_ticks)

    # Create labels for the ticks (showing the iFAR values)
    tick_labels = [f"10$^{{{i}}}$" for i in log_ifar_ticks]
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel(
        "Inverse false alarm rate (yrs)", labelpad=15, fontsize="16"
    )

    # Add a grid for readability
    ax[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(outdir / "figures" / "figure4.pdf", bbox_inches="tight")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
