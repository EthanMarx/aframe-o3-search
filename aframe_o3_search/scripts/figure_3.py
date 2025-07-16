from jsonargparse import auto_cli
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
import aframe_o3_search.constants as c

plt.rcParams.update(
    {"text.usetex": True, "font.family": "Computer Modern", "font.size": 16}
)


def main(
    catalog_path: Path,
    outdir: Path,
):
    catalog = pd.read_hdf(catalog_path, key="events")
    mask = catalog["ifos"].apply(lambda x: {"H1", "L1"}.issubset(x))
    catalog = catalog[mask]
    non_det = catalog[catalog.aframe_far > c.LOW_SIGNIFICANCE_THRESHOLD]

    out_training_range = non_det.mass_2_source < 5
    non_det[out_training_range]
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))

    cmap = mpl.cm.viridis

    scatter = axes[0].scatter(
        catalog.chirp_mass_source,
        catalog.network_matched_filter_snr,
        c=catalog.aframe_p_astro,
        cmap=cmap,
    )
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label(r"$p_\mathrm{astro}$")

    axes[0].set_xlabel(r"Chirp mass $M_\odot$")
    axes[0].set_ylabel("Network SNR")

    axes[0].grid(alpha=0.5, linewidth=0.5)
    scatter = axes[1].scatter(
        catalog.mass_1_source,
        catalog.mass_2_source,
        c=catalog.aframe_p_astro,
        cmap=cmap,
    )
    axes[1].set_xlabel(r"Primary mass $M_\odot$")
    axes[1].set_ylabel(r"Secondary mass $M_\odot$")
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label(r"$p_\mathrm{astro}$")

    axes[1].grid(alpha=0.5, linewidth=0.5)
    plt.savefig(
        outdir / "figures" / "figure3.pdf", dpi=350, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
