"""
Generate latex tables for aframe events from the gwtc3 catalog
"""

import numpy as np
from ledger.injections import InjectionParameterSet
from ledger.events import EventSet, RecoveredInjectionSet

from aframe_o3_offline.catalogs import validate_catalog, build_gwtc3_catalog, recover
import aframe_o3_offline.constants as c
from aframe_o3_offline.p_astro import fit_or_load_pastro
from aframe_o3_offline.utils import build_table
import logging
from pathlib import Path
from num2tex import num2tex


def main(
    data_dir: Path,
    out_dir: Path,
    low_sig_threshold: float = 2 * 365
):


    logging.basicConfig()
    logger = logging.getLogger('make_gwtc3_table')
    logger.setLevel(logging.INFO)

    # read in segments analyzed by aframe
    train_segments = np.loadtxt(data_dir / "train_segments.txt", dtype=float)[:, 1:3]
    test_segments = np.loadtxt(data_dir / "test_segments.txt", dtype=float)[:, 1:3]
    segments = np.concatenate([train_segments, test_segments])

    # read in background, foreground, rejected parameters 
    # and 0lag triggers from aframe
    rejected = InjectionParameterSet.read(data_dir / "rejected-parameters.hdf5")
    size = len(rejected)

  
    background = EventSet.read(data_dir / "background.hdf5")
    zero_lag = EventSet.read(data_dir / "0lag.hdf5")
    foreground = RecoveredInjectionSet.read(data_dir / "foreground.hdf5")

    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True, parents=True)
    # build catalog, and then filter for events that were detected by 
    # (at least) the given interferometers; Also append an analyzed column
    # that determines if aframe actually analyzed the event
    catalog = build_gwtc3_catalog()
    catalog = validate_catalog(catalog, segments, c.PSD_LENGTH, c.BUFFER)
    in_burn_in = catalog[catalog.in_burn_in]
    
    with open(out_dir / "gwtc3-burn-in.txt", "w") as f:
        for time in in_burn_in.time.values:
            f.write(f"{time}\n")

    logger.info("There are {} events in the catalog".format(len(catalog)))

    in_train_mask = catalog.time < c.TRAIN_STOP
    logger.info("There are {} events in the training set".format(sum(in_train_mask)))
    catalog = catalog.assign(in_train=in_train_mask)

    mask = catalog["ifos"].apply(lambda x: set(c.IFOS).issubset(x))
    logger.info("There are {} events detected by the given interferometers".format(mask.sum()))

    aframe_analyzed = catalog["aframe_analyzed"]
    logger.info("There are {} events analyzed by aframe".format(aframe_analyzed.sum()))
    logger.info("There are {} events analyzed by aframe that werent H1, L1 triggers".format((aframe_analyzed & ~mask).sum()))


    p_astro = fit_or_load_pastro(
        background, 
        foreground, 
        rejected,
        data_dir,
    )

    # match up gwtc3 catalog with aframe triggers
    # and restrict only to events that were analyzed by aframe
    recovered = recover(catalog, zero_lag, background)
    analyzed = recovered[recovered.aframe_analyzed]

    # add p_astro to the analyzed events
    analyzed = analyzed.assign(aframe_p_astro=p_astro(analyzed.aframe_detection_statistic.values))

    # append asterisks to events that were in the training set
    analyzed.loc[:, "event"] = analyzed.event.apply(lambda x: x.split("-")[0].replace("_", "\_"))
    analyzed.loc[analyzed.in_train, "event"] = analyzed.loc[analyzed.in_train, "event"] + "$^{*}$"


    # only include events that were more significant than 2 per day
    analyzed = analyzed.sort_values("aframe_detection_statistic", ascending=False)
    found = analyzed[analyzed.aframe_far < low_sig_threshold]

    def mass_formatter(row, mass_base):
        # Get column names based on mass_base
        mass_col = f"{mass_base}_source"
        upper_col = f"{mass_base}_source_upper"
        lower_col = f"{mass_base}_source_lower"
        
        # Get values
        value = row[mass_col]
        upper_unc = row[upper_col]
        lower_unc = row[lower_col]
    
        # Format with asymmetric uncertainties
        return f"${value}^{{+{abs(upper_unc)}}}_{{{-abs(lower_unc)}}}$"


    def aframe_far_formatter(x):
        lowest = background.min_far
        if x <= background.min_far:
            return f"$\leq {lowest:.2f}$"
        return "{:.2f}".format(x)

    def far_formatter(x):
        if x < 0.01:
            return "${:.2e}$".format(num2tex(x, precision=2))
        else:
            return "{:.2f}".format(num2tex(x, precision=2))


    # build the latex table and write to file
    found['mass_1_latex'] = found.apply(lambda row: mass_formatter(row, "mass_1"), axis=1)
    found['mass_2_latex'] = found.apply(lambda row: mass_formatter(row, "mass_2"), axis=1)
    columns = ["event", "aframe_far", "far", "aframe_p_astro", "p_astro", "network_matched_filter_snr", "mass_1_latex", "mass_2_latex"]
    table = found[columns]


    mask = table.aframe_p_astro >= 0.5
    significant = table[mask]
    non_significant = table[~mask]
    
    significant.index = range(1, len(significant) + 1)
    non_significant.index = range(1, len(non_significant) + 1)

    latex_sig = significant.to_latex(
        columns=columns,
        header = ["Event", "Aframe FAR $\mathrm{{yr}}^{{-1}}$", "FAR", "Aframe $p_\mathrm{{astro}}$", "$p_\mathrm{{astro}}$", "Network SNR", "$m_1 (M_{{\odot}})$", "$m_2 (M_{{\odot}})$"],
        formatters = {
            "aframe_far": aframe_far_formatter, 
            "far": far_formatter,         
        },
        float_format="%.2f",
        column_format="l" + len(columns) * "c",
        index=True
    )

    sig_label = "table:gwtc3_significant"
    sig_caption = "Events reported in GWTC-3 detected by \\textt{Aframe} with $p_{astro} > 0.5$. Event's marked with an asterisk occurred during the training period."
    latex_sig = build_table(latex_sig, sig_caption, sig_label, len(columns))


    with open(table_dir / "gwtc3-significant.tex", "w") as f:
        f.write(latex_sig)    

    # pick out a couple event to test pe on
    pe_events = ["GW191109\_010717", "GW191204\_171526"]

    times_dir = out_dir / "trigger_times"
    times_dir.mkdir(exist_ok=True, parents=True)

    for event in pe_events:
        time = found[found.event == event].time.values
        event = event.replace("\\", "")
        bilby_dir = out_dir / "bilby" / event
        bilby_dir.mkdir(exist_ok=True, parents=True)
        with open(bilby_dir / f'trigger_time.txt', "w") as f:
            arg = np.argmin(np.abs(zero_lag.detection_time - time))
            aframe_time = zero_lag.detection_time[arg]
            f.write(f"{aframe_time}\n")

    latex_non_sig = non_significant.to_latex(
        columns=columns,
        header = ["Event", "Aframe FAR $\mathrm{{yr}}^{{-1}}$", "FAR", "Aframe $p_\mathrm{{astro}}$", "$p_\mathrm{{astro}}$", "Network SNR", "$m_1 (M_{{\odot}})$", "$m_2 (M_{{\odot}})$"],
        formatters = {
            "aframe_far": aframe_far_formatter, 
            "far": far_formatter,         
        },
        float_format="%.2f",
        index=True
    )

    non_sig_label = "table:gwtc3_non_significant"
    non_sig_caption = "Events reported in GWTC-3 Aframe does not detect with $p_{astro} > 0.5$. Event's marked with an asterisk occurred during the training period."
    latex_non_sig = build_table(latex_non_sig, non_sig_caption, non_sig_label, len(columns))

    with open(table_dir / "gwtc3-non-significant.tex", "w") as f:
        f.write(latex_non_sig)      

     
if __name__ == "__main__":
    data_dir = Path("/home/ethan.marx/projects/aframe-o3-offline/data/aframe/vetoed/")
    out_dir = Path("/home/ethan.marx/projects/aframe-o3-offline/production_data/")
    out_dir.mkdir(exist_ok=True)
    main(data_dir, out_dir)