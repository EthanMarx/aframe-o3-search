"""
Generate latex tables for aframe events from the gwtc3 catalog
"""

import re
import pandas as pd
import numpy as np
from ledger.injections import InjectionParameterSet
from ledger.events import EventSet, RecoveredInjectionSet
from astropy.table import Table
from aframe_o3_search.catalogs import (
    validate_catalog,
    build_gwtc3_catalog,
    recover,
)
import aframe_o3_search.constants as c
from aframe_o3_search.p_astro import fit_or_load_pastro
from aframe_o3_search.utils import build_table, filter_lal_warnings
import logging
from pathlib import Path
from num2tex import num2tex
from aframe_o3_search.utils import gwtc_event_to_time
from jsonargparse import auto_cli

def add_brackets_to_latex(latex_str):
    latex_str = latex_str.replace("{", "")
    latex_str = latex_str.replace("}", "")

    # Add brackets after _ and + to enclose the numbers
    latex_str = re.sub(r"_(\-?\d+\.?\d*)", r"_{\1}", latex_str)
    latex_str = re.sub(r"\^(\S+)", r"^{\1}", latex_str)
    latex_str = f"${latex_str}$"
    return latex_str


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
    return rf"${value}^{{+{abs(upper_unc)}}}_{{{-abs(lower_unc)}}}$"


def far_formatter(x):
    if x < 0.01:
        return "${:.2e}$".format(num2tex(x, precision=2))
    else:
        return "{:.2f}".format(num2tex(x, precision=2))


def main(
    foreground_path: Path,
    zero_lag_path: Path,
    background_path: Path,
    rejected_path: Path,
    segments_dir: Path,
    out_dir: Path,
    data_dir: Path,
    pipeline_data_dir: Path,
    low_sig_threshold: float = 2 * 365,
    recovery_dt: float = 1.0,
):
    filter_lal_warnings()
    logging.basicConfig()
    logger = logging.getLogger("make_tables")
    logger.setLevel(logging.INFO)

    # master dataframes that will
    # ultimately be used to create tables

    columns = [
        "event",
        "catalog",
        "aframe_far",
        "far",
        "aframe_p_astro",
        "p_astro",
        "mass_1",
        "mass_2",
    ]
    significant = pd.DataFrame(columns=columns)
    non_significant = pd.DataFrame(columns=columns)

    # read in segments analyzed by aframe
    train_segments = np.loadtxt(segments_dir / "train_segments.txt", dtype=float)[
        :, 1:3
    ]
    test_segments = np.loadtxt(segments_dir / "test_segments.txt", dtype=float)[
        :, 1:3
    ]
    segments = np.concatenate([train_segments, test_segments])

    # read in background, foreground, rejected parameters
    # and 0lag triggers from aframe
    rejected = InjectionParameterSet.read(rejected_path)

    background = EventSet.read(background_path)
    zero_lag = EventSet.read(zero_lag_path)
    foreground = RecoveredInjectionSet.read(foreground_path)

    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True, parents=True)

    # build gwtc-3 catalog, and then filter for events that were detected by
    # (at least) the given interferometers; Also append an analyzed column
    # that determines if aframe actually analyzed the event
    logger.info("Building GWTC3 catalog")
    catalog = build_gwtc3_catalog()

    # special case: drop 190916 and 190926 since these
    # were first actually reported by OGC, and will be added when
    # appending the OGC events
    catalog = validate_catalog(catalog, segments, c.PSD_LENGTH, c.BUFFER)
    catalog["event"] = catalog["event"].apply(lambda x: x.split("-")[0])
    catalog = catalog[catalog["event"] != "GW190916_200658"]
    catalog = catalog[catalog["event"] != "GW190926_050336"]

    logger.info("There are {} events in the catalog".format(len(catalog)))

    in_train_mask = catalog.time < c.TRAIN_STOP
    logger.info(
        "There are {} events in the training set".format(sum(in_train_mask))
    )
    catalog = catalog.assign(in_train=in_train_mask)

    mask = catalog["ifos"].apply(lambda x: set(c.IFOS).issubset(x))
    logger.info(
        "There are {} events detected by the given interferometers".format(
            mask.sum()
        )
    )

    aframe_analyzed = catalog["aframe_analyzed"]
    logger.info(
        "There are {} events analyzed by aframe".format(aframe_analyzed.sum())
    )
    logger.info(
        "There are {} events analyzed by aframe that werent H1, L1 triggers".format(
            (aframe_analyzed & ~mask).sum()
        )
    )

    p_astro = fit_or_load_pastro(
        background,
        foreground,
        rejected,
        data_dir,
    )

    gpstimes = []

    # match up gwtc3 catalog with aframe triggers
    # and restrict only to events that were analyzed by aframe
    recovered = recover(catalog, zero_lag, background)

    recovered = recovered.assign(
        aframe_p_astro=p_astro(recovered.aframe_detection_statistic.values)
    )
    recovered.to_hdf(out_dir / "gwtc3.hdf5", key="events")

    analyzed = recovered[recovered.aframe_analyzed]

    gpstimes.extend(recovered.time)

    # add p_astro to the analyzed events

    # append asterisks to events that were in the training set
    analyzed.loc[:, "event"] = analyzed.event.apply(
        lambda x: x.replace("_", r"\_")
    )
    analyzed.loc[analyzed.in_train, "event"] = (
        analyzed.loc[analyzed.in_train, "event"] + "$^{*}$"
    )

    # only include events that were more significant than 2 per day
    analyzed = analyzed.sort_values(
        "aframe_detection_statistic", ascending=False
    )
    found = analyzed[analyzed.aframe_far < low_sig_threshold]
    found["mass_1"] = found.apply(
        lambda row: mass_formatter(row, "mass_1"), axis=1
    )
    found["mass_2"] = found.apply(
        lambda row: mass_formatter(row, "mass_2"), axis=1
    )
    found["catalog"] = "GWTC-3"
    table = found[columns]

    mask = table.aframe_p_astro >= 0.5
    significant = pd.concat([significant, table[mask]], ignore_index=True)
    non_significant = pd.concat(
        [non_significant, table[~mask]], ignore_index=True
    )

    # ogc
    logger.info("Building OGC catalog")

    # read in the new candidates reported by OGC
    ogc: pd.DataFrame = Table.read(
        pipeline_data_dir / "ogc" / "new.tex", format="latex"
    ).to_pandas()

    # sort the new ogc candidates table
    # and aframe events by gpstime
    ogc.sort_values("gpstime", inplace=True)
    args = zero_lag.detection_time.argsort()
    zero_lag = zero_lag[args]

    # filter out aframe events that correspond to the ogc events
    # and add the aframe p astro to the table
    aframe_ogc_detection_statistic = []
    for time in ogc.gpstime:
        arg = np.argmin(np.abs(zero_lag.detection_time - time))
        if np.abs(time - zero_lag.detection_time[arg]) < recovery_dt:
            gpstimes.append(zero_lag.detection_time[arg])
            aframe_ogc_detection_statistic.append(
                zero_lag.detection_statistic[arg]
            )
        else:
            aframe_ogc_detection_statistic.append(-2)
    aframe_ogc_detection_statistic = np.array(aframe_ogc_detection_statistic)

    ogc = ogc.assign(aframe_p_astro=p_astro(aframe_ogc_detection_statistic))
    ogc = ogc.assign(aframe_far=background.far(aframe_ogc_detection_statistic))
    ogc = ogc.assign(far=1 / ogc.ifar)
    ogc = ogc[ogc.aframe_far < c.LOW_SIGNIFICANCE_THRESHOLD]
    ogc["catalog"] = "OGC"
    ogc_columns = [
        "event",
        "catalog",
        "aframe_far",
        "far",
        "aframe_p_astro",
        "p_astro",
        "m_1",
        "m_2",
    ]
    ogc = ogc[ogc_columns]
    ogc.rename(
        columns={
            ogc: col for ogc, col in zip(ogc_columns, columns, strict=False)
        },
        inplace=True,
    )

    mask = ogc.aframe_p_astro >= 0.5
    significant = pd.concat(
        [significant, ogc[mask]],
        ignore_index=True,
    )
    non_significant = pd.concat(
        [non_significant, ogc[~mask]], ignore_index=True
    )

    # ares gw

    logger.info("Building AresGW catalog")

    # read in the new candidates reported by AresGW
    ares: pd.DataFrame = Table.read(
        pipeline_data_dir / "aresgw" / "new.tex", format="latex"
    ).to_pandas()

    # sort the new ares candidates table
    # and aframe events by gpstime
    ares.sort_values("gpstime", inplace=True)

    # filter out aframe events that correspond to the aresgw events
    # and add the aframe p astro to the table
    mask = (
        np.abs(zero_lag.detection_time - ares.gpstime.values[:, None])
        < recovery_dt
    )
    mask = mask.any(axis=0)
    ares_gw_candidates = zero_lag[mask]

    gpstimes.extend(ares_gw_candidates.detection_time)

    # add aframe p astro and far to the ares gw candidate catalog
    ares = ares.assign(
        aframe_p_astro=p_astro(ares_gw_candidates.detection_statistic)
    )
    ares = ares.assign(
        aframe_far=background.far(ares_gw_candidates.detection_statistic)
    )

    # filter for events found by aframe that are above the 2 per day threshold set by GWTC-3
    ares = ares[ares.aframe_far < c.LOW_SIGNIFICANCE_THRESHOLD]

    ares["catalog"] = "AresGW"
    ares_columns = [
        "event",
        "catalog",
        "aframe_far",
        "far",
        "aframe_p_astro",
        "p_astro",
        "m_1",
        "m_2",
    ]
    ares = ares[ares_columns]
    ares.rename(
        columns={
            ares: col for ares, col in zip(ares_columns, columns, strict=False)
        },
        inplace=True,
    )

    mask = ares.aframe_p_astro > 0.5
    significant = pd.concat([significant, ares[mask]], ignore_index=True)
    non_significant = pd.concat(
        [non_significant, ares[~mask]], ignore_index=True
    )

    # IAS
    logger.info("Building IAS catalog")
    # read in the new candidates reported by IAS
    # from o3a and b and combine them
    ias_o3a: pd.DataFrame = Table.read(
        pipeline_data_dir / "ias" / "new_o3a.latex", format="latex"
    ).to_pandas()
    ias_o3a["m_1"] = ias_o3a["m_1"].apply(add_brackets_to_latex)
    ias_o3a["m_2"] = ias_o3a["m_2"].apply(add_brackets_to_latex)

    ias_o3b = (
        Table.read(pipeline_data_dir / "ias" / "new_o3b.latex", format="latex")
        .to_pandas()
        .drop(columns=["ifar bank"])
    )
    ias = pd.concat([ias_o3a, ias_o3b], ignore_index=True)
    ias = ias.assign(gpstime=ias.event.apply(gwtc_event_to_time).astype(float))

    # sort the new ares candidates table
    # and aframe events by gpstime
    ias.sort_values("gpstime", inplace=True)
    args = zero_lag.detection_time.argsort()
    zero_lag = zero_lag[args]

    # filter out aframe events that correspond to the ias events
    # and add the aframe p astro to the table
    mask = np.abs(zero_lag.detection_time - ias.gpstime.values[:, None]) < 4
    mask = mask.any(axis=0)
    ias_gw_candidates = zero_lag[mask]
    gpstimes.extend(zero_lag[mask].detection_time)
    ias = ias.assign(
        aframe_p_astro=p_astro(ias_gw_candidates.detection_statistic)
    )
    ias = ias.assign(
        aframe_far=background.far(ias_gw_candidates.detection_statistic)
    )
    ias = ias.assign(far=1 / ias.ifar)
    ias = ias[ias.aframe_far < c.LOW_SIGNIFICANCE_THRESHOLD]
    ias["catalog"] = "IAS"
    ias_columns = [
        "event",
        "catalog",
        "aframe_far",
        "far",
        "aframe_p_astro",
        "p_astro",
        "m_1",
        "m_2",
    ]
    ias = ias[ias_columns]
    ias.rename(
        columns={
            ias: col for ias, col in zip(ias_columns, columns, strict=False)
        },
        inplace=True,
    )
    mask = ias.aframe_p_astro >= 0.5
    significant = pd.concat([significant, ias[mask]], ignore_index=True)
    non_significant = pd.concat(
        [non_significant, ias[~mask]], ignore_index=True
    )

    # now reset index
    significant = significant.sort_values("aframe_p_astro", ascending=False)
    non_significant = non_significant.sort_values(
        "aframe_p_astro", ascending=False
    )

    significant.index = range(1, len(significant) + 1)
    non_significant.index = range(1, len(non_significant) + 1)

    def aframe_far_formatter(x):
        lowest = background.min_far
        if x <= background.min_far:
            return rf"$\leq {lowest:.2f}$"
        return "{:.2f}".format(x)

    latex_sig = significant.to_latex(
        columns=columns,
        header=[
            "Event",
            "Catalog",
            r"Aframe FAR $\mathrm{{yr}}^{{-1}}$",
            r"FAR $\mathrm{{yr}}^{{-1}}$",
            r"Aframe $p_\mathrm{{astro}}$",
            r"$p_\mathrm{{astro}}$",
            r"$m_1 (M_{{\odot}})$",
            r"$m_2 (M_{{\odot}})$",
        ],
        formatters={
            "aframe_far": aframe_far_formatter,
            "far": far_formatter,
        },
        float_format="%.2f",
        column_format="l" + len(columns) * "c",
        index=True,
    )

    latex_non_sig = non_significant.to_latex(
        columns=columns,
        header=[
            "Event",
            "Catalog",
            r"Aframe FAR $\mathrm{{yr}}^{{-1}}$",
            r"FAR $\mathrm{{yr}}^{{-1}}$",
            r"Aframe $p_\mathrm{{astro}}$",
            r"$p_\mathrm{{astro}}$",
            r"$m_1 (M_{{\odot}})$",
            r"$m_2 (M_{{\odot}})$",
        ],
        formatters={
            "aframe_far": aframe_far_formatter,
            "far": far_formatter,
        },
        float_format="%.2f",
        column_format="l" + len(columns) * "c",
        index=True,
    )

    sig_caption = r"Candidates that Aframe detects with $p_\mathrm{astro} \geq 0.5$. The catalog each candidate was originally reported in is listed. Events marked with an asterisk occurred during the training period."
    sig_label = "table:significant"
    latex_sig = build_table(latex_sig, sig_caption, sig_label, len(columns))
    non_sig_caption = r"Candidates that Aframe detects with $p_\mathrm{astro} < 0.5$. The catalog each candidate was originally reported in is listed. Events marked with an asterisk occurred during the training period."
    non_sig_label = "table:non_significant"
    latex_non_sig = build_table(
        latex_non_sig, non_sig_caption, non_sig_label, len(columns)
    )

    with open(table_dir / "significant.tex", "w") as f:
        f.write(latex_sig)

    with open(table_dir / "non_significant.tex", "w") as f:
        f.write(latex_non_sig)



if __name__ == "__main__":
    auto_cli(main, as_positional=False) 