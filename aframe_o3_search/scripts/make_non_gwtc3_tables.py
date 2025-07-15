"""
Generate tables of non-GWTC-3 events for the AresGW, IAS, OGC, and Aframe catalogs
"""

import gwpy
import re
import numpy as np
import pandas as pd
from pathlib import Path
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
from gwpy.segments import SegmentList
import logging


def add_brackets_to_latex(latex_str):
    latex_str = latex_str.replace("{", "")
    latex_str = latex_str.replace("}", "")

    # Add brackets after _ and + to enclose the numbers
    latex_str = re.sub(r'_(\-?\d+\.?\d*)', r'_{\1}', latex_str)
    latex_str = re.sub(r'\^(\S+)', r'^{\1}', latex_str)
    latex_str = f"${latex_str}$"
    return latex_str

def main(
    aframe_data_dir: Path,
    pipeline_data_dir: Path,
    out_dir: Path,
    recovery_dt: float = 1.0
):
        
    logging.basicConfig()
    logger = logging.getLogger('make_non_gwtc3_table')
    logger.setLevel(logging.INFO)

    # where latex tables are stored
    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True, parents=True)

    # where gpstimes for bilby followup are stored

    segments = SegmentList.read(aframe_data_dir / "train_segments.txt")
    test_segments = SegmentList.read(aframe_data_dir / "test_segments.txt")
    segments.extend(test_segments)
    segments.coalesce()


    # read in background and foreground triggers from aframe
    logger.info("Reading in aframe background, foreground and 0lag data")
    rejected = InjectionParameterSet.read(aframe_data_dir / "rejected-parameters.hdf5")
    background = EventSet.read(aframe_data_dir / "background.hdf5")
    foreground = RecoveredInjectionSet.read(aframe_data_dir / "foreground.hdf5")

    # read in the zero-lag triggers
    zero_lag = EventSet.read(aframe_data_dir / "0lag.hdf5")

    # read in zero-lag triggers with "correct gpstime" estimates
    # there was a minor bug in initial analysis that led to 
    # potential off by one estimates in gpstime. This corresponds 
    # to ~ 0.25 seconds for certain events.
    # Note: only gpstimes from this file should be used and NOT detection stats
    # b/c we didn't reanalyze the background with this bug fix
    zero_lag_gpstimes = EventSet.read(aframe_data_dir / "0lag_correct_gpstime.hdf5")

    # sort the zero-lag triggers by detection tim
    args = zero_lag.detection_time.argsort()
    zero_lag = zero_lag[args]
    logger.info("Fitting or loading p astro")
    p_astro = fit_or_load_pastro(
        background,
        foreground,
        rejected,
        aframe_data_dir
    )


    ##########
    # AresGW #
    ##########

    logger.info("Building AresGW catalog")

    # read in the new candidates reported by AresGW
    ares = Table.read(pipeline_data_dir / "aresgw" / "new.tex", format="latex").to_pandas()

    # sort the new ares candidates table 
    # and aframe events by gpstime
    ares.sort_values("gpstime", inplace=True)


    # filter out aframe events that correspond to the aresgw events
    # and add the aframe p astro to the table
    mask = np.abs(zero_lag.detection_time - ares.gpstime.values[:, None]) < recovery_dt
    mask = mask.any(axis=0)
    ares_gw_candidates = zero_lag[mask]
    
    
    # add aframe p astro and far to the ares gw candidate catalog
    ares = ares.assign(aframe_p_astro=p_astro(ares_gw_candidates.detection_statistic))
    ares = ares.assign(aframe_far=background.far(ares_gw_candidates.detection_statistic))

    # filter for events found by aframe that are above the 2 per day threshold set by GWTC-3
    ares = ares[ares.aframe_far < c.LOW_SIGNIFICANCE_THRESHOLD]

    columns = ["event", "aframe_far", "far", "aframe_p_astro", "p_astro", "m_1", "m_2"]
    ares = ares.sort_values("p_astro", ascending=False)
    latex_ares = ares[columns]
    latex_ares.index = range(1, len(latex_ares) + 1)
    latex_ares = latex_ares.to_latex(
        columns=columns,
        header = ["Event", "Aframe FAR $\mathrm{{yr}}^{{-1}}$", "AresGW FAR $\mathrm{{yr}}^{{-1}}$", "Aframe $p_\mathrm{{astro}}$", "AresGW $p_\mathrm{{astro}}$", "$m_1 (M_{{\odot}})$", "$m_2 (M_{{\odot}})$"],
        float_format="%.2f",
        index=True,
    )

    aresgw_label = "tab:aresgw_candidates"
    aresgw_caption = "Candidate events reported in the AresGW analysis. \\texttt{Aframe} finds none of these events with $p_\mathrm{astro} > 0.5$" 
    latex_aresgw = build_table(latex_ares, aresgw_caption, aresgw_label, len(columns))


    with open(table_dir / "aresgw_candidates.tex", "w") as f:
        f.write(latex_aresgw)

    # write gps times of significant non catalog candidates 
    # to a file for bilby followup
    ares_significant = ares[ares.aframe_p_astro >= 0.5]

    # calculate the gpstime from the closest 
    # aframe candidate event
   
    for event, time in zip(ares_significant.event, ares_significant.gpstime):
        event = event.replace("\\", "")
        bilby_dir = out_dir / "bilby" / event
        bilby_dir.mkdir(exist_ok=True, parents=True)
        with open(bilby_dir / f"trigger_time.txt", "w") as f:
            arg = np.argmin(np.abs(zero_lag_gpstimes.detection_time - time))
            aframe_time = zero_lag_gpstimes.detection_time[arg]
            f.write(f"{aframe_time}\n")
    



    #######
    # IAS #
    #######

    logger.info("Building IAS catalog")

    # read in the new candidates reported by IAS
    # from o3a and b and combine them
    ias_o3a = Table.read(pipeline_data_dir / "ias" / "new_o3a.latex", format="latex").to_pandas()
    ias_o3b = Table.read(pipeline_data_dir / "ias" / "new_o3b.latex", format="latex").to_pandas().drop(columns=["ifar bank"])
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


    ias = ias.assign(aframe_p_astro=p_astro(ias_gw_candidates.detection_statistic))
    ias = ias.assign(aframe_far=background.far(ias_gw_candidates.detection_statistic))
    ias = ias.assign(far=1 / ias.ifar)
    ias = ias[ias.aframe_far < c.LOW_SIGNIFICANCE_THRESHOLD]

    columns = ["event", "aframe_far", "far", "aframe_p_astro", "p_astro", "m_1", "m_2"]
    ias = ias.sort_values("aframe_p_astro", ascending=False)
    latex_ias = ias[columns]
    latex_ias.index = range(1, len(latex_ias) + 1)
    
    latex_ias = latex_ias.to_latex(
        columns=columns,
        header = ["Event", "Aframe FAR $\mathrm{{yr}}^{{-1}}$", "IAS FAR $\mathrm{{yr}}^{{-1}}$", "Aframe $p_\mathrm{{astro}}$", "IAS $p_\mathrm{{astro}}$", "$m_1 (M_{{\odot}})$", "$m_2 (M_{{\odot}})$"],
        float_format="%.2f",
        formatters={"event": lambda x: x.replace("_", "\_"), "m_1": add_brackets_to_latex, "m_2": add_brackets_to_latex, "chi_eff": add_brackets_to_latex},
        index=True,
    ) 

    ias_label = "tab:ias_candidates"
    ias_caption = "Candidate events reported in the IAS analysis. \\texttt{Aframe} finds two events with $p_\mathrm{astro} > 0.5$" 
    latex_ias = build_table(latex_ias, ias_caption, ias_label, len(columns))

    with open(table_dir / "ias_candidates.tex", "w") as f:
        f.write(latex_ias)

    # write gps times of significant non catalog candidates 
    # to a file for bilby followup
    ias_significant = ias[ias.aframe_p_astro >= 0.5]


    for event, time in zip(ias_significant.event, ias_significant.gpstime):
        event = event.replace("\\", "")
        bilby_dir = out_dir / "bilby" / event
        bilby_dir.mkdir(exist_ok=True, parents=True)
        with open(bilby_dir / f"trigger_time.txt", "w") as f:
            arg = np.argmin(np.abs(zero_lag_gpstimes.detection_time - time))
            aframe_time = zero_lag_gpstimes.detection_time[arg]
            f.write(f"{aframe_time}\n")
    

    #######
    # OGC #
    #######

    logger.info("Building OGC catalog")

    # read in the new candidates reported by OGC
    ogc = Table.read(pipeline_data_dir / "ogc" / "new.tex", format="latex").to_pandas()
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
            aframe_ogc_detection_statistic.append(zero_lag.detection_statistic[arg])
        else:
            aframe_ogc_detection_statistic.append(-2)
    aframe_ogc_detection_statistic = np.array(aframe_ogc_detection_statistic)

    ogc = ogc.assign(aframe_p_astro=p_astro(aframe_ogc_detection_statistic))
    ogc = ogc.assign(aframe_far=background.far(aframe_ogc_detection_statistic))
    ogc = ogc.assign(far=1 / ogc.ifar)
    ogc = ogc[ogc.aframe_far < c.LOW_SIGNIFICANCE_THRESHOLD]



    columns = ["event", "aframe_far", "far", "aframe_p_astro", "p_astro", "m_1", "m_2"]
    ogc = ogc.sort_values("aframe_p_astro", ascending=False)
    latex_ogc = ogc[columns]
    latex_ogc.index = range(1, len(latex_ogc) + 1)
    latex_ogc = latex_ogc.to_latex(
        columns=columns,
        header = ["Event", "Aframe FAR $\mathrm{{yr}}^{{-1}}$", "OGC FAR $\mathrm{{yr}}^{{-1}}$", "Aframe $p_\mathrm{{astro}}$", "OGC $p_\mathrm{{astro}}$", "$m_1 (M_{{\odot}})$", "$m_2 (M_{{\odot}})$"],
        float_format="%.2f",
        #formatters={"event": lambda x: x.replace("_", "\_")},
        index=True,
    )   

    ogc_label = "tab:ogc_candidates"
    ogc_caption = "Candidate events reported in the OGC analysis. \\texttt{Aframe} does not find any events with $p_\mathrm{astro} > 0.5$" 
    latex_ogc = build_table(latex_ogc, ogc_caption, ogc_label, len(columns))

    with open(table_dir / "ogc_candidates.tex", "w") as f:
        f.write(latex_ogc)

    # write gps times of significant non catalog candidates 
    # to a file for bilby followup
    ogc_significant = ogc[ogc.aframe_p_astro >= 0.5]
    # calculate the gpstime from the closest 
    # aframe candidate event


    for event, time in zip(ogc_significant.event, ogc_significant.gpstime):
        event = event.replace("\\", "")
        bilby_dir = out_dir / "bilby" / event
        bilby_dir.mkdir(exist_ok=True, parents=True)
        with open(bilby_dir / f"trigger_time.txt", "w") as f:
            arg = np.argmin(np.abs(zero_lag_gpstimes.detection_time - time))
            aframe_time = zero_lag_gpstimes.detection_time[arg]
            f.write(f"{aframe_time}\n")
    
    ###############
    # Aframe Only # 
    ###############

    logger.info("Building Aframe catalog")

    # build catalog
    catalog = build_gwtc3_catalog()
    times = catalog.time.values

    # filter out events that correspond to gwtc-3, ias, ogc, or aresgw events
    mask = (np.abs(zero_lag.detection_time - times[:, None]) < recovery_dt).any(axis=0)
    mask |= (np.abs(zero_lag.detection_time - ares.gpstime.values[:, None]) < recovery_dt).any(axis=0)
    mask |= (np.abs(zero_lag.detection_time - ias.gpstime.values[:, None]) < recovery_dt).any(axis=0)
    mask |= (np.abs(zero_lag.detection_time - ogc.gpstime.values[:, None]) < recovery_dt).any(axis=0)


    catalog_removed = zero_lag[~mask]
    catalog_removed = catalog_removed[catalog_removed.detection_statistic > 6]

    p_astros = p_astro(catalog_removed.detection_statistic)


    mask = p_astros > 0.5
    significant_non_catalog = catalog_removed[mask]
    p_astros = p_astros[mask]

    # build pd.DataFrame of significant events
    aframe = {}
    aframe["gpstime"] = [time for time in significant_non_catalog.detection_time]
    aframe["event"] = [time_to_gwtc_event(time) for time in significant_non_catalog.detection_time]
    aframe["far"] = [background.far(stat) for stat in significant_non_catalog.detection_statistic]
    aframe["p_astro"] = p_astros
    aframe["detection_statistic"] = significant_non_catalog.detection_statistic
    aframe["gpstime"] = significant_non_catalog.detection_time
    aframe = pd.DataFrame(aframe)


    columns = ["event", "far", "p_astro"]
    aframe = aframe.sort_values("detection_statistic", ascending=False)
    aframe.index = range(1, len(aframe) + 1)

    latex_aframe = aframe.to_latex(
        columns=columns,
        index=True, 
        header = ["Event",  "Aframe FAR $\mathrm{{yr}}^{{-1}}$", "Aframe $p_\mathrm{{astro}}$"],
        formatters = {"event": lambda x: x.split("-")[0].replace("_", "\_")},
        float_format="%.2f",
        column_format="l" + len(columns) * "c"
    )

    aframe_label = "tab:aframe_candidates"
    aframe_caption = "Candidates detected by \\texttt{Aframe} with $p_\mathrm{astro} > 0.5$ that were not reported in the GWTC-3, OGC, or IAS catalogs"
    latex_aframe = build_table(latex_aframe, aframe_caption, aframe_label, len(columns))

    with open(table_dir / "aframe_candidates.tex", "w") as f:
        f.write(latex_aframe)
    
    for event, time in zip(aframe.event, aframe.gpstime):
        event = event.replace("\\", "")
        bilby_dir = out_dir / "bilby" / event
        bilby_dir.mkdir(exist_ok=True, parents=True)
        with open(bilby_dir / f"trigger_time.txt", "w") as f:
            arg = np.argmin(np.abs(zero_lag_gpstimes.detection_time - time))
            aframe_time = zero_lag_gpstimes.detection_time[arg]
            f.write(f"{aframe_time}\n")
        

if __name__ == "__main__":
    main(
        aframe_data_dir=Path("/home/ethan.marx/projects/aframe-o3-offline/data/aframe/vetoed"),
        pipeline_data_dir=Path("/home/ethan.marx/projects/aframe-o3-offline/data/"),
        out_dir=Path("/home/ethan.marx/projects/aframe-o3-offline/production_data"),
        recovery_dt = 1.0
    )
       
        

