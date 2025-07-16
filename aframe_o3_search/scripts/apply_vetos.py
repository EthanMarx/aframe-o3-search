"""Apply Vetos to Background, Foreground, and 0lag Aframe Events"""

import logging
import numpy as np
import pandas as pd
from ledger.events import EventSet, RecoveredInjectionSet
from gwpy.segments import DataQualityDict, DataQualityFlag, SegmentList, Segment
import aframe_o3_search.constants as c
from aframe_o3_search.utils import filter_lal_warnings
from gwosc import datasets
from pathlib import Path
from typing import Dict, List
from jsonargparse import auto_cli

filter_lal_warnings()

VETO_CATEGORIES = ["CAT1", "CAT2", "GATES", "CATALOG"] 
O3A_END = 1253977218
O3B_START = 1256655618

def gates_to_veto_segments(path: Path):
    """Naively convert gate files to vetos segments"""
    gates = np.loadtxt(path)
    centers = gates[:, 0]
    windows = gates[:, 1]
    tapers = gates[:, 2] + 0.375

    vetos = np.array(
        [
            [center - window - taper, center + window + taper]
            for center, window, taper in zip(
                centers, windows, tapers, strict=False
            )
        ]
    )

    return vetos


def get_catalog_vetos(start: float, stop: float, delta: float = 1.0):
    events = datasets.query_events(
        select=[f"gps-time >= {start}", f"gps-time <= {stop}"]
    )
    times = np.array([datasets.event_gps(event) for event in events])
    vetos = np.column_stack([times - delta, times + delta])
    return vetos

def get_open_vetos(category: str, ifo: str, start: float, stop: float):
    analyzed = SegmentList([Segment(start, stop)])
    passed = SegmentList()
    passed.extend(DataQualityFlag.fetch_open_data(f"{ifo}_CBC_{category}", start, O3A_END).active)
    passed.extend(DataQualityFlag.fetch_open_data(f"{ifo}_CBC_{category}", O3B_START, stop).active)

    vetos = analyzed - passed
    vetos = SegmentList([Segment(*x) for x in vetos])

    return np.array(vetos) 

def get_vetos(
    category: str, 
    ifo: str,
    start: float,
    stop: float,
    gates: dict[str, Path]
):
    if category == "CAT":
        vetos = get_catalog_vetos(start, stop)
    elif category == "GATES":
        vetos = gates_to_veto_segments(gates[ifo])
    else:
        vetos = get_open_vetos(category, ifo, start, stop)
    return vetos

def main(
    data_dir: Path,
    out_dir: Path,
    gates: Dict[str, Path],
):
    log_format = "%(levelname)s - %(message)s"
    logging.basicConfig(format=log_format)
    logger = logging.getLogger("apply_vetos")
    logger.setLevel(logging.INFO)

    out_dir.mkdir(exist_ok=True, parents=True)
    
    # read in background, foreground, rejected parameters
    # and 0lag triggers from aframe
    logger.info("Reading in background foreground and 0lag triggers")
    background = EventSet.read(data_dir / "background.hdf5")
    zero_lag = EventSet.read(data_dir / "0lag.hdf5")
    zero_lag_train = EventSet.read(data_dir / "0lag_train.hdf5")
    foreground = RecoveredInjectionSet.read(data_dir / "foreground.hdf5")

    # read in segments for livetime calculation
    logger.info("Reading in segments for livetime calculation")
    segments = SegmentList.read(data_dir / "train_segments.txt")
    test_segments = SegmentList.read(data_dir / "test_segments.txt")

    # append test segments to training segments
    segments.extend(test_segments)
    # evaluate livetime before and after coalescing segments;
    # the two livetimes should be equal
    livetime = np.sum([seg.end - seg.start for seg in segments])
    logger.info(f"Total livetime before coalescing segments: {livetime}")
    segments.coalesce()
    livetime = np.sum([seg.end - seg.start for seg in segments])
    logger.info(f"Total livetime after coalescing segments: {livetime}")

    logger.info("Combining 0lag triggers from training and testing")
    zero_lag.append(zero_lag_train)

    logger.info("Parsing vetos")
    start, stop = (
        background.detection_time.min(),
        background.detection_time.max(),
    )

    logger.info(f"{len(background)} background events before vetos")
    logger.info(f"{len(foreground)} foreground events before vetos")
    logger.info(f"{len(zero_lag)} zero-lag events before vetos")
    for cat in VETO_CATEGORIES:
        for i, ifo in enumerate(c.IFOS):
            logger.info(f"Applying vetos for {cat} to {ifo}")
            vetos = get_vetos(cat, ifo, start, stop, gates) 
            
            # remove veto times from segments
            # for calculating livetime
            segments -= SegmentList(vetos)

            foreground_count = len(foreground)
            background_count = len(background)
            zero_lag_count = len(zero_lag)

            if len(vetos) > 0:
                background = background.apply_vetos(vetos, i, chunk_size=1000)
                foreground = foreground.apply_vetos(vetos, i, chunk_size=1000)
                if cat != "CATALOG":
                    zero_lag = zero_lag.apply_vetos(vetos, i, chunk_size=1000)
            logger.info(
                f"\t{background_count - len(background)} {cat} "
                f"background events removed for ifo {ifo}"
            )
            logger.info(
                f"\t{foreground_count - len(foreground)} {cat} "
                f"foreground events removed for ifo {ifo}"
            )
            logger.info(
                f"\t{zero_lag_count - len(zero_lag)} {cat} "
                f"zero-lag events removed for ifo {ifo}"
            )

    # calculate livetime after removing vetos
    segments.coalesce()
    livetime = np.sum([seg.end - seg.start for seg in segments])
    np.savetxt(out_dir / "livetime_postveto.txt", np.array([livetime]))
    logger.info(f"Total livetime after removing vetod times: {livetime}")

    logger.info(f"{len(background)} background events after vetos")
    logger.info(f"{len(foreground)} foreground events after vetos")
    logger.info(f"{len(zero_lag)} zero-lag events after vetos")

    logger.info("Writing trigger files")
    background = background.sort_by("detection_statistic")
    foreground.write(out_dir / "foreground.hdf5")
    background.write(out_dir / "background.hdf5")
    zero_lag.write(out_dir / "0lag.hdf5")


if __name__ == "__main__":
    auto_cli(main, as_positional=False) 
