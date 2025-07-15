"""Apply Vetos to Background, Foreground, and 0lag Aframe Events"""
import logging
import numpy as np
import pandas as pd 
from ledger.events import EventSet, RecoveredInjectionSet
from gwpy.segments import DataQualityDict, SegmentList
from aframe_o3_offline.catalogs import validate_catalog, build_gwtc3_catalog, recover
import aframe_o3_offline.constants as c
from aframe_o3_offline.utils import filter_lal_warnings
from gwosc import datasets
import logging
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
from gwosc import datasets
from gwpy.segments import DataQualityDict

filter_lal_warnings()

VETO_CATEGORIES = ["CAT1", "CAT2", "GATES", "CATALOG", "GSPY"]

def gates_to_veto_segments(path: Path):
    """Naively convert gate files to vetos segments"""
    gates = np.loadtxt(path)
    centers = gates[:, 0]
    windows = gates[:, 1]
    tapers = gates[:, 2] + 0.375

    vetos = np.array(
        [
            [center - window - taper, center + window + taper]
            for center, window, taper in zip(centers, windows, tapers)
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

class VetoParser:
    def __init__(
        self,
        veto_definer_file: Path,
        gate_paths: Dict[str, Path],
        start: float,
        stop: float,
        ifos: List[str],
    ):
        self.logger = logging.getLogger("vizapp")
        self.vetos = DataQualityDict.from_veto_definer_file(veto_definer_file)
        self.logger.info("Populating vetos")
        self.vetos.populate(segments=[[start, stop]], verbose=True)
        self.logger.info("Vetos populated")
        self.gate_paths = gate_paths
        self.ifos = ifos
        self.veto_cache = {}

    def get_vetos(self, category: str):
        vetos = {}

        for ifo in self.ifos:
            if category == "GATES":
                ifo_vetos = gates_to_veto_segments(self.gate_paths[ifo])
            else:
                cat_number = int(category[-1])
                ifo_vetos = DataQualityDict(
                    {
                        k: v
                        for k, v in self.vetos.items()
                        if v.ifo == ifo and v.category == cat_number
                    }
                )
                ifo_vetos = ifo_vetos.union().active

            vetos[ifo] = np.array(ifo_vetos)

        return vetos


def main(
    data_dir: Path,
    outdir: Path,
    veto_definer_file: Path,
    gates: Dict[str, Path],
):
    logging.basicConfig()
    logger = logging.getLogger('apply_vetos')
    logger.setLevel(logging.INFO)

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
    start, stop = background.detection_time.min(), background.detection_time.max()
    veto_parser = VetoParser(veto_definer_file, gates, start, stop, c.IFOS)
    catalog_vetos = get_catalog_vetos(start, stop)

    logger.info(f"{len(background)} background events before vetos")
    logger.info(f"{len(foreground)} foreground events before vetos")
    logger.info(f"{len(zero_lag)} zero-lag events before vetos")
    for cat in VETO_CATEGORIES:

        for i, ifo in enumerate(c.IFOS):

            print(f"Applying vetos for {cat} to {ifo}")
            logging.info(f"Applying vetos for {cat} to {ifo}")
            if cat == "CATALOG":
                vetos = catalog_vetos
            elif cat == "GSPY":
                data = pd.read_csv(f"/home/ethan.marx/projects/aframe-o3-offline/gravityspy/Data/O3_{ifo}.csv")
                data = data[data.label != "Chirp"]
                data = data[data.label != "No_Glitch"]
                data = data[data.confidence > 0.99]
                vetos = np.column_stack([data.GPStime.values - 0.5, data.GPStime.values + 0.5])
            else:
                vetos = veto_parser.get_vetos(cat)[ifo]
            # remove veto times from segments
            # for calculating livetime
            segments -= SegmentList(vetos)

            foreground_count = len(foreground)
            background_count = len(background)
            zero_lag_count = len(zero_lag)


            if len(vetos) > 0:
                #background = background.apply_vetos(vetos, i, chunk_size=1000)
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
    np.savetxt(outdir / "livetime_postveto.txt", np.array([livetime]))
    logger.info(f"Total livetime after removing vetod times: {livetime}")

    logger.info(f"{len(background)} background events after vetos")
    logger.info(f"{len(foreground)} foreground events after vetos")
    logger.info(f"{len(zero_lag)} zero-lag events after vetos")

    logger.info("Writing trigger files")
    #background = background.sort_by("detection_statistic")
    foreground.write(outdir /"foreground.hdf5")
    #background.write(outdir / "background.hdf5")
    zero_lag.write(outdir /  "0lag.hdf5")

if __name__ == "__main__":
    veto_definer_file = "/home/ethan.marx/projects/aframe-o3-offline//aframev2/projects/plots/plots/vetos/H1L1-HOFT_C01_O3_CBC.xml"
    gates = {ifo: f"/home/ethan.marx/projects/aframe-o3-offline//aframev2/projects/plots/plots/vetos/{ifo}-O3_GATES_1238166018-31197600.txt" for ifo in c.IFOS}
    data_dir = Path("/home/ethan.marx/projects/aframe-o3-offline/data/aframe")
    outdir = data_dir / "vetoed-gspy"
    outdir.mkdir(exist_ok=True, parents=True)
    main(data_dir, outdir, veto_definer_file, gates)