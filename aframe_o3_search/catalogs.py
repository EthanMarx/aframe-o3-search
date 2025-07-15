from collections import defaultdict
from gwosc import datasets
from gwosc.api import fetch_event_json
import numpy as np
import pandas as pd
from ledger.events import EventSet


def build_gwtc3_catalog():
    """
    Query catalog for all events that were detected by a superset of
    the given interferometers; Add in reported FARs from MF pipelines
    """

    # query catalog events over O3
    start = min(datasets.run_segment("O3a_4KHZ_R1"))
    stop = max(datasets.run_segment("O3b_4KHZ_R1"))
    events = datasets.query_events(
        select=[f"gps-time >= {start}", f"gps-time <= {stop}"]
    )

    pipelines = ["gstlal", "pycbc_bbh", "mbta", "cwb", "pycbc_broad"]

    # build the catalog of interest
    catalog = defaultdict(list)
    parameters = [
        "mass_1_source",
        "mass_1_source_upper",
        "mass_1_source_lower",
        "mass_2_source",
        "mass_2_source_upper",
        "mass_2_source_lower",
        "chirp_mass_source",
        "luminosity_distance",
        "redshift",
        "network_matched_filter_snr",
        "far",
        "p_astro",
    ]

    # loop over events and fetch the parameters of interest;
    for event in events:
        params = fetch_event_json(event)["events"][event]
        # only consider confident events
        catalog_shortname = params["catalog.shortName"]
        if catalog_shortname not in ["GWTC-2.1-confident", "GWTC-3-confident"]:
            continue

        # find the ifos that detected the event
        event_ifos = np.unique([s["detector"] for s in params["strain"]])
        catalog["ifos"].append(event_ifos)
        catalog["event"].append(event)

        # append the parameters of interest
        # for the event to the master dictonary
        for p in parameters:
            catalog[p].append(params[p])

        catalog["time"].append(datasets.event_gps(event))

        # append pipeline specific information like p astro
        for pipeline in pipelines:
            event = event.split("-")[0]
            try:
                p_astro = params["parameters"][
                    f"{catalog_shortname}_{pipeline}_{event}"
                ]["p_astro"]
            except KeyError:
                p_astro = 0.0
            catalog[f"{pipeline}_p_astro"].append(p_astro)

    # create a pandas dataframe from the catalog
    catalog = pd.DataFrame(catalog)

    return catalog


def append_cnn_catalog(catalog: pd.DataFrame):
    # read in fars from MF pipelines and merge
    pipeline_catalog = pd.read_csv("./data/mf/o3-catalog-clean.csv")
    pipeline_catalog.replace(np.nan, 1e10, inplace=True)
    catalog = catalog.merge(pipeline_catalog, on=["event"], how="inner")
    catalog.event = [event.split("-")[0] for event in catalog.event]

    # read in fars from ccn search and merge
    cnn_catalog = pd.read_csv("./data/cnn-2d-search/events-clean.csv")
    events = cnn_catalog.event.values
    cnn_analyzed = np.array(
        [event in events for event in catalog.event.values]
    )

    catalog = catalog.merge(cnn_catalog, on=["event"], how="left")
    catalog["cnn_analyzed"] = cnn_analyzed.astype(bool)

    return catalog


def validate_catalog(
    catalog: pd.DataFrame,
    segments: np.ndarray,
    psd_length: float,
    buffer: float,
):
    """
    Given a catalog of detections and array of segments
    analyzed by aframe, determine which events aframe analyzed;

    Args:
        catalog: pd.DataFrame
            catalog of events to be validated
        segments: np.ndarray
            array of segments analyzed by aframe
        psd_length: float
            length of the psd burn in
        buffer: float
            length of the end buffer
    """

    # get the times of the events
    times = catalog["time"].values[:, None]

    # get the start and end times of the segments
    starts, ends = segments.T

    # first check if the event was in the segments at all
    mask = (times >= starts) & (times <= ends)
    not_in_segment = ~mask.any(axis=1)

    # then check if the event was in psd burn in
    in_burn_in = ((times - starts < psd_length) & (times - starts > 0)).any(
        axis=1
    )

    # then check if the event was in the end buffer
    in_last_batch = (
        ((ends - times < buffer) & (ends - times >= 0)).any(axis=1).sum()
    )

    # determine of the event was analyzed by aframe
    analyzed = ~not_in_segment & ~in_burn_in & ~in_last_batch

    catalog["aframe_analyzed"] = analyzed.astype(bool)
    catalog["not_in_segment"] = not_in_segment.astype(bool)
    catalog["in_burn_in"] = in_burn_in.astype(bool)
    catalog["in_last_batch"] = in_last_batch.astype(bool)
    return catalog


def append_ares_gw_catalog(catalog: pd.DataFrame):
    pass


def recover(catalog: pd.DataFrame, foreground: EventSet, background: EventSet):
    """
    Append Aframe FAR and time difference to the catalog events
    """
    times = catalog.time.values
    diffs = np.abs(foreground.detection_time - times[:, None])
    idx = np.argmin(diffs, axis=1)
    diffs = np.min(diffs, axis=1)
    masked = foreground[idx]
    catalog["aframe_detection_statistic"] = masked.detection_statistic
    fars = np.zeros(len(masked))
    for i in range(len(masked)):
        far = background.far(masked.detection_statistic[i])
        fars[i] = far
    catalog["aframe_far"] = fars
    catalog["aframe_time_diff"] = diffs

    return catalog
