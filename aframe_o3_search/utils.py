import gwpy
from datetime import datetime

def build_table(
    latex: str,
    caption: str,
    label: str,
    num_columns: int,
):
    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\rowcolors{2}{white}{gray!15}")

    for line in latex.split("\n"):
        if line.endswith("rule"):
            line += "[1pt]"

        if line.startswith("\\begin{tabular}"):
            width_str = "{\\vrule width 1pt}"
            columns = "l" + "c" * num_columns + "!"
            line = "\\begin{tabular}{!" + width_str + columns + width_str + "}"
        lines.append(line)

    lines.append("\\end{table*}")
    latex = "\n".join(lines)

    return latex

def time_to_gwtc_event(time):
    """
    Convert a gpstime to a GWTC event-like name
    """
    dt = gwpy.time.from_gps(time)
    return dt.strftime("GW%y%m%d_%H%M%S")

def gwtc_event_to_time(event):
    """
    Convert a GWTC event-like name to a gpstime
    """
    event = event.replace("\\", "")
    date = datetime.strptime(event, "GW%y%m%d_%H%M%S")
    gpstime = gwpy.time.to_gps(date)
    return gpstime

def filter_lal_warnings():
    import warnings
    warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")