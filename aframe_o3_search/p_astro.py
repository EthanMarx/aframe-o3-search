from p_astro.background import KdeAndPolynomialBackground
from p_astro.foreground import KdeForeground
from p_astro.p_astro import Pastro
import aframe_o3_search.constants as c
import pickle
import logging


def fit_or_load_pastro(
    background,
    foreground,
    rejected,
    cache_dir,
):
    p_astro_cache = cache_dir / "pastro.pkl"
    if p_astro_cache.exists():
        logging.info("loading cached pastro model")
        with open(p_astro_cache, "rb") as f:
            p_astro = pickle.load(f)
    else:
        logging.info("fitting pastro model")
        background_model = KdeAndPolynomialBackground(background)
        foreground_model = KdeForeground(foreground, rejected, c.ASTRO_RATE)
        p_astro = Pastro(foreground_model, background_model)
        with open(p_astro_cache, "wb") as f:
            pickle.dump(p_astro, f)
    return p_astro
