"""
Automatisch gegenereerd door elicitatie_verwerker.py.
Bevat de coëfficiënten NA verwerking van de expert-scenario's uit de
elicitatie-CSV. Kopieer de gewenste waarden over naar FASE_INSTELLINGEN
in easicon_model_v2.py (mean/sd per factor, mu_0 per fase).
"""

import numpy as np

GEUPDATETE_FASE_WAARDEN = {
    "A": {
        "n_scenarios": 5,
        "mu_0_nieuw": np.log(1.0914),  # was 12.0d
        "sigma_nieuw": 0.1472,  # was 0.35
        "factoren": {
            "GFA": {"mean": -0.06057, "sd": 0.08040},
            "verdiepingen": {"mean": 0.21685, "sd": 0.90341},
            "hoogte": {"mean": 0.13318, "sd": 1.24734},
            "schadegraad": {"mean": 0.38313, "sd": 0.49391},
            "asbest": {"mean": -0.08368, "sd": 1.09738},
            "bereikbaarheid": {"mean": 0.07130, "sd": 0.80577},
        },
    },
    "B1": {
        "n_scenarios": 0,
        "mu_0_nieuw": np.log(1.0000),  # was 20.0d
        "sigma_nieuw": 0.3500,  # was 0.35
        "factoren": {
            "GFA": {"mean": 0.00000, "sd": 3.50000},
            "verdiepingen": {"mean": 0.00000, "sd": 3.50000},
            "hoogte": {"mean": 0.00000, "sd": 3.50000},
            "funderingsschade": {"mean": 0.00000, "sd": 3.50000},
            "bodemtype": {"mean": 0.00000, "sd": 3.50000},
            "onderbemaling": {"mean": 0.00000, "sd": 3.50000},
        },
    },
    "B2": {
        "n_scenarios": 0,
        "mu_0_nieuw": np.log(1.0000),  # was 28.0d
        "sigma_nieuw": 0.3500,  # was 0.35
        "factoren": {
            "GFA": {"mean": 0.00000, "sd": 3.50000},
            "verdiepingen": {"mean": 0.00000, "sd": 3.50000},
            "hoogte": {"mean": 0.00000, "sd": 3.50000},
            "casoschade": {"mean": 0.00000, "sd": 3.50000},
            "bouwmethode": {"mean": 0.00000, "sd": 3.50000},
            "structuuringrepen": {"mean": 0.00000, "sd": 3.50000},
        },
    },
    "C1": {
        "n_scenarios": 3,
        "mu_0_nieuw": np.log(19.6132),  # was 22.0d
        "sigma_nieuw": 0.2565,  # was 0.35
        "factoren": {
            "GFA": {"mean": -0.02848, "sd": 0.06155},
            "verdiepingen": {"mean": 0.00000, "sd": 2.56496},
            "hoogte": {"mean": 0.00000, "sd": 2.56496},
            "geveltype": {"mean": 0.00000, "sd": 2.56496},
            "daktype": {"mean": 0.00000, "sd": 2.56496},
            "schade_m2": {"mean": 0.00000, "sd": 2.56496},
        },
    },
    "C2": {
        "n_scenarios": 0,
        "mu_0_nieuw": np.log(1.0000),  # was 20.0d
        "sigma_nieuw": 0.3500,  # was 0.35
        "factoren": {
            "GFA": {"mean": 0.00000, "sd": 3.50000},
            "verdiepingen": {"mean": 0.00000, "sd": 3.50000},
            "hoogte": {"mean": 0.00000, "sd": 3.50000},
            "elektra_pct": {"mean": 0.00000, "sd": 3.50000},
            "verwarmingssysteem": {"mean": 0.00000, "sd": 3.50000},
            "n_onderaannemers": {"mean": 0.00000, "sd": 3.50000},
        },
    },
    "C3": {
        "n_scenarios": 0,
        "mu_0_nieuw": np.log(1.0000),  # was 15.0d
        "sigma_nieuw": 0.3500,  # was 0.35
        "factoren": {
            "GFA": {"mean": 0.00000, "sd": 3.50000},
            "verdiepingen": {"mean": 0.00000, "sd": 3.50000},
            "hoogte": {"mean": 0.00000, "sd": 3.50000},
            "afwerkingsniveau": {"mean": 0.00000, "sd": 3.50000},
            "schade_pct": {"mean": 0.00000, "sd": 3.50000},
            "n_wijzigingen": {"mean": 0.00000, "sd": 3.50000},
        },
    },
}
