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
        "mu_0_nieuw": np.log(1.0168),  # was 12.0d
        "sigma_nieuw": 0.1455,  # was 0.35
        "factoren": {
            "GFA": {"mean": 0.03803, "sd": 0.10320},
            "verdiepingen": {"mean": -0.11280, "sd": 0.60394},
            "hoogte": {"mean": -0.05353, "sd": 1.30524},
            "schadegraad": {"mean": -0.07906, "sd": 0.61171},
            "asbest": {"mean": -0.02384, "sd": 1.12200},
            "bereikbaarheid": {"mean": 0.06480, "sd": 0.79447},
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
        "mu_0_nieuw": np.log(1.1150),  # was 22.0d
        "sigma_nieuw": 0.2104,  # was 0.35
        "factoren": {
            "GFA": {"mean": -0.01002, "sd": 0.04840},
            "verdiepingen": {"mean": 0.10884, "sd": 2.07896},
            "hoogte": {"mean": 0.65302, "sd": 0.85085},
            "geveltype": {"mean": 0.00000, "sd": 2.10354},
            "daktype": {"mean": 0.00000, "sd": 2.10354},
            "schade_m2": {"mean": 0.00000, "sd": 2.10354},
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
