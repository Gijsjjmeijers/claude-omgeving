"""
EaSiCon Bayesiaans Duratie-Schattingsmodel — VERSIE 2
======================================================
Auteur  : Gijs Meijers (4957822) | TU Delft x Haskoning | 2026
Thesis  : Self-learning Construction Phase Estimation for a Digital Twin

Dit script implementeert een Bayesiaans lognormaal REGRESSIEMODEL voor het
schatten van bouwfase-durations in het EaSiCon-project (aardbeving-renovaties
in Eemsdelta, uitgevoerd door VolkerWessels).

WAT IS ER ANDERS DAN IN `easicon_model.py` (lees dit eerst)
-------------------------------------------------------------------------
`easicon_model.py` (versie 1) gebruikt een "Normal-Normal" model: alleen de
baseline-duur (mu_n) per fase leert mee met nieuwe projecten. De covariaten
(schadegraad, asbest, GFA, ...) hebben daar VASTE coëfficiënten die nooit
veranderen, ongeacht hoeveel projecten er binnenkomen. Zie de code daar:
`update()` trekt het covariate-effect er eerst vanaf en leert daarna alleen
over de rest (de intercept).

Dit bestand, versie 2, gebruikt in plaats daarvan het volledige Bayesiaanse
regressiemodel uit `estimation_module.py` (klasse `PhaseDurationModel`), dat
NIET alleen de baseline maar OOK elke covariaat-coëfficiënt EN de
ruis-parameter sigma laat meeleren met elk voltooid project. Dat is een
Normal-Inverse-Gamma (NIG) conjugate update in plaats van een simpele
Normal-Normal update op alleen het intercept.

Dit bestand is dus GEEN los model meer. Het is een Nederlandstalige "schil"
(config, output, grafieken, EaSiCon-fasenamen) om het generieke model in
`estimation_module.py` heen. Alle daadwerkelijke Bayesiaanse wiskunde staat
in `estimation_module.py` — lees dat bestand erbij als je de update-formules
zelf wilt narekenen.

Concreet, per fase, wordt nu geleerd (in plaats van alleen mu_n in v1):
    beta_0f  (baseline/intercept)                -- leerde al mee in v1
    beta_jf  (algemene covariaten: GFA,           -- NIEUW t.o.v. v1
              verdiepingen, hoogte)
    gamma_kf (fase-specifieke covariaten,         -- NIEUW t.o.v. v1
              bv. schadegraad, asbest)
    sigma_f  (project-tot-project ruis)           -- NIEUW t.o.v. v1

Theorie (zie thesis H5/H6, Kim & Reinschmidt 2009, Gelman et al. 2013):
  - log(duur) ~ Normaal(mu, sigma²)             [lognormaal voor positieve durations]
  - mu = beta_0f + som_j(beta_jf * x_j) + som_k(gamma_kf * z_kf)   (lineaire predictor)
  - Prior op (beta, sigma²): Normal-Inverse-Gamma (conjugate met bovenstaande)
  - Update: exact, gesloten vorm na elk project (geen MCMC nodig)

Gebruik : python easicon_model_v2.py
Output  :
  1. Tabel — huidige schatting per fase in dagen, incl. covariaten + hun onzekerheid
  2. TODO-lijst — openstaande punten na interviews
  3. easicon_v2_resultaten.png     — convergentie van de verwachte duur per fase (100 projecten)
  4. easicon_v2_zekerheid.png     — zekerheidstoename per fase (intervalsbreedte + sigma-schatting)
  5. easicon_v2_coefficienten.png — NIEUW: convergentie van de covariaat-coëfficiënten zelf

Fases (NEN 2574 + Flapper 2005):
  A   Voorbereiding
  B1  Constructief: Fundering & Onderbouw
  B2  Constructief: Skelet & Casco
  C1  Afbouw: Gevel & Dak
  C2  Afbouw: Technische Installaties
  C3  Afbouw: Functionele Afwerking
"""

import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from estimation_module import PhaseDurationModel

# Zorg dat de Windows-console UTF-8 uitvoert (box-tekens, β, enz.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# SECTIE 1: INSTELLINGEN
# Alle prior-waarden en coëfficiënten staan hier centraal.
# Na expert-interviews vervang je ALLEEN de waarden in dit blok.
# ==============================================================================

# --- Observatieruis (nu een PRIOR, geen vaste waarde zoals in v1) ---
# SIGMA_LOG_PRIOR = startpunt voor de schatting van sigma (std van log(duur)
# tussen projecten van hetzelfde type). Elk model leert zijn eigen sigma bij
# vanuit deze startwaarde — in v1 was sigma een vaste constante die nooit
# veranderde.
# 0.35 betekent dat ~68% van de durations binnen een factor exp(0.35) ≈ 1.42
# van het gemiddelde valt, VOORDAT er projectdata is binnengekomen.
# LITERATUURSCHATTING (Lind 2005, Kim & Reinschmidt 2009): 0.30–0.45 voor
# gestandaardiseerde renovatieprojecten. Eemsdelta-woningen zijn relatief uniform
# → ondergrens van de range.
# !! VALIDEREN: bespreek met Ragnar Klabbers / VolkerWessels-planners.
SIGMA_LOG_PRIOR = 0.35

# SIGMA_CONFIDENCE = hoeveel "pseudo-projecten" de sigma-prior weegt.
# Hoger = meer vertrouwen in SIGMA_LOG_PRIOR, dus trager bijgesteld door data.
# Lager = de eerste paar echte projecten buigen de sigma-schatting snel om.
SIGMA_CONFIDENCE = 5.0

# --- Algemene covariate-coëfficiënten (alle fases) ---
# Dit zijn nu PRIOR-GEMIDDELDEN, geen vaste effecten zoals in v1: het model
# leert na elk project een bijgestelde waarde. De onzekerheid rond elke prior
# wordt apart ingesteld via *_SD hieronder (kleiner = stelliger prior, buigt
# trager om bij nieuwe data).
# LITERATUURSCHATTINGEN — bronnen: Lawal et al. 2023, Kim et al. 2019,
#   Chan & Kumaraswamy 1999, Odeh & Battaineh 2002.
# Interpretatie: effect = exp(β × waarde) als factor op de duur.
# !! VALIDEREN met expert-interviews voor Eemsdelta-context.
#
#   BETA_GFA = 0.003/m²:
#     100m² vs 80m² woning → exp(0.003×20) = 1.06 → 6% langer   [valideer]
BETA_GFA             = 0.003
BETA_GFA_SD          = 0.0015   # onzekerheid op BETA_GFA (prior std)
#   BETA_VERDIEPINGEN = 0.10:
#     extra verdieping → exp(0.10) = 1.11 → 11% langer            [valideer]
BETA_VERDIEPINGEN    = 0.10
BETA_VERDIEPINGEN_SD = 0.05
#   BETA_HOOGTE = 0.020/m:
#     extra meter hoogte → exp(0.020) = 1.02 → 2% langer          [valideer]
BETA_HOOGTE          = 0.020
BETA_HOOGTE_SD       = 0.010

# Covariaten die in elke fase meedoen, met hun prior en onzekerheid.
# Volgorde ligt vast: dit is ook de kolomvolgorde in de design-matrix X die
# naar PhaseDurationModel gaat.
ALGEMENE_COVARIATEN = ["GFA", "verdiepingen", "hoogte"]
ALGEMENE_PRIOR_MEAN = {"GFA": BETA_GFA, "verdiepingen": BETA_VERDIEPINGEN,
                       "hoogte": BETA_HOOGTE}
ALGEMENE_PRIOR_SD   = {"GFA": BETA_GFA_SD, "verdiepingen": BETA_VERDIEPINGEN_SD,
                       "hoogte": BETA_HOOGTE_SD}
# "Ware" waarden voor de simulatie (SQ5-verificatie); onbekend in de praktijk.
ALGEMENE_WARE_WAARDE = {"GFA": 0.004, "verdiepingen": 0.12, "hoogte": 0.018}

# --- Fase-instellingen ---
# Per fase:
#   mu_0       : startschatting op log-schaal = log(verwachte dagen), prior-
#                gemiddelde van de baseline (intercept), voor een project
#                met alle covariaten = 0
#   breedte_0  : (optimistisch, pessimistisch) drie-punts-schatting in dagen,
#                gebruikt om de startonzekerheid tau_0 wiskundig af te leiden
#                (zie threepoint_to_lognormal in estimation_module.py)
#   ware_mu, fase_specifieke_factoren[...]["ware_waarde"] :
#                ALLEEN voor de simulatie/verificatiegrafieken (SQ5) — de
#                "ware" waarden die het model, uitgaand van een (mogelijk
#                bewust foute) prior, terug moet vinden uit gesimuleerde
#                projectdata. Onbekend in de praktijk.
#   kleur      : voor de grafiek
#   fase_specifieke_factoren : per factor een dict met:
#                "mean"        prior-gemiddelde van het effect (literatuur)
#                "sd"          onzekerheid op die prior
#                "ware_waarde" simulatiedoel (alleen voor verificatie)

# Toelichting op de LITERATUURSCHATTINGEN hieronder
# ─────────────────────────────────────────────────
# mu_0  : log van de verwachte duur in werkdagen (literatuur + expert-kennis TU Delft).
#          !! VALIDEREN: vervang met drie-punts-schatting uit expert-interview.
# breedte_0 : (optimistisch, pessimistisch) in dagen, gebruikt om tau_0 af te leiden.
#          !! VALIDEREN: uit expert-interview.
# beta / gamma "mean"/"sd" : effect op log(duur) per eenheid covariate, MET
#          bijbehorende prior-sd (hoe zeker je ervan bent). Praktisch:
#          exp(β × waarde) = factor op de duur.
#          !! VALIDEREN: alle waarden zijn literatuurschattingen — check met experts.

FASE_INSTELLINGEN = {
    "A": {
        "naam":      "Voorbereiding",
        # Literatuur: voorbereiding aardbevingrenovatie ~10-16 werkdagen
        # (Bron: Flapper 2005; eigen schatting op basis van EaSiCon-context)
        "mu_0":      np.log(12),   # !! VALIDEREN: expert-interview (nu: 12 werkdagen)
        "breedte_0": (8, 20),      # !! VALIDEREN: (optimistisch, pessimistisch) dagen
        "ware_mu":   np.log(18),   # simulatiedoel (onbekend in praktijk)
        "kleur":     "#2196F3",
        "fase_specifieke_factoren": {
            # schadegraad (0-10 schaal, 10 = zwaar beschadigd):
            #   β=0.04 → score 5: exp(5×0.04)=1.22 → 22% langer
            #            score 10: exp(10×0.04)=1.49 → 49% langer
            "schadegraad":    {"mean": 0.04, "sd": 0.02,  "ware_waarde": 0.05},  # !! VALIDEREN
            # asbest aanwezig (0=nee, 1=ja):
            #   β=0.30 → aanwezig: exp(0.30)=1.35 → 35% langer (sanering + herplanning)
            "asbest":         {"mean": 0.30, "sd": 0.15,  "ware_waarde": 0.40},  # !! VALIDEREN
            # bereikbaarheid (1=goed, 5=slecht):
            #   β=0.07 → slechtste score (5 vs 1 = +4): exp(4×0.07)=1.32 → 32% langer
            "bereikbaarheid": {"mean": 0.07, "sd": 0.035, "ware_waarde": 0.06},  # !! VALIDEREN
        },
    },
    "B1": {
        "naam":      "Constructief: Fundering & Onderbouw",
        # Literatuur: fundering aardbevingrenovatie ~18-28 werkdagen
        # (Bron: Kim & Reinschmidt 2009; Odeh & Battaineh 2002)
        "mu_0":      np.log(20),   # !! VALIDEREN: expert-interview (nu: 20 werkdagen)
        "breedte_0": (13, 32),     # iets breder: funderingswerk is onzeker
        "ware_mu":   np.log(30),   # simulatiedoel
        "kleur":     "#F44336",
        "fase_specifieke_factoren": {
            # funderingsschade (0-10):
            #   β=0.06 → score 10: exp(10×0.06)=1.82 → 82% langer
            "funderingsschade": {"mean": 0.06, "sd": 0.03,  "ware_waarde": 0.08},  # !! VALIDEREN
            # bodemtype (0=zand, 1=klei, 2=veen):
            #   β=0.18 → veen vs zand: exp(2×0.18)=1.43 → 43% langer
            "bodemtype":        {"mean": 0.18, "sd": 0.09,  "ware_waarde": 0.15},  # !! VALIDEREN
            # onderbemaling noodzakelijk (0=nee, 1=ja):
            #   β=0.25 → noodzakelijk: exp(0.25)=1.28 → 28% langer
            "onderbemaling":    {"mean": 0.25, "sd": 0.125, "ware_waarde": 0.30},  # !! VALIDEREN
        },
    },
    "B2": {
        "naam":      "Constructief: Skelet & Casco",
        # Literatuur: skelet/casco aardbevingrenovatie ~24-35 werkdagen
        # (Bron: Lawal et al. 2023; eigen schatting)
        "mu_0":      np.log(28),   # !! VALIDEREN: expert-interview (nu: 28 werkdagen)
        "breedte_0": (19, 40),
        "ware_mu":   np.log(32),   # simulatiedoel
        "kleur":     "#FF9800",
        "fase_specifieke_factoren": {
            # casoschade (0-10):
            #   β=0.05 → score 10: exp(10×0.05)=1.65 → 65% langer
            "casoschade":        {"mean": 0.05, "sd": 0.025, "ware_waarde": 0.06},  # !! VALIDEREN
            # bouwmethode (0=prefab, 1=traditioneel, 2=complex maatwerk):
            #   β=0.20 → complex vs prefab: exp(2×0.20)=1.49 → 49% langer
            "bouwmethode":       {"mean": 0.20, "sd": 0.10,  "ware_waarde": 0.18},  # !! VALIDEREN
            # structuuringrepen (0-5 ingrijpende aanpassingen):
            #   β=0.10 → 5 ingrepen: exp(5×0.10)=1.65 → 65% langer
            "structuuringrepen": {"mean": 0.10, "sd": 0.05,  "ware_waarde": 0.12},  # !! VALIDEREN
        },
    },
    "C1": {
        "naam":      "Afbouw: Gevel & Dak",
        # Literatuur: gevel & dak aardbevingrenovatie ~18-28 werkdagen
        # (Bron: Chan & Kumaraswamy 1999; eigen schatting Eemsdelta-context)
        "mu_0":      np.log(22),   # !! VALIDEREN: expert-interview (nu: 22 werkdagen)
        "breedte_0": (15, 32),
        "ware_mu":   np.log(28),   # simulatiedoel
        "kleur":     "#4CAF50",
        "fase_specifieke_factoren": {
            # geveltype (0=enkelvoudig baksteen, 1=gemengd, 2=samengesteld):
            #   β=0.18 → samengesteld vs enkelvoudig: exp(2×0.18)=1.43 → 43% langer
            "geveltype": {"mean": 0.18,  "sd": 0.09,  "ware_waarde": 0.20},   # !! VALIDEREN
            # daktype (0=plat dak, 1=hellend, 2=mansarde/complex):
            #   β=0.15 → mansarde vs plat: exp(2×0.15)=1.35 → 35% langer
            "daktype":   {"mean": 0.15,  "sd": 0.075, "ware_waarde": 0.13},   # !! VALIDEREN
            # schade_m2 (m² beschadigde gevelbekleding, typisch 0-60 m²):
            #   β=0.008 → 30 m² schade: exp(30×0.008)=1.27 → 27% langer
            "schade_m2": {"mean": 0.008, "sd": 0.004, "ware_waarde": 0.010},  # !! VALIDEREN
        },
    },
    "C2": {
        "naam":      "Afbouw: Technische Installaties",
        # Literatuur: installaties aardbevingrenovatie ~16-25 werkdagen
        # (Bron: Odeh & Battaineh 2002; eigen schatting)
        "mu_0":      np.log(20),   # !! VALIDEREN: expert-interview (nu: 20 werkdagen)
        "breedte_0": (14, 29),
        "ware_mu":   np.log(27),   # simulatiedoel
        "kleur":     "#9C27B0",
        "fase_specifieke_factoren": {
            # elektra_pct (% van elektra dat nieuw wordt, 0-100):
            #   β=0.004 → 100% nieuw: exp(100×0.004)=1.49 → 49% langer
            "elektra_pct":        {"mean": 0.004, "sd": 0.002, "ware_waarde": 0.005},  # !! VALIDEREN
            # verwarmingssysteem (0=handhaven, 1=upgraden, 2=volledig vervangen):
            #   β=0.22 → volledig vervangen vs handhaven: exp(2×0.22)=1.55 → 55% langer
            "verwarmingssysteem": {"mean": 0.22,  "sd": 0.11,  "ware_waarde": 0.25},   # !! VALIDEREN
            # n_onderaannemers (0-5 onderaannemers, coördinatie-overhead):
            #   β=0.08 → 4 onderaannemers: exp(4×0.08)=1.38 → 38% langer
            "n_onderaannemers":   {"mean": 0.08,  "sd": 0.04,  "ware_waarde": 0.07},   # !! VALIDEREN
        },
    },
    "C3": {
        "naam":      "Afbouw: Functionele Afwerking",
        # Literatuur: afwerking aardbevingrenovatie ~12-20 werkdagen
        # (Bron: Flapper 2005; eigen schatting Eemsdelta-context)
        "mu_0":      np.log(15),   # !! VALIDEREN: expert-interview (nu: 15 werkdagen)
        "breedte_0": (11, 21),     # afwerking is relatief voorspelbaar
        "ware_mu":   np.log(22),   # simulatiedoel
        "kleur":     "#795548",
        "fase_specifieke_factoren": {
            # afwerkingsniveau (1=basis, 5=hoogwaardig):
            #   β=0.12 → hoogwaardig vs basis (4 niveaus omhoog): exp(4×0.12)=1.62 → 62% langer
            "afwerkingsniveau": {"mean": 0.12,  "sd": 0.06,  "ware_waarde": 0.14},  # !! VALIDEREN
            # schade_pct (% van afwerkoppervlak beschadigd, 0-100):
            #   β=0.004 → 100% beschadigd: exp(100×0.004)=1.49 → 49% langer
            "schade_pct":        {"mean": 0.004, "sd": 0.002, "ware_waarde": 0.005},  # !! VALIDEREN
            # n_wijzigingen (aantal scopewijzigingen tijdens afbouw, 0-10):
            #   β=0.09 → 5 wijzigingen: exp(5×0.09)=1.57 → 57% langer
            "n_wijzigingen":     {"mean": 0.09,  "sd": 0.045, "ware_waarde": 0.10},   # !! VALIDEREN
        },
    },
}


# ==============================================================================
# SECTIE 2: HET MODEL
# ==============================================================================

class BayesianesFaseModel:
    """
    Nederlandstalige schil om `PhaseDurationModel` (uit estimation_module.py)
    voor één EaSiCon-bouwfase.

    BELANGRIJK — wat dit object wel en niet doet:
    Dit object doet ZELF geen Bayesiaanse wiskunde meer (dat deed v1 nog
    wel, met de Normal-Normal update op mu_n/tau_n). In plaats daarvan:
      1. vertaalt het de Nederlandse fase-config (mu_0, breedte_0,
         fase_specifieke_factoren, ...) naar de generieke parameters die
         `PhaseDurationModel.from_elicitation()` verwacht;
      2. onthoudt het een `PhaseDurationModel`-instantie (`self._model`) die
         alle eigenlijke posterior-state bijhoudt (m, V, a, b in NIG-vorm);
      3. vertaalt de output van dat model terug naar dagen en Nederlandse
         namen voor de tabellen/grafieken.

    Wat leert er mee na elk `update()`?
      - de baseline (intercept)                 zoals in v1
      - ELKE covariate-coëfficiënt afzonderlijk  NIEUW t.o.v. v1
      - sigma (project-tot-project ruis)         NIEUW t.o.v. v1
    In v1 stonden de covariate-coëfficiënten en sigma vast; hier zijn het
    net als de baseline kansverdelingen die scherper worden met elk project.
    """

    def __init__(self, code, fase_config):
        """
        Parameters
        ----------
        code        : str  — fasecode, bv. "A", "B1"
        fase_config : dict — het bijbehorende blok uit FASE_INSTELLINGEN
        """
        self.code = code
        self.naam = fase_config["naam"]
        self.fase_config = fase_config

        # Namen van alle covariaten in vaste volgorde: eerst de algemene
        # (GFA, verdiepingen, hoogte), dan de fase-specifieke. Deze volgorde
        # bepaalt de kolomvolgorde van de design-matrix X.
        self.fase_covariate_namen = list(fase_config["fase_specifieke_factoren"].keys())
        self.covariate_namen = ALGEMENE_COVARIATEN + self.fase_covariate_namen

        # Prior-gemiddelden en prior-sd's samenvoegen (algemeen + fase-specifiek)
        coef_prior_means = dict(ALGEMENE_PRIOR_MEAN)
        coef_prior_sds   = dict(ALGEMENE_PRIOR_SD)
        for naam_f, factor in fase_config["fase_specifieke_factoren"].items():
            coef_prior_means[naam_f] = factor["mean"]
            coef_prior_sds[naam_f]   = factor["sd"]

        # baseline_threepoint = (optimistisch, meest_waarschijnlijk, pessimistisch)
        # meest_waarschijnlijk volgt uit mu_0 (dat al log(dagen) is).
        opt, pess = fase_config["breedte_0"]
        meest_waarschijnlijk = np.exp(fase_config["mu_0"])

        # Bouw het onderliggende NIG-regressiemodel. Dit ene aanroep vervangt
        # wat in v1 de losse mu_0/tau_0-velden deden: het legt tegelijk de
        # prior voor intercept, ALLE covariaten, én sigma vast.
        self._model = PhaseDurationModel.from_elicitation(
            phase=code,
            covariate_names=self.covariate_namen,
            baseline_threepoint=(opt, meest_waarschijnlijk, pess),
            coef_prior_means=coef_prior_means,
            coef_prior_sds=coef_prior_sds,
            sigma_prior=SIGMA_LOG_PRIOR,
            sigma_confidence=SIGMA_CONFIDENCE,
        )

        # Bewaar de prior-toestand apart zodat reset() niet opnieuw hoeft te
        # elicteren (en om precies te laten zien wat "voor" en "na" is).
        self._prior_model = PhaseDurationModel(
            phase=code, covariate_names=self.covariate_namen,
            m=self._model.m.copy(), V=self._model.V.copy(),
            a=self._model.a, b=self._model.b,
        )

        self.n_projecten = 0

        # Geschiedenis voor convergentiegrafieken (index 0 = prior, vóór data)
        prior_pred = self._model.predict(np.zeros(len(self.covariate_namen)))
        self.verwacht_geschiedenis = [prior_pred["median_days"]]
        self.lower_geschiedenis    = [prior_pred["interval"][0]]
        self.upper_geschiedenis    = [prior_pred["interval"][1]]
        self.sigma_geschiedenis    = [self._model.sigma_estimate()]
        self.coef_geschiedenis     = [self._model.m.copy()]

    # ------------------------------------------------------------------
    # Interne hulpfunctie: covariate-dict -> geordende vector
    # ------------------------------------------------------------------

    def _naar_vector(self, project_kenmerken=None):
        """Zet een dict {covariate_naam: waarde} om in een (p,) array in de
        vaste kolomvolgorde van self.covariate_namen. Ontbrekende sleutels
        worden als 0 (= referentiewaarde) behandeld."""
        project_kenmerken = project_kenmerken or {}
        return np.array([project_kenmerken.get(naam, 0.0)
                         for naam in self.covariate_namen])

    # ------------------------------------------------------------------
    # Publieke methodes
    # ------------------------------------------------------------------

    def voorspel(self, project_kenmerken=None, interval=0.90):
        """
        Geeft de huidige schatting: verwachte duur + predictie-interval.

        Ga onder de motorkap naar `PhaseDurationModel.predict()`: de
        voorspellende verdeling op log-schaal is een Student-t (niet een
        Normaal zoals in v1), omdat ook de onzekerheid over sigma zelf nu
        meetelt. Bij veel data nadert de Student-t een Normaalverdeling.

        Parameters
        ----------
        project_kenmerken : dict of None — bv. {"GFA": 1.2, "schadegraad": 6}
        interval          : float — betrouwbaarheidsniveau (default 90%)

        Returns
        -------
        tuple: (verwachte_duur, lower, upper) in DAGEN
               verwachte_duur is hier de MEDIAAN (net als v1's gebruik van de
               predictieve verdeling voor het interval; zie estimation_module
               predict() voor de exacte definitie).
        """
        x = self._naar_vector(project_kenmerken)
        pred = self._model.predict(x, interval=interval)
        return pred["median_days"], pred["interval"][0], pred["interval"][1]

    def update(self, geobserveerde_duur, project_kenmerken=None):
        """
        Past de volledige posterior aan na een voltooid project: intercept,
        ELKE covariate-coëfficiënt, én sigma.

        In v1 werd het covariate-effect eerst van de geobserveerde duur AFGE-
        TROKKEN, en leerde alleen de rest (intercept) bij. Hier gebeurt dat
        niet: de covariate-waarden gaan gewoon mee de regressie in, en het
        model bepaalt zelf, gegeven alle data tot nu toe, hoe de duur het
        beste over intercept + covariaten kan worden verdeeld.

        Parameters
        ----------
        geobserveerde_duur : float — werkelijke duur van het project (DAGEN)
        project_kenmerken  : dict of None — kenmerken van het voltooide project
        """
        x = self._naar_vector(project_kenmerken).reshape(1, -1)
        self._model.update(x, np.array([geobserveerde_duur]))

        self.n_projecten += 1
        prior_kenmerken_leeg = np.zeros(len(self.covariate_namen))
        pred = self._model.predict(prior_kenmerken_leeg)
        self.verwacht_geschiedenis.append(pred["median_days"])
        self.lower_geschiedenis.append(pred["interval"][0])
        self.upper_geschiedenis.append(pred["interval"][1])
        self.sigma_geschiedenis.append(self._model.sigma_estimate())
        self.coef_geschiedenis.append(self._model.m.copy())

    def coefficient_summary(self):
        """Posterior gemiddelde + sd per coëfficiënt (incl. intercept)."""
        return self._model.coefficient_summary()

    def sigma_estimate(self):
        """Huidige posterior-schatting van sigma (leert mee, i.t.t. v1)."""
        return self._model.sigma_estimate()

    def reset(self):
        """
        Zet het model terug naar de beginprior (vóór alle updates).
        Gebruik dit om een simulatie-experiment te herhalen (SQ5-verificatie).
        """
        self._model = PhaseDurationModel(
            phase=self.code, covariate_names=self.covariate_namen,
            m=self._prior_model.m.copy(), V=self._prior_model.V.copy(),
            a=self._prior_model.a, b=self._prior_model.b,
        )
        self.n_projecten = 0
        prior_pred = self._model.predict(np.zeros(len(self.covariate_namen)))
        self.verwacht_geschiedenis = [prior_pred["median_days"]]
        self.lower_geschiedenis    = [prior_pred["interval"][0]]
        self.upper_geschiedenis    = [prior_pred["interval"][1]]
        self.sigma_geschiedenis    = [self._model.sigma_estimate()]
        self.coef_geschiedenis     = [self._model.m.copy()]

    def __repr__(self):
        v, l, u = self.voorspel()
        return (
            f"BayesianesFaseModel('{self.naam}', "
            f"verwacht={v:.1f}d, 90%=[{l:.1f}-{u:.1f}]d, n={self.n_projecten})"
        )


# ==============================================================================
# SECTIE 3: SIMULATIE  (SQ5 — verificatie van het leergedrag)
# ==============================================================================
#
# BELANGRIJK VERSCHIL MET v1: in v1 genereerde de simulatie alleen een
# duur uit één "ware_mu" (covariaten deden niet mee, want ze leerden toch
# niet mee). Hier genereert de simulatie OOK random covariaat-waarden per
# gesimuleerd project, en de "ware" duur wordt berekend met de VOLLEDIGE
# lineaire predictor (ware intercept + ware coëfficiënten × covariaten +
# ruis). Zo kun je zien of het model niet alleen de baseline, maar ook elke
# individuele coëfficiënt terugvindt uit de data — inclusief wanneer de
# prior daarover bewust fout stond.

def simuleer_projecten(model, n, seed=42):
    """
    Genereert n gesimuleerde projecten (covariaten + duur) voor verificatie.

    In de echte toepassing vervangt EaSiCon-data deze functie. Hier zijn de
    ware waarden bekend (in tegenstelling tot de praktijk), zodat we kunnen
    controleren of het model naar de ware coëfficiënten convergeert.

    Parameters
    ----------
    model : BayesianesFaseModel — gebruikt om te weten welke covariaten en
            welke "ware_waarde" per covariate horen (uit FASE_INSTELLINGEN)
    n     : int — aantal te simuleren projecten
    seed  : int — voor reproduceerbaarheid

    Returns
    -------
    (X, durations) : covariatenmatrix (n, p) en durations (n,) in DAGEN
    """
    rng = np.random.default_rng(seed)
    p = len(model.covariate_namen)

    # Standaardgenormeerde covariaten (net als in estimation_module/verify_sq5)
    X = rng.normal(size=(n, p))

    # Ware coëfficiënten in dezelfde volgorde als model.covariate_namen
    ware_beta = np.array([
        ALGEMENE_WARE_WAARDE[naam] if naam in ALGEMENE_WARE_WAARDE
        else model.fase_config["fase_specifieke_factoren"][naam]["ware_waarde"]
        for naam in model.covariate_namen
    ])
    ware_mu = model.fase_config["ware_mu"]

    Phi = np.hstack([np.ones((n, 1)), X])
    ware_beta_vol = np.concatenate([[ware_mu], ware_beta])
    log_duur = Phi @ ware_beta_vol + rng.normal(0, SIGMA_LOG_PRIOR, size=n)

    return X, np.exp(log_duur)


def voer_convergentie_experiment_uit(model, n_projecten=100, seed=42):
    """
    Voert het convergentie-experiment uit voor één fase.

    Reset het model, simuleert n_projecten (mét covariaten), en slaat na
    elke update de schatting op. Hiermee toon je SQ5: convergeert het model
    naar de ware waarden als het (bewust) een foute prior heeft?

    Parameters
    ----------
    model       : BayesianesFaseModel
    n_projecten : int
    seed        : int

    Returns
    -------
    dict met sleutels 'verwacht', 'lower', 'upper', 'sigma', 'coefs', 'ware_duur'
    """
    model.reset()
    X, durations = simuleer_projecten(model, n_projecten, seed=seed)

    for t in range(n_projecten):
        kenmerken = dict(zip(model.covariate_namen, X[t]))
        model.update(durations[t], kenmerken)

    ware_mu = model.fase_config["ware_mu"]
    return {
        "verwacht":  np.array(model.verwacht_geschiedenis),
        "lower":     np.array(model.lower_geschiedenis),
        "upper":     np.array(model.upper_geschiedenis),
        "sigma":     np.array(model.sigma_geschiedenis),
        "coefs":     np.array(model.coef_geschiedenis),   # (n+1, p+1)
        "ware_duur": np.exp(ware_mu + SIGMA_LOG_PRIOR**2 / 2),
    }


# ==============================================================================
# SECTIE 4: GRAFIEKEN
# ==============================================================================

def maak_convergentiegrafiek(modellen, n_projecten=100,
                             bestandsnaam="easicon_v2_resultaten.png"):
    """
    Convergentiegrafiek: 2x3 subplots, één per fase (100 projecten).

    Per subplot:
      - Gekleurde lijn    : verwachte duur (mediaan) na elk project
      - Gekleurd vlak     : 90% predictie-interval
      - Zwart gestippeld  : de 'ware' waarde (target voor convergentie)
      - Grijs gestippeld  : initiële prior-schatting
      - Tekstvak rechts   : covariaten van de fase met prior-beta-waarden
    """
    fig, assen = plt.subplots(2, 3, figsize=(16, 10))
    assen = assen.flatten()

    fig.suptitle(
        "EaSiCon v2 — Bayesiaans Duratie-Schattingsmodel (volledige regressie)\n"
        f"Convergentie van verwachte duur per bouwfase (SQ5-simulatie, {n_projecten} projecten)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    milestones = [10, 25, 50]

    for i, (fase_code, model) in enumerate(modellen.items()):
        ax     = assen[i]
        config = FASE_INSTELLINGEN[fase_code]

        resultaat = voer_convergentie_experiment_uit(model, n_projecten, seed=42 + i)

        x         = np.arange(n_projecten + 1)
        verwacht  = resultaat["verwacht"]
        lower     = resultaat["lower"]
        upper     = resultaat["upper"]
        ware_duur = resultaat["ware_duur"]
        prior_duur = verwacht[0]
        kleur      = config["kleur"]

        ax.fill_between(x, lower, upper, alpha=0.20, color=kleur, label="90% interval")
        ax.plot(x, verwacht, color=kleur, linewidth=2.5, label="Modelschatting")
        ax.axhline(ware_duur, color="black", linestyle="--",
                   linewidth=1.5, label=f"Ware waarde ({ware_duur:.0f}d)")
        ax.axhline(prior_duur, color="#888888", linestyle=":",
                   linewidth=1.2, alpha=0.8, label=f"Prior ({prior_duur:.0f}d)")

        for m in milestones:
            if m <= n_projecten:
                ax.axvline(m, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.text(m + 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1,
                        f"n={m}", fontsize=6, color="#888888", va="bottom")

        factoren = config.get("fase_specifieke_factoren", {})
        cov_regels = [
            f"GFA        prior β={BETA_GFA:.3f}/m²",
            f"Verdiep.   prior β={BETA_VERDIEPINGEN:.3f}",
            f"Hoogte     prior β={BETA_HOOGTE:.3f}/m",
            "─────────────────",
        ]
        for naam_f, factor in factoren.items():
            cov_regels.append(f"{naam_f:<16} prior β={factor['mean']:.3f}")
        cov_tekst = "\n".join(cov_regels)
        ax.text(
            0.02, 0.98, cov_tekst,
            transform=ax.transAxes, fontsize=6.2,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#CCCCCC", alpha=0.85),
        )

        ax.set_title(f"Fase {fase_code}: {config['naam']}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Voltooide projecten", fontsize=9)
        ax.set_ylabel("Geschatte duur (dagen)", fontsize=9)
        ax.set_xlim(0, n_projecten)
        ax.set_ylim(0, max(upper.max(), ware_duur) * 1.18)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")

        model.reset()

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafiek opgeslagen: {bestandsnaam}")


def maak_onzekerheidsgrafiek(modellen, n_projecten=100,
                             bestandsnaam="easicon_v2_zekerheid.png"):
    """
    Toont hoe de zekerheid per fase toeneemt na 0-100 projecten.

    VERSCHIL MET v1: v1 toonde tau_n (onzekerheid op de intercept) naast een
    VASTE sigma-lijn. Hier is sigma zelf ook een schatting die convergeert
    (linkerpaneel), naast de intervalsbreedte (rechterpaneel) die nu behalve
    door minder intercept-onzekerheid ook door minder coëfficiënt-onzekerheid
    daalt.
    """
    fig, (ax_sigma, ax_breedte) = plt.subplots(1, 2, figsize=(15, 6))

    fig.suptitle(
        f"EaSiCon v2 — Zekerheidstoename per bouwfase ({n_projecten} projecten)\n"
        "Links = geleerde sigma convergeert naar ware ruis  |  "
        "Rechts = 90%-intervalsbreedte daalt (intercept + covariaten samen)",
        fontsize=12, fontweight="bold",
    )

    milestones = [10, 25, 50]

    for i, (fase_code, model) in enumerate(modellen.items()):
        config = FASE_INSTELLINGEN[fase_code]
        kleur  = config["kleur"]
        label  = f"{fase_code}: {config['naam']}"

        resultaat = voer_convergentie_experiment_uit(model, n_projecten, seed=42 + i)

        x       = np.arange(n_projecten + 1)
        sigma   = resultaat["sigma"]
        breedte = resultaat["upper"] - resultaat["lower"]

        ax_sigma.plot(x, sigma, color=kleur, linewidth=2.2, label=label)
        ax_sigma.annotate(f"{sigma[-1]:.3f}", xy=(n_projecten, sigma[-1]),
                          fontsize=6.5, color=kleur,
                          xytext=(3, 0), textcoords="offset points", va="center")

        ax_breedte.plot(x, breedte, color=kleur, linewidth=2.2, label=label)

        model.reset()

    ax_sigma.axhline(SIGMA_LOG_PRIOR, color="#CCCCCC", linestyle=":", linewidth=1.0)
    ax_sigma.text(1, SIGMA_LOG_PRIOR + 0.01,
                  f"ware sigma (simulatie) = {SIGMA_LOG_PRIOR}",
                  fontsize=7, color="#999999", va="bottom")

    for ax in (ax_sigma, ax_breedte):
        for m in milestones:
            ax.axvline(m, color="#DDDDDD", linestyle="--", linewidth=0.9)
            ax.text(m + 0.5, 0, f"n={m}", fontsize=6.5, color="#AAAAAA", va="bottom")
        ax.set_xlabel("Voltooide projecten", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0, n_projecten)
        ax.set_ylim(bottom=0)

    ax_sigma.set_title(
        "Geleerde sigma (ruis op log-schaal) per fase\n"
        "NIEUW t.o.v. v1: sigma was daar een vaste constante, leert hier mee",
        fontsize=10, fontweight="bold",
    )
    ax_sigma.set_ylabel("sigma-schatting (posterior mean)", fontsize=10)

    ax_breedte.set_title(
        "90%-intervalsbreedte per fase (dagen)\n"
        "daalt door minder onzekerheid over intercept ÉN covariaten",
        fontsize=10, fontweight="bold",
    )
    ax_breedte.set_ylabel("Upper - Lower  (dagen)", fontsize=10)

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafiek opgeslagen: {bestandsnaam}")


def maak_coefficientgrafiek(modellen, n_projecten=100,
                            bestandsnaam="easicon_v2_coefficienten.png"):
    """
    NIEUW t.o.v. v1 (had geen equivalent, want coëfficiënten leerden niet mee).

    Toont per fase de convergentie van elke covariate-coëfficiënt naar zijn
    ware waarde. Dit is het directe bewijs dat de coëfficiënten zelf ook
    leren: elke gekleurde lijn hoort bij één covariate en moet naar de
    bijbehorende gestippelde "ware waarde"-lijn toe bewegen.
    """
    fig, assen = plt.subplots(2, 3, figsize=(17, 10))
    assen = assen.flatten()

    fig.suptitle(
        "EaSiCon v2 — Convergentie van covariate-coëfficiënten per fase\n"
        "Elke lijn = één coëfficiënt; gestippeld = bijbehorende ware waarde (simulatie)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for i, (fase_code, model) in enumerate(modellen.items()):
        ax     = assen[i]
        config = FASE_INSTELLINGEN[fase_code]

        resultaat = voer_convergentie_experiment_uit(model, n_projecten, seed=42 + i)
        coefs = resultaat["coefs"]          # (n+1, p+1), kolom 0 = intercept

        namen = ["intercept"] + model.covariate_namen
        ware_beta = np.concatenate([
            [config["ware_mu"]],
            [ALGEMENE_WARE_WAARDE[n] if n in ALGEMENE_WARE_WAARDE
             else config["fase_specifieke_factoren"][n]["ware_waarde"]
             for n in model.covariate_namen],
        ])

        x = np.arange(n_projecten + 1)
        kleuren = plt.cm.tab10(np.linspace(0, 1, len(namen)))
        for j, naam in enumerate(namen):
            if naam == "intercept":
                continue  # aparte schaal (dagen op log-schaal vs. kleine beta's); zie resultatengrafiek
            ax.plot(x, coefs[:, j], color=kleuren[j], linewidth=1.8, label=naam)
            ax.axhline(ware_beta[j], color=kleuren[j], linestyle=":", linewidth=1.0)

        ax.set_title(f"Fase {fase_code}: {config['naam']}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Voltooide projecten", fontsize=9)
        ax.set_ylabel("Coëfficiëntwaarde (posterior mean)", fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0, n_projecten)

        model.reset()

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafiek opgeslagen: {bestandsnaam}")


# ==============================================================================
# SECTIE 5: TEKSTOUTPUT
# ==============================================================================

def print_schattingstabel(modellen):
    """Print overzichtstabel met huidige schattingen + covariaten per fase."""
    breedte = 88
    print("\n" + "=" * breedte)
    print("  EaSiCon v2 — Huidige faseschattingen  (prior, vóór projectdata)")
    print("=" * breedte)
    print(f"  {'Fase':<5} {'Naam':<36} {'Verwacht':>9} {'90%-interval':>20}")
    print("-" * breedte)

    for fase_code, model in modellen.items():
        config = FASE_INSTELLINGEN[fase_code]
        naam   = config["naam"]
        v, l, u = model.voorspel()
        print(f"  {fase_code:<5} {naam:<36} {v:>7.1f}d  [{l:>5.1f} - {u:>6.1f}d]")
        print(f"  {'':>5}  sigma (ruis, leert mee) = {model.sigma_estimate():.3f}")

        print(f"  {'':>5}  ├─ Algemeen :  "
              f"GFA prior β={BETA_GFA:.3f}/m²  |  "
              f"Verdiep. prior β={BETA_VERDIEPINGEN:.2f}  |  "
              f"Hoogte prior β={BETA_HOOGTE:.3f}/m")

        factoren = config.get("fase_specifieke_factoren", {})
        items    = list(factoren.items())
        for j, (naam_f, factor) in enumerate(items):
            prefix = "  └─" if j == len(items) - 1 else "  ├─"
            print(f"  {'':>5} {prefix} {naam_f:<22}  prior β = {factor['mean']:+.4f}"
                  f"  (sd={factor['sd']:.4f})")
        print()

    print("-" * breedte)
    print("  * Alle waarden zijn PLACEHOLDERS op basis van literatuurschattingen.")
    print("    Vervang mu_0, breedte_0 en per-factor mean/sd na expert-interviews.")
    print("    NIEUW t.o.v. v1: elke β hierboven is een startwaarde die met elk")
    print("    voltooid project verder bijgesteld wordt (zie coefficient_summary()).")
    print("=" * breedte)


def print_todo_lijst():
    """Print gestructureerde lijst van openstaande acties."""
    todos = [
        {
            "nr": 1,
            "wat": "Interceptwaarden (mu_0) per fase instellen na interviews",
            "hoe": (
                'Stel per fase: "Hoe lang duurt fase X gemiddeld? Geef een optimistische,\n'
                "         meest waarschijnlijke en pessimistische schatting in dagen.\"\n"
                "         Zet mu_0 = log(meest_waarschijnlijk) en breedte_0 = (opt, pess)\n"
                "         in FASE_INSTELLINGEN."
            ),
            "locatie": "FASE_INSTELLINGEN -> mu_0, breedte_0 per fase",
        },
        {
            "nr": 2,
            "wat": "Prior-sd per coëfficiënt bepalen (hoe zeker ben je van elk effect?)",
            "hoe": (
                "Voor elke covariate: hoe zeker is de expert van de genoemde\n"
                "         coëfficiënt? Kleinere sd = stelliger prior = langzamer bijgesteld\n"
                "         door nieuwe projectdata. Vuistregel: sd ≈ 0.5 × |mean| als startpunt."
            ),
            "locatie": "FASE_INSTELLINGEN -> fase_specifieke_factoren[...]['sd'], en *_SD-constanten",
        },
        {
            "nr": 3,
            "wat": "Fase-specifieke covariaten valideren (3 per fase)",
            "hoe": (
                'Stel per fase: "Welke 3 projectkenmerken bepalen het meest\n'
                "         hoe lang fase X duurt?\" Stel mean+sd in op basis van de\n"
                "         interviews. Eenheden vastleggen en documenteren."
            ),
            "locatie": "FASE_INSTELLINGEN -> fase_specifieke_factoren per fase",
        },
        {
            "nr": 4,
            "wat": "Algemene covariate-coëfficiënten valideren voor Eemsdelta-woningen",
            "hoe": (
                "BETA_GFA (0.003/m²), BETA_VERDIEPINGEN (0.10), BETA_HOOGTE (0.020/m)\n"
                "         zijn literatuurplaceholders. Controleer orde van grootte voor\n"
                "         typische Eemsdelta-woning: ~100 m², 1-2 lagen, ~6-8 m hoog."
            ),
            "locatie": "BETA_GFA, BETA_VERDIEPINGEN, BETA_HOOGTE — bovenaan script",
        },
        {
            "nr": 5,
            "wat": "Startpunt sigma (SIGMA_LOG_PRIOR) en vertrouwen (SIGMA_CONFIDENCE) valideren",
            "hoe": (
                "SIGMA_LOG_PRIOR = 0.35 -> ~35% variatie tussen vergelijkbare projecten,\n"
                "         VOORDAT covariaten dit verklaren. Bespreek met Ragnar Klabbers of\n"
                "         dit realistisch is; SIGMA_CONFIDENCE bepaalt hoe snel dit bijstelt."
            ),
            "locatie": "SIGMA_LOG_PRIOR, SIGMA_CONFIDENCE — bovenaan script",
        },
        {
            "nr": 6,
            "wat": "SQ5-verificatie afronden: convergentie van ELKE coëfficiënt controleren",
            "hoe": (
                "Voer uit nadat echte priors zijn ingesteld:\n"
                "         (a) Convergentie: beweegt elke coëfficiënt (niet alleen de\n"
                "             baseline) richting zijn ware waarde?\n"
                "         (b) Correctie: corrigeert het model een bewust foute prior\n"
                "             op een coëfficiënt (bv. verkeerd teken)?\n"
                "         (c) Sigma: convergeert de geleerde ruis naar de ware ruis?\n"
                "         -> easicon_v2_resultaten.png + _zekerheid.png + _coefficienten.png"
            ),
            "locatie": "voer_convergentie_experiment_uit()  /  drie PNG-bestanden",
        },
        {
            "nr": 7,
            "wat": "Echte EaSiCon-projectdata koppelen zodra beschikbaar",
            "hoe": (
                "Vervang simuleer_projecten() door een functie die data inleest\n"
                "         vanuit het EaSiCon datamanagementsysteem (VolkerWessels).\n"
                "         Roep daarna aan: model.update(duur_in_dagen, project_kenmerken),\n"
                "         waarbij project_kenmerken een dict is met dezelfde covariate-\n"
                "         namen als in FASE_INSTELLINGEN (bv. {'GFA': 95, 'schadegraad': 6})."
            ),
            "locatie": "Sectie 3 — simuleer_projecten()",
        },
    ]

    breedte = 74
    print("\n" + "=" * breedte)
    print("  OPENSTAANDE TODO's — In te vullen na expert-interviews")
    print("=" * breedte)

    for todo in todos:
        print(f"\n  [{todo['nr']}] {todo['wat']}")
        print(f"      Hoe     : {todo['hoe']}")
        print(f"      Locatie : {todo['locatie']}")

    print("\n" + "-" * breedte)
    print(f"  Totaal openstaande TODO's: {len(todos)}")
    print("=" * breedte + "\n")


# ==============================================================================
# HOOFDPROGRAMMA
# ==============================================================================

def main():
    N_PROJECTEN = 100  # simulatielengte voor alle grafieken

    print("\nEaSiCon v2 Bayesiaans Duratie-Schattingsmodel (volledige regressie)")
    print("Gijs Meijers (4957822) | TU Delft x Haskoning | 2026")

    # Maak modelinstanties voor alle 6 fases
    modellen = {
        code: BayesianesFaseModel(code, cfg)
        for code, cfg in FASE_INSTELLINGEN.items()
    }

    # 1. Tabel met huidige schattingen + covariaten (prior-stand)
    print_schattingstabel(modellen)

    # 2. Openstaande TODO's
    print_todo_lijst()

    # 3. Convergentiegrafiek: verwachte duur per fase (100 projecten)
    print("  Grafieken worden aangemaakt...")
    maak_convergentiegrafiek(modellen, n_projecten=N_PROJECTEN)

    # 4. Zekerheidstoename-grafiek (sigma + intervalsbreedte over 100 projecten)
    maak_onzekerheidsgrafiek(modellen, n_projecten=N_PROJECTEN)

    # 5. NIEUW: convergentie van elke coëfficiënt afzonderlijk
    maak_coefficientgrafiek(modellen, n_projecten=N_PROJECTEN)

    print("\nKlaar.")
    print("  easicon_v2_resultaten.png     — convergentie verwachte duur per fase")
    print("  easicon_v2_zekerheid.png     — sigma-convergentie + intervalsbreedte")
    print("  easicon_v2_coefficienten.png — NIEUW: convergentie per covariate-coëfficiënt")
    print()


if __name__ == "__main__":
    main()
