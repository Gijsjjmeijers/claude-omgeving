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

WIJZIGINGEN IN DEZE VERSIE (juli 2026) — na review van de eerste v2-opzet
-------------------------------------------------------------------------
1. mu_0 telt nu écht mee. `threepoint_to_lognormal()` in estimation_module.py
   gebruikt voortaan de meest-waarschijnlijke waarde als mediaan op log-schaal;
   voorheen werd mu_0 stilzwijgend genegeerd en verving het midden van
   (log opt, log pess) de geconfigureerde startschatting.
2. REFERENTIEWONING & CENTRERING. Covariaten gaan in RUWE eenheden het model
   in (GFA in m², schadegraad 0-10, ...) en worden intern gecentreerd rond een
   expliciete referentiewaarde per covariaat. De intercept is daardoor de
   log-duur van de "referentiewoning" (100 m², 1 laag, 6 m, geen schade, ...)
   in plaats van een onzinnige "woning van 0 m²". Eerder was de conventie
   inconsistent: priors in fysieke eenheden, simulatie in N(0,1), en de
   documentatie vroeg om ruwe waarden — drie verschillende aannames.
3. WARE_SIGMA is losgekoppeld van SIGMA_LOG_PRIOR. De simulatie gebruikte de
   prior-constante ook als "ware" ruis, waardoor de sigma-convergentietest
   triviaal was (het model startte al op de waarheid). Nu start sigma bewust
   fout (0.35) en moet het naar de ware ruis (0.45) toe leren.
4. Mediaan vs. mediaan in de grafieken. De "ware waarde"-lijn was het
   GEMIDDELDE van de ware lognormaal (exp(mu + sigma²/2)) terwijl de
   modellijn de MEDIAAN toont; het model leek daardoor structureel ~6% te
   laag te convergeren. Beide zijn nu de mediaan.
5. Realistische simulatie. Covariaten worden getrokken uit realistische
   verdelingen per covariaat (binaire asbest is echt 0/1, GFA ~ N(100,15),
   schalen 0-10 zijn gehele scores, ...) in plaats van allemaal N(0,1).
6. Elke fase wordt nog maar ÉÉN keer gesimuleerd; alle drie de grafieken
   delen dezelfde run (voorheen 3x dezelfde simulatie per fase).

Theorie (zie thesis H5/H6, Kim & Reinschmidt 2009, Gelman et al. 2013):
  - log(duur) ~ Normaal(mu, sigma²)             [lognormaal voor positieve durations]
  - mu = beta_0f + som_j(beta_jf * x_j) + som_k(gamma_kf * z_kf)   (lineaire predictor)
  - x en z zijn GECENTREERDE covariaten: (ruwe waarde - referentiewaarde)
  - Prior op (beta, sigma²): Normal-Inverse-Gamma (conjugate met bovenstaande)
  - Update: exact, gesloten vorm na elk project (geen MCMC nodig)

Gebruik : python easicon_model_v2.py
Output  :
  1. Tabel — huidige schatting per fase in dagen, incl. covariaten + hun onzekerheid
  2. TODO-lijst — openstaande punten na interviews
  3. easicon_v2_resultaten.png     — convergentie van de mediaan-duur per fase (100 projecten)
  4. easicon_v2_zekerheid.png     — zekerheidstoename per fase (intervalsbreedte + sigma-schatting)
  5. easicon_v2_coefficienten.png — convergentie van de covariaat-coëfficiënten zelf

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

# --- Schakelaar: vlakke (niet-informatieve) prior i.p.v. literatuurwaarden ---
# Zet dit op True om het model te laten starten met VRIJWEL 0 informatie:
# alle coëfficiënten beginnen op 0 met een zeer brede onzekerheid, in plaats
# van op de literatuurschattingen hieronder (BETA_GFA, schadegraad, etc.).
# De literatuurwaarden in FASE_INSTELLINGEN blijven dan in de code staan,
# maar worden GENEGEERD bij het bouwen van het model — ze dienen dan alleen
# nog als leesbare documentatie/referentie, niet als startpunt.
#
# Waarom dit geen "0.0 sd" of "geen prior" kan zijn: de NIG-wiskunde in
# estimation_module.py vereist een geldige (eindige) V-matrix en (a, b) om
# de eerste update() te kunnen doen. "0 informatie" wordt hier daarom
# benaderd met een extreem brede prior (VLAKKE_PRIOR_SD, VLAKKE_SIGMA_CONFIDENCE)
# in plaats van een letterlijk oneindige — dat laatste is wiskundig ongedefinieerd.
# Met deze instelling bepalen UITSLUITEND de expert-scenario's (via
# elicitatie_verwerker.py) of latere projectdata waar de coëfficiënten landen.
VLAKKE_PRIOR = True

# Hoe breed de "ik weet niets"-prior is voor elke coëfficiënt (intercept en
# covariaten). LET OP de betekenis: in de NIG-familie is de coëfficiëntprior
# N(m, sigma² · V), dus V is een SCHAAL t.o.v. sigma. Met VLAKKE_PRIOR_SD = 10
# en sigma ≈ 0.35 is de effectieve prior-sd per coëfficiënt ≈ 10 × 0.35 = 3.5
# op log-schaal (een factor e^3.5 ≈ 33 op de duur) — ruimschoots "weet niets",
# maar klein genoeg om numeriek stabiel te blijven.
VLAKKE_PRIOR_SD = 10.0

# Hoeveel "pseudo-projecten" de vlakke sigma-prior weegt: laag, zodat de
# eerste paar scenario's/projecten de sigma-schatting al snel overnemen.
# NB: dit geeft a0 = 1.5 < 2, dus een Inverse-Gamma-prior met oneindige
# variantie — bewust, want dat is precies "vrijwel geen informatie".
VLAKKE_SIGMA_CONFIDENCE = 1.0

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

# --- "Ware" ruis, ALLEEN voor de simulatie (SQ5-verificatie) ---
# Bewust ANDERS dan SIGMA_LOG_PRIOR: de sigma-convergentietest is alleen
# informatief als het model met een foute startwaarde (0.35) naar de ware
# ruis (0.45) toe moet leren. Waren deze gelijk (zoals in de eerste v2-opzet),
# dan startte de sigma-lijn al op de waarheid en bewees de grafiek niets.
# In de praktijk is deze waarde uiteraard onbekend.
WARE_SIGMA = 0.45

# --- Algemene covariate-coëfficiënten (alle fases) ---
# Dit zijn nu PRIOR-GEMIDDELDEN, geen vaste effecten zoals in v1: het model
# leert na elk project een bijgestelde waarde. De onzekerheid rond elke prior
# wordt apart ingesteld via *_SD hieronder (kleiner = stelliger prior, buigt
# trager om bij nieuwe data).
# LITERATUURSCHATTINGEN — bronnen: Lawal et al. 2023, Kim et al. 2019,
#   Chan & Kumaraswamy 1999, Odeh & Battaineh 2002.
# Interpretatie: effect = exp(β × (waarde - referentie)) als factor op de duur.
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

# --- DE REFERENTIEWONING ---
# Covariaten gaan in RUWE eenheden het model in en worden intern gecentreerd:
# het model rekent met (waarde - referentie). De intercept (mu_0) is daardoor
# de log-duur van deze concrete referentiewoning, niet van een onzinnige
# "woning met alle covariaten op 0" (0 m², 0 verdiepingen, ...).
# Referentie hier: typische Eemsdelta-rijwoning, 100 m² GFA, 1 verdieping,
# 6 m hoog, zonder schade, goed bereikbaar, basis-afwerkingsniveau.
# !! VALIDEREN: leg in het interview expliciet vast op welke referentiewoning
#    de baseline-schatting (mu_0/breedte_0) betrekking heeft.
ALGEMENE_REFERENTIE = {"GFA": 100.0, "verdiepingen": 1.0, "hoogte": 6.0}

# Hoe de simulatie (SectIE 3) realistische ruwe covariaatwaarden trekt.
# Vormen: ("normaal", gemiddelde, sd), ("uniform", laag, hoog),
#         ("uniform_int", laag, hoog), ("keuze", [waarden], [kansen]),
#         ("binair", kans_op_1)
ALGEMENE_SIM = {
    "GFA":          ("normaal", 100.0, 15.0),          # m², typisch 70-130
    "verdiepingen": ("keuze", [1.0, 2.0], [0.6, 0.4]), # meest 1, soms 2 lagen
    "hoogte":       ("normaal", 6.5, 1.0),             # m nokhoogte
}

# --- Fase-instellingen ---
# Per fase:
#   mu_0       : startschatting op log-schaal = log(verwachte dagen), prior-
#                MEDIAAN van de baseline (intercept), voor de REFERENTIEWONING
#                (alle covariaten op hun referentiewaarde). Wordt via
#                exp(mu_0) als meest-waarschijnlijke waarde in de drie-punts-
#                schatting gebruikt en telt dus écht mee (zie wijziging 1).
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
#                "referentie"  referentiewaarde (centrering); weggelaten = 0
#                "sim"         hoe de simulatie deze covariaat trekt (zie boven)

# Toelichting op de LITERATUURSCHATTINGEN hieronder
# ─────────────────────────────────────────────────
# mu_0  : log van de verwachte duur in werkdagen (literatuur + expert-kennis TU Delft).
#          !! VALIDEREN: vervang met drie-punts-schatting uit expert-interview.
# breedte_0 : (optimistisch, pessimistisch) in dagen, gebruikt om tau_0 af te leiden.
#          !! VALIDEREN: uit expert-interview.
# beta / gamma "mean"/"sd" : effect op log(duur) per RUWE eenheid covariate
#          (per m², per schaalpunt, ...), MET bijbehorende prior-sd (hoe zeker
#          je ervan bent). Praktisch: exp(β × (waarde - referentie)) = factor
#          op de duur.
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
            "schadegraad":    {"mean": 0.04, "sd": 0.02,  "ware_waarde": 0.05,
                               "sim": ("uniform_int", 0, 10)},               # !! VALIDEREN
            # asbest aanwezig (0=nee, 1=ja):
            #   β=0.30 → aanwezig: exp(0.30)=1.35 → 35% langer (sanering + herplanning)
            "asbest":         {"mean": 0.30, "sd": 0.15,  "ware_waarde": 0.40,
                               "sim": ("binair", 0.35)},                     # !! VALIDEREN
            # bereikbaarheid (1=goed, 5=slecht); referentie = 1 (goed):
            #   β=0.07 → slechtste score (5 vs 1 = +4): exp(4×0.07)=1.32 → 32% langer
            "bereikbaarheid": {"mean": 0.07, "sd": 0.035, "ware_waarde": 0.06,
                               "referentie": 1, "sim": ("uniform_int", 1, 5)},  # !! VALIDEREN
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
            "funderingsschade": {"mean": 0.06, "sd": 0.03,  "ware_waarde": 0.08,
                                 "sim": ("uniform_int", 0, 10)},             # !! VALIDEREN
            # bodemtype (0=zand, 1=klei, 2=veen):
            #   β=0.18 → veen vs zand: exp(2×0.18)=1.43 → 43% langer
            "bodemtype":        {"mean": 0.18, "sd": 0.09,  "ware_waarde": 0.15,
                                 "sim": ("keuze", [0, 1, 2], [0.3, 0.5, 0.2])},  # !! VALIDEREN
            # onderbemaling noodzakelijk (0=nee, 1=ja):
            #   β=0.25 → noodzakelijk: exp(0.25)=1.28 → 28% langer
            "onderbemaling":    {"mean": 0.25, "sd": 0.125, "ware_waarde": 0.30,
                                 "sim": ("binair", 0.30)},                   # !! VALIDEREN
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
            "casoschade":        {"mean": 0.05, "sd": 0.025, "ware_waarde": 0.06,
                                  "sim": ("uniform_int", 0, 10)},            # !! VALIDEREN
            # bouwmethode (0=prefab, 1=traditioneel, 2=complex maatwerk):
            #   β=0.20 → complex vs prefab: exp(2×0.20)=1.49 → 49% langer
            "bouwmethode":       {"mean": 0.20, "sd": 0.10,  "ware_waarde": 0.18,
                                  "sim": ("keuze", [0, 1, 2], [0.4, 0.4, 0.2])},  # !! VALIDEREN
            # structuuringrepen (0-5 ingrijpende aanpassingen):
            #   β=0.10 → 5 ingrepen: exp(5×0.10)=1.65 → 65% langer
            "structuuringrepen": {"mean": 0.10, "sd": 0.05,  "ware_waarde": 0.12,
                                  "sim": ("uniform_int", 0, 5)},             # !! VALIDEREN
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
            "geveltype": {"mean": 0.18,  "sd": 0.09,  "ware_waarde": 0.20,
                          "sim": ("keuze", [0, 1, 2], [0.5, 0.3, 0.2])},     # !! VALIDEREN
            # daktype (0=plat dak, 1=hellend, 2=mansarde/complex):
            #   β=0.15 → mansarde vs plat: exp(2×0.15)=1.35 → 35% langer
            "daktype":   {"mean": 0.15,  "sd": 0.075, "ware_waarde": 0.13,
                          "sim": ("keuze", [0, 1, 2], [0.2, 0.6, 0.2])},     # !! VALIDEREN
            # schade_m2 (m² beschadigde gevelbekleding, typisch 0-60 m²):
            #   β=0.008 → 30 m² schade: exp(30×0.008)=1.27 → 27% langer
            "schade_m2": {"mean": 0.008, "sd": 0.004, "ware_waarde": 0.010,
                          "sim": ("uniform", 0, 60)},                        # !! VALIDEREN
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
            "elektra_pct":        {"mean": 0.004, "sd": 0.002, "ware_waarde": 0.005,
                                   "sim": ("uniform", 0, 100)},              # !! VALIDEREN
            # verwarmingssysteem (0=handhaven, 1=upgraden, 2=volledig vervangen):
            #   β=0.22 → volledig vervangen vs handhaven: exp(2×0.22)=1.55 → 55% langer
            "verwarmingssysteem": {"mean": 0.22,  "sd": 0.11,  "ware_waarde": 0.25,
                                   "sim": ("keuze", [0, 1, 2], [0.3, 0.4, 0.3])},  # !! VALIDEREN
            # n_onderaannemers (0-5 onderaannemers, coördinatie-overhead):
            #   β=0.08 → 4 onderaannemers: exp(4×0.08)=1.38 → 38% langer
            "n_onderaannemers":   {"mean": 0.08,  "sd": 0.04,  "ware_waarde": 0.07,
                                   "sim": ("uniform_int", 0, 5)},            # !! VALIDEREN
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
            # afwerkingsniveau (1=basis, 5=hoogwaardig); referentie = 1 (basis):
            #   β=0.12 → hoogwaardig vs basis (4 niveaus omhoog): exp(4×0.12)=1.62 → 62% langer
            "afwerkingsniveau": {"mean": 0.12,  "sd": 0.06,  "ware_waarde": 0.14,
                                 "referentie": 1, "sim": ("uniform_int", 1, 5)},  # !! VALIDEREN
            # schade_pct (% van afwerkoppervlak beschadigd, 0-100):
            #   β=0.004 → 100% beschadigd: exp(100×0.004)=1.49 → 49% langer
            "schade_pct":        {"mean": 0.004, "sd": 0.002, "ware_waarde": 0.005,
                                  "sim": ("uniform", 0, 100)},               # !! VALIDEREN
            # n_wijzigingen (aantal scopewijzigingen tijdens afbouw, 0-10):
            #   β=0.09 → 5 wijzigingen: exp(5×0.09)=1.57 → 57% langer
            "n_wijzigingen":     {"mean": 0.09,  "sd": 0.045, "ware_waarde": 0.10,
                                  "sim": ("uniform_int", 0, 10)},            # !! VALIDEREN
        },
    },
}


# ==============================================================================
# SECTIE 2: HET MODEL
# ==============================================================================

class BayesiaansFaseModel:
    """
    Nederlandstalige schil om `PhaseDurationModel` (uit estimation_module.py)
    voor één EaSiCon-bouwfase.

    BELANGRIJK — wat dit object wel en niet doet:
    Dit object doet ZELF geen Bayesiaanse wiskunde meer (dat deed v1 nog
    wel, met de Normal-Normal update op mu_n/tau_n). In plaats daarvan:
      1. vertaalt het de Nederlandse fase-config (mu_0, breedte_0,
         fase_specifieke_factoren, ...) naar de generieke parameters die
         `PhaseDurationModel.from_elicitation()` verwacht;
      2. CENTREERT het alle covariaten rond hun referentiewaarde, zodat de
         gebruiker ruwe waarden kan doorgeven ({"GFA": 95, "schadegraad": 6})
         en de intercept de referentiewoning beschrijft;
      3. onthoudt het een `PhaseDurationModel`-instantie (`self._model`) die
         alle eigenlijke posterior-state bijhoudt (m, V, a, b in NIG-vorm);
      4. vertaalt de output van dat model terug naar dagen en Nederlandse
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

        # Referentiewaarden (voor centrering) en simulatie-specificaties,
        # in dezelfde vaste volgorde.
        self.referenties = dict(ALGEMENE_REFERENTIE)
        self.sim_specs   = dict(ALGEMENE_SIM)
        for naam_f, factor in fase_config["fase_specifieke_factoren"].items():
            self.referenties[naam_f] = float(factor.get("referentie", 0.0))
            self.sim_specs[naam_f]   = factor["sim"]

        # Prior-gemiddelden en prior-sd's samenvoegen (algemeen + fase-specifiek)
        coef_prior_means = dict(ALGEMENE_PRIOR_MEAN)
        coef_prior_sds   = dict(ALGEMENE_PRIOR_SD)
        for naam_f, factor in fase_config["fase_specifieke_factoren"].items():
            coef_prior_means[naam_f] = factor["mean"]
            coef_prior_sds[naam_f]   = factor["sd"]

        if VLAKKE_PRIOR:
            # "0 informatie"-modus: negeer de literatuurwaarden in
            # FASE_INSTELLINGEN volledig. Alle coëfficiënten (intercept +
            # covariaten) starten op 0 met een zeer brede onzekerheid, en
            # sigma start op SIGMA_LOG_PRIOR maar met minimaal vertrouwen
            # (VLAKKE_SIGMA_CONFIDENCE), zodat de eerste paar scenario's of
            # projecten de schatting vrijwel volledig bepalen.
            #
            # We bouwen dit rechtstreeks op in NIG-vorm (m, V, a, b) in
            # plaats van via from_elicitation(), omdat from_elicitation()
            # een informatieve drie-punts-schatting als input verwacht — die
            # willen we hier expliciet NIET gebruiken.
            # NB: V is de NIG-schaal t.o.v. sigma², dus de effectieve
            # prior-sd per coëfficiënt is VLAKKE_PRIOR_SD × sigma (zie de
            # toelichting bij VLAKKE_PRIOR_SD bovenaan).
            p = len(self.covariate_namen)
            m0 = np.zeros(p + 1)                       # alles op 0: intercept + covariaten
            V0 = np.eye(p + 1) * (VLAKKE_PRIOR_SD ** 2) # zeer brede onzekerheid
            a0 = VLAKKE_SIGMA_CONFIDENCE / 2.0 + 1.0
            b0 = SIGMA_LOG_PRIOR ** 2 * (a0 - 1.0)
            self._model = PhaseDurationModel(
                phase=code, covariate_names=self.covariate_namen,
                m=m0, V=V0, a=a0, b=b0,
            )
        else:
            # baseline_threepoint = (optimistisch, meest_waarschijnlijk, pessimistisch)
            # meest_waarschijnlijk volgt uit mu_0 (dat al log(dagen) is) en
            # wordt door threepoint_to_lognormal() als prior-mediaan gebruikt.
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
        self._start_geschiedenis()

    def _start_geschiedenis(self):
        """(Her)initialiseer de geschiedenis voor de convergentiegrafieken
        (index 0 = prior, vóór data). Voorspelling voor de referentiewoning."""
        prior_pred = self._model.predict(np.zeros(len(self.covariate_namen)))
        self.verwacht_geschiedenis = [prior_pred["median_days"]]
        self.lower_geschiedenis    = [prior_pred["interval"][0]]
        self.upper_geschiedenis    = [prior_pred["interval"][1]]
        self.sigma_geschiedenis    = [self._model.sigma_estimate()]
        self.coef_geschiedenis     = [self._model.m.copy()]

    # ------------------------------------------------------------------
    # Interne hulpfunctie: covariate-dict -> geordende, GECENTREERDE vector
    # ------------------------------------------------------------------

    def _naar_vector(self, project_kenmerken=None):
        """Zet een dict {covariate_naam: ruwe waarde} om in een (p,) array in
        de vaste kolomvolgorde van self.covariate_namen, GECENTREERD rond de
        referentiewaarden: (waarde - referentie). Ontbrekende sleutels krijgen
        de referentiewaarde (= gecentreerd 0), zodat "niet opgegeven" altijd
        "zoals de referentiewoning" betekent."""
        project_kenmerken = project_kenmerken or {}
        return np.array([
            project_kenmerken.get(naam, self.referenties[naam]) - self.referenties[naam]
            for naam in self.covariate_namen
        ])

    # ------------------------------------------------------------------
    # Publieke methodes
    # ------------------------------------------------------------------

    def voorspel(self, project_kenmerken=None, interval=0.90):
        """
        Geeft de huidige schatting: mediaan-duur + predictie-interval.

        Ga onder de motorkap naar `PhaseDurationModel.predict()`: de
        voorspellende verdeling op log-schaal is een Student-t (niet een
        Normaal zoals in v1), omdat ook de onzekerheid over sigma zelf nu
        meetelt. Bij veel data nadert de Student-t een Normaalverdeling.

        Parameters
        ----------
        project_kenmerken : dict of None — RUWE waarden, bv.
                            {"GFA": 110, "schadegraad": 6}; centrering rond de
                            referentiewoning gebeurt intern. None = de
                            referentiewoning zelf.
        interval          : float — betrouwbaarheidsniveau (default 90%)

        Returns
        -------
        tuple: (mediaan_duur, lower, upper) in DAGEN
               mediaan_duur is de MEDIAAN van de voorspellende verdeling
               (het gemiddelde van een log-Student-t bestaat niet; de mediaan
               is de robuuste puntschatting hier).
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
        project_kenmerken  : dict of None — RUWE kenmerken van het voltooide
                             project, bv. {"GFA": 95, "schadegraad": 6};
                             centrering gebeurt intern.
        """
        x = self._naar_vector(project_kenmerken).reshape(1, -1)
        self._model.update(x, np.array([geobserveerde_duur]))

        self.n_projecten += 1
        referentie_x = np.zeros(len(self.covariate_namen))
        pred = self._model.predict(referentie_x)
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
        self._start_geschiedenis()

    def __repr__(self):
        v, l, u = self.voorspel()
        return (
            f"BayesiaansFaseModel('{self.naam}', "
            f"mediaan={v:.1f}d, 90%=[{l:.1f}-{u:.1f}]d, n={self.n_projecten})"
        )


# Oude (typefout-)naam als alias, zodat bestaande imports blijven werken.
BayesianesFaseModel = BayesiaansFaseModel


# ==============================================================================
# SECTIE 3: SIMULATIE  (SQ5 — verificatie van het leergedrag)
# ==============================================================================
#
# BELANGRIJK VERSCHIL MET v1: in v1 genereerde de simulatie alleen een
# duur uit één "ware_mu" (covariaten deden niet mee, want ze leerden toch
# niet mee). Hier genereert de simulatie OOK realistische covariaat-waarden
# per gesimuleerd project (volgens de "sim"-specificatie per covariaat:
# binaire factoren zijn echt 0/1, schalen zijn gehele scores, GFA is ~N(100,15)),
# en de "ware" duur wordt berekend met de VOLLEDIGE lineaire predictor
# (ware intercept + ware coëfficiënten × gecentreerde covariaten + ruis).
# Zo kun je zien of het model niet alleen de baseline, maar ook elke
# individuele coëfficiënt terugvindt uit de data — inclusief wanneer de
# prior daarover bewust fout stond.

def _trek_covariaatwaarden(rng, sim_spec, n):
    """Trekt n realistische RUWE covariaatwaarden volgens de sim-specificatie.

    Ondersteunde vormen (zie ALGEMENE_SIM / FASE_INSTELLINGEN):
      ("normaal", gemiddelde, sd)
      ("uniform", laag, hoog)          — continu
      ("uniform_int", laag, hoog)      — gehele scores, grenzen inclusief
      ("keuze", [waarden], [kansen])   — kansen optioneel (default uniform)
      ("binair", kans_op_1)
    """
    soort = sim_spec[0]
    if soort == "normaal":
        return rng.normal(sim_spec[1], sim_spec[2], size=n)
    if soort == "uniform":
        return rng.uniform(sim_spec[1], sim_spec[2], size=n)
    if soort == "uniform_int":
        return rng.integers(sim_spec[1], sim_spec[2] + 1, size=n).astype(float)
    if soort == "keuze":
        kansen = sim_spec[2] if len(sim_spec) > 2 else None
        return rng.choice(np.asarray(sim_spec[1], dtype=float), size=n, p=kansen)
    if soort == "binair":
        return (rng.random(n) < sim_spec[1]).astype(float)
    raise ValueError(f"Onbekende sim-specificatie: {sim_spec!r}")


def simuleer_projecten(model, n, seed=42):
    """
    Genereert n gesimuleerde projecten (RUWE covariaten + duur) voor verificatie.

    In de echte toepassing vervangt EaSiCon-data deze functie. Hier zijn de
    ware waarden bekend (in tegenstelling tot de praktijk), zodat we kunnen
    controleren of het model naar de ware coëfficiënten convergeert.

    De ruis wordt getrokken met WARE_SIGMA — bewust ANDERS dan de prior
    SIGMA_LOG_PRIOR, zodat de sigma-convergentiegrafiek daadwerkelijk toont
    dat het model een foute ruis-prior corrigeert.

    Parameters
    ----------
    model : BayesiaansFaseModel — levert de covariaatnamen, referenties,
            sim-specificaties en "ware_waarde"s (uit FASE_INSTELLINGEN)
    n     : int — aantal te simuleren projecten
    seed  : int — voor reproduceerbaarheid

    Returns
    -------
    (X_ruw, durations) : ruwe covariatenmatrix (n, p) en durations (n,) in DAGEN
    """
    rng = np.random.default_rng(seed)

    # Realistische RUWE covariaatwaarden per kolom (vaste volgorde)
    X_ruw = np.column_stack([
        _trek_covariaatwaarden(rng, model.sim_specs[naam], n)
        for naam in model.covariate_namen
    ])

    # Ware coëfficiënten in dezelfde volgorde als model.covariate_namen
    ware_beta = np.array([
        ALGEMENE_WARE_WAARDE[naam] if naam in ALGEMENE_WARE_WAARDE
        else model.fase_config["fase_specifieke_factoren"][naam]["ware_waarde"]
        for naam in model.covariate_namen
    ])
    ware_mu = model.fase_config["ware_mu"]

    # Het "ware" data-genererende proces gebruikt dezelfde centrering als het
    # model: ware_mu geldt voor de referentiewoning.
    referentie = np.array([model.referenties[naam] for naam in model.covariate_namen])
    X_gecentreerd = X_ruw - referentie

    log_duur = ware_mu + X_gecentreerd @ ware_beta + rng.normal(0, WARE_SIGMA, size=n)

    return X_ruw, np.exp(log_duur)


def voer_convergentie_experiment_uit(model, n_projecten=100, seed=42):
    """
    Voert het convergentie-experiment uit voor één fase.

    Reset het model, simuleert n_projecten (mét covariaten), en slaat na
    elke update de schatting op. Hiermee toon je SQ5: convergeert het model
    naar de ware waarden als het (bewust) een foute prior heeft?

    Het model wordt aan het EIND weer teruggezet naar de prior, zodat de
    aanroeper een schoon model overhoudt; de geschiedenis wordt vóór die
    reset gekopieerd naar het resultaat.

    Parameters
    ----------
    model       : BayesiaansFaseModel
    n_projecten : int
    seed        : int

    Returns
    -------
    dict met sleutels 'mediaan', 'lower', 'upper', 'sigma', 'coefs',
    'ware_mediaan' (= exp(ware_mu): de ware mediaan-duur van de
    referentiewoning; vergelijk mediaan-met-mediaan, zie wijziging 4)
    """
    model.reset()
    X_ruw, durations = simuleer_projecten(model, n_projecten, seed=seed)

    for t in range(n_projecten):
        kenmerken = dict(zip(model.covariate_namen, X_ruw[t]))
        model.update(durations[t], kenmerken)

    resultaat = {
        "mediaan":      np.array(model.verwacht_geschiedenis),
        "lower":        np.array(model.lower_geschiedenis),
        "upper":        np.array(model.upper_geschiedenis),
        "sigma":        np.array(model.sigma_geschiedenis),
        "coefs":        np.array(model.coef_geschiedenis),   # (n+1, p+1)
        "ware_mediaan": float(np.exp(model.fase_config["ware_mu"])),
    }
    model.reset()
    return resultaat


# ==============================================================================
# SECTIE 4: GRAFIEKEN
# De drie grafiekfuncties DELEN dezelfde simulatieresultaten (dict
# `resultaten`, één keer berekend in main()) in plaats van elk opnieuw
# dezelfde 100 projecten te simuleren.
# ==============================================================================

def _teken_milestones(ax, milestones, n_projecten):
    """Verticale n=... markeringen; y-positie in as-coördinaten zodat de
    labels netjes onderin blijven, óók nadat set_ylim is aangeroepen."""
    for m in milestones:
        if m <= n_projecten:
            ax.axvline(m, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(m + 0.5, 0.02, f"n={m}", fontsize=6, color="#888888",
                    va="bottom", transform=ax.get_xaxis_transform())


def maak_convergentiegrafiek(modellen, resultaten, n_projecten=100,
                             bestandsnaam="easicon_v2_resultaten.png"):
    """
    Convergentiegrafiek: 2x3 subplots, één per fase (100 projecten).

    Per subplot:
      - Gekleurde lijn    : mediaan-duur (referentiewoning) na elk project
      - Gekleurd vlak     : 90% predictie-interval
      - Zwart gestippeld  : de 'ware' mediaan (target voor convergentie)
      - Grijs gestippeld  : initiële prior-schatting
      - Tekstvak rechts   : covariaten van de fase met prior-beta-waarden
    """
    fig, assen = plt.subplots(2, 3, figsize=(16, 10))
    assen = assen.flatten()

    fig.suptitle(
        "EaSiCon v2 — Bayesiaans Duratie-Schattingsmodel (volledige regressie)\n"
        f"Convergentie van mediaan-duur per bouwfase (SQ5-simulatie, {n_projecten} projecten)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    milestones = [10, 25, 50]

    for i, (fase_code, model) in enumerate(modellen.items()):
        ax     = assen[i]
        config = FASE_INSTELLINGEN[fase_code]

        resultaat = resultaten[fase_code]

        x            = np.arange(n_projecten + 1)
        mediaan      = resultaat["mediaan"]
        lower        = resultaat["lower"]
        upper        = resultaat["upper"]
        ware_mediaan = resultaat["ware_mediaan"]
        prior_duur   = mediaan[0]
        kleur        = config["kleur"]

        ax.fill_between(x, lower, upper, alpha=0.20, color=kleur, label="90% interval")
        ax.plot(x, mediaan, color=kleur, linewidth=2.5, label="Modelschatting (mediaan)")
        ax.axhline(ware_mediaan, color="black", linestyle="--",
                   linewidth=1.5, label=f"Ware mediaan ({ware_mediaan:.0f}d)")
        ax.axhline(prior_duur, color="#888888", linestyle=":",
                   linewidth=1.2, alpha=0.8, label=f"Prior ({prior_duur:.0f}d)")

        factoren = config.get("fase_specifieke_factoren", {})
        if VLAKKE_PRIOR:
            cov_regels = ["VLAKKE PRIOR: alle β=0", "(literatuurwaarden genegeerd)",
                         "─────────────────"]
            cov_regels += [f"{naam_f:<16}" for naam_f in factoren]
        else:
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
        # y-limiet: sla de eerste paar punten over — bij een (vrijwel) vlakke
        # prior is het prior-interval extreem breed en zou het de rest van de
        # curve platdrukken.
        kijk_vanaf = min(5, len(upper) - 1)
        ax.set_ylim(0, max(upper[kijk_vanaf:].max(), ware_mediaan) * 1.18)
        _teken_milestones(ax, milestones, n_projecten)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafiek opgeslagen: {bestandsnaam}")


def maak_onzekerheidsgrafiek(modellen, resultaten, n_projecten=100,
                             bestandsnaam="easicon_v2_zekerheid.png"):
    """
    Toont hoe de zekerheid per fase toeneemt na 0-100 projecten.

    VERSCHIL MET v1: v1 toonde tau_n (onzekerheid op de intercept) naast een
    VASTE sigma-lijn. Hier is sigma zelf ook een schatting die convergeert
    (linkerpaneel), naast de intervalsbreedte (rechterpaneel) die nu behalve
    door minder intercept-onzekerheid ook door minder coëfficiënt-onzekerheid
    daalt. De sigma-lijnen starten op de prior (0.35) en moeten naar de ware
    ruis (WARE_SIGMA = 0.45) toe bewegen — een echte correctietest.
    """
    fig, (ax_sigma, ax_breedte) = plt.subplots(1, 2, figsize=(15, 6))

    fig.suptitle(
        f"EaSiCon v2 — Zekerheidstoename per bouwfase ({n_projecten} projecten)\n"
        "Links = geleerde sigma convergeert van foute prior naar ware ruis  |  "
        "Rechts = 90%-intervalsbreedte daalt (intercept + covariaten samen)",
        fontsize=12, fontweight="bold",
    )

    milestones = [10, 25, 50]
    max_breedte = 0.0
    max_sigma = WARE_SIGMA

    for fase_code, model in modellen.items():
        config = FASE_INSTELLINGEN[fase_code]
        kleur  = config["kleur"]
        label  = f"{fase_code}: {config['naam']}"

        resultaat = resultaten[fase_code]

        x       = np.arange(n_projecten + 1)
        sigma   = resultaat["sigma"]
        breedte = resultaat["upper"] - resultaat["lower"]

        ax_sigma.plot(x, sigma, color=kleur, linewidth=2.2, label=label)
        ax_sigma.annotate(f"{sigma[-1]:.3f}", xy=(n_projecten, sigma[-1]),
                          fontsize=6.5, color=kleur,
                          xytext=(3, 0), textcoords="offset points", va="center")

        ax_breedte.plot(x, breedte, color=kleur, linewidth=2.2, label=label)

        # Voor de y-limiet van het breedte-paneel: eerste paar punten overslaan
        # (extreem breed prior-interval bij vlakke prior drukt de rest plat).
        kijk_vanaf = min(5, len(breedte) - 1)
        max_breedte = max(max_breedte, breedte[kijk_vanaf:].max())
        max_sigma = max(max_sigma, sigma.max())

    ax_sigma.axhline(WARE_SIGMA, color="black", linestyle="--", linewidth=1.2)
    ax_sigma.text(1, WARE_SIGMA + 0.005,
                  f"ware sigma (simulatie) = {WARE_SIGMA}",
                  fontsize=7, color="#444444", va="bottom")
    ax_sigma.axhline(SIGMA_LOG_PRIOR, color="#CCCCCC", linestyle=":", linewidth=1.0)
    ax_sigma.text(1, SIGMA_LOG_PRIOR - 0.005,
                  f"prior-start = {SIGMA_LOG_PRIOR}",
                  fontsize=7, color="#999999", va="top")

    for ax in (ax_sigma, ax_breedte):
        _teken_milestones(ax, milestones, n_projecten)
        ax.set_xlabel("Voltooide projecten", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0, n_projecten)

    ax_sigma.set_ylim(0, max_sigma * 1.15)
    ax_breedte.set_ylim(0, max_breedte * 1.15)

    ax_sigma.set_title(
        "Geleerde sigma (ruis op log-schaal) per fase\n"
        f"start = prior {SIGMA_LOG_PRIOR} (bewust fout), doel = ware ruis {WARE_SIGMA}",
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


def maak_coefficientgrafiek(modellen, resultaten, n_projecten=100,
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

        resultaat = resultaten[fase_code]
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

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafiek opgeslagen: {bestandsnaam}")


# ==============================================================================
# SECTIE 5: TEKSTOUTPUT
# ==============================================================================

def print_schattingstabel(modellen):
    """Print overzichtstabel met huidige schattingen + covariaten per fase.
    Alle schattingen gelden voor de REFERENTIEWONING (covariaten op hun
    referentiewaarde)."""
    breedte = 88
    print("\n" + "=" * breedte)
    print("  EaSiCon v2 — Huidige faseschattingen  (prior, vóór projectdata)")
    print("  Schattingen gelden voor de referentiewoning "
          "(100 m², 1 laag, 6 m, geen schade)")
    print("=" * breedte)
    print(f"  {'Fase':<5} {'Naam':<36} {'Mediaan':>9} {'90%-interval':>20}")
    print("-" * breedte)

    for fase_code, model in modellen.items():
        config = FASE_INSTELLINGEN[fase_code]
        naam   = config["naam"]
        v, l, u = model.voorspel()
        print(f"  {fase_code:<5} {naam:<36} {v:>7.1f}d  [{l:>5.1f} - {u:>6.1f}d]")
        print(f"  {'':>5}  sigma (ruis, leert mee) = {model.sigma_estimate():.3f}")

        if VLAKKE_PRIOR:
            print(f"  {'':>5}  VLAKKE PRIOR actief: alle β=0, literatuurwaarden "
                  f"hieronder zijn GENEGEERD.")
            factoren = config.get("fase_specifieke_factoren", {})
            print(f"  {'':>5}  Covariaten van deze fase: "
                  f"GFA, verdiepingen, hoogte, {', '.join(factoren)}")
        else:
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
    if VLAKKE_PRIOR:
        print("  * VLAKKE_PRIOR = True: het model start met vrijwel 0 informatie.")
        print("    De mediaan van 1.0d hierboven is exp(0) — een bewust informatie-")
        print("    loos startpunt, GEEN inhoudelijke schatting. De literatuurwaarden")
        print("    in FASE_INSTELLINGEN worden genegeerd; alleen expert-scenario's")
        print("    (elicitatie_verwerker.py) of echte projectdata bepalen waar de")
        print("    coëfficiënten landen. Zet VLAKKE_PRIOR=False om de literatuur-")
        print("    waarden weer als startpunt te gebruiken.")
    else:
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
                'Stel per fase: "Hoe lang duurt fase X voor de REFERENTIEWONING?\n'
                "         Geef een optimistische, meest waarschijnlijke en pessimistische\n"
                '         schatting in dagen." Zet mu_0 = log(meest_waarschijnlijk) en\n'
                "         breedte_0 = (opt, pess) in FASE_INSTELLINGEN. Leg vast op welke\n"
                "         referentiewoning de schatting slaat (zie ALGEMENE_REFERENTIE)."
            ),
            "locatie": "FASE_INSTELLINGEN -> mu_0, breedte_0 per fase; ALGEMENE_REFERENTIE",
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
                "         interviews. Eenheden én referentiewaarde vastleggen en documenteren."
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
                "         (c) Sigma: convergeert de geleerde ruis (start 0.35) naar de\n"
                "             ware ruis (WARE_SIGMA = 0.45)?\n"
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
                "         waarbij project_kenmerken een dict is met RUWE waarden en dezelfde\n"
                "         covariate-namen als in FASE_INSTELLINGEN (bv. {'GFA': 95,\n"
                "         'schadegraad': 6}); centrering rond de referentiewoning gebeurt\n"
                "         automatisch in BayesiaansFaseModel._naar_vector()."
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
        code: BayesiaansFaseModel(code, cfg)
        for code, cfg in FASE_INSTELLINGEN.items()
    }

    # 1. Tabel met huidige schattingen + covariaten (prior-stand)
    print_schattingstabel(modellen)

    # 2. Openstaande TODO's
    print_todo_lijst()

    # 3. Simulaties: ÉÉN run per fase, gedeeld door alle drie de grafieken
    print("  Simulaties draaien (1x per fase, gedeeld door alle grafieken)...")
    resultaten = {
        code: voer_convergentie_experiment_uit(model, N_PROJECTEN, seed=42 + i)
        for i, (code, model) in enumerate(modellen.items())
    }

    # 4. Convergentiegrafiek: mediaan-duur per fase (100 projecten)
    maak_convergentiegrafiek(modellen, resultaten, n_projecten=N_PROJECTEN)

    # 5. Zekerheidstoename-grafiek (sigma + intervalsbreedte over 100 projecten)
    maak_onzekerheidsgrafiek(modellen, resultaten, n_projecten=N_PROJECTEN)

    # 6. Convergentie van elke coëfficiënt afzonderlijk
    maak_coefficientgrafiek(modellen, resultaten, n_projecten=N_PROJECTEN)

    print("\nKlaar.")
    print("  easicon_v2_resultaten.png     — convergentie mediaan-duur per fase")
    print("  easicon_v2_zekerheid.png     — sigma-convergentie + intervalsbreedte")
    print("  easicon_v2_coefficienten.png — convergentie per covariate-coëfficiënt")
    print()


if __name__ == "__main__":
    main()
