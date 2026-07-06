"""
EaSiCon — Verwerking van expert-elicitatie scenario's (Sectie 4.4.2 van de thesis)
====================================================================================
Auteur: Gijs Meijers (4957822) | TU Delft x Haskoning | 2026

WAT DIT SCRIPT DOET
--------------------
Tijdens de expert-interviews (elicitatie stap 2, Sectie 4.4.2) krijgt elke
respondent een concreet, volledig ingevuld scenario voorgelegd: een woning met
specifieke waarden voor alle relevante factoren (bv. GFA=95, schadegraad=6,
asbest=1, geveltype=2, ...). De respondent geeft daarbij drie schattingen in
werkdagen: optimistisch, meest waarschijnlijk, pessimistisch, voor de duur van
één fase.

Dit script leest die antwoorden in vanuit een CSV-bestand (één rij per
scenario-antwoord van één respondent) en verwerkt ze als "nep-projecten": elk
scenario wordt op EXACT dezelfde manier door het Bayesiaanse regressiemodel
(PhaseDurationModel.update(), zie estimation_module.py) gehaald als een echt
voltooid EaSiCon-project. Dit is dezelfde sequentiële Bayesiaanse update die
Sectie 5.4 van de thesis beschrijft (Vergelijking 5.3): elk scenario draagt
bewijs bij over de coëfficiënten van de fase, en de posterior na alle
scenario's is precies de prior waarmee het model straks echte projectdata
gaat verwerken.

WAAROM GEEN LOSSE REGRESSIE-STAP
----------------------------------
Een alternatief zou zijn om eerst een klassieke (OLS) regressie te draaien
over alle scenario-antwoorden, en pas daarna de uitkomst als prior in te
vullen. Dat is een extra, aparte berekening die losstaat van het Bayesiaanse
model. Door de scenario's direct als observaties door `update()` te sturen,
gebruik je dezelfde wiskunde die het model toch al gebruikt voor echte
projecten — er is dus geen aparte "vertaalstap" nodig, en de scenario's en
latere echte projectdata zijn uitwisselbaar als input.

HOE ÉÉN SCENARIO WORDT OMGEZET IN EEN OBSERVATIE
--------------------------------------------------
Voor een scenario met covariaten x en antwoorden (opt, ml, pess):
  1. threepoint_to_lognormal(opt, ml, pess) uit estimation_module.py zet dit
     om in (mu, sigma) op log-schaal: mu is de "beste schatting" van
     log(duur) voor DIT specifieke scenario, sigma is hoe breed de
     drie-punts-schatting is (breder = onzekerder antwoord).
  2. mu wordt behandeld als een geobserveerde log(duur): duur = exp(mu).
  3. Deze (x, duur) wordt via model.update() aan de posterior toegevoegd,
     precies zoals BayesianesFaseModel.update() dat doet met echte projecten.

Scenario's met een smallere (opt, pess)-marge wegen zwaarder mee in de
uiteindelijke coëfficiënt-schatting, omdat een kleinere sigma in de
NIG-wiskunde automatisch meer gewicht geeft aan die observatie — dit gebeurt
vanzelf via de bestaande update()-formules, er is geen aparte weegfactor
nodig.

MEERDERE RESPONDENTEN
-----------------------
Omdat elk scenario gewoon een observatie is, worden meerdere respondenten
automatisch gecombineerd: elke respondent levert zijn eigen scenario's aan,
en die worden allemaal na elkaar verwerkt door dezelfde sequentiële update.
Antwoorden van verschillende experts over vergelijkbare scenario's wegen mee
naar rato van hun eigen (opt, pess)-breedte — een expert die zeer stellig is
("ik weet zeker dat dit tussen de 20 en 24 dagen duurt") weegt zwaarder mee
dan een expert die twijfelt ("ergens tussen 10 en 40 dagen").

CSV-FORMAAT
------------
Kolommen (vaste namen, hoofdlettergevoelig):
    fase                 : fasecode, bv. "A", "B1", "C1"
    respondent           : naam/ID van de expert (voor traceerbaarheid)
    optimistisch         : optimistische schatting in werkdagen
    meest_waarschijnlijk : meest waarschijnlijke schatting in werkdagen
    pessimistisch        : pessimistische schatting in werkdagen
    <factornaam>...      : één kolom per covariate-naam uit FASE_INSTELLINGEN
                           (bv. GFA, verdiepingen, hoogte, schadegraad, asbest, ...)
                           Waarden zijn RUWE waarden (GFA in m², schadegraad 0-10);
                           centrering rond de referentiewoning gebeurt automatisch
                           in BayesiaansFaseModel. Lege cellen betekenen "zoals de
                           referentiewoning" (= de referentiewaarde).

Zie `voorbeeld_elicitatie.csv` voor een compleet voorbeeld.

GEBRUIK
--------
    python elicitatie_verwerker.py voorbeeld_elicitatie.csv

Output:
  - Console: per fase de bijgewerkte coëfficiënten (mean + sd) en sigma,
    vergeleken met de literatuur-prior waarmee begonnen is.
  - elicitatie_geupdatete_instellingen.py: een kant-en-klaar Python-blok dat
    je kunt kopiëren/plakken in FASE_INSTELLINGEN in easicon_model_v2.py.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from estimation_module import threepoint_to_lognormal
from easicon_model_v2 import (
    FASE_INSTELLINGEN, ALGEMENE_COVARIATEN, ALGEMENE_PRIOR_MEAN,
    ALGEMENE_PRIOR_SD, SIGMA_LOG_PRIOR, SIGMA_CONFIDENCE, VLAKKE_PRIOR,
    BayesiaansFaseModel,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


VASTE_KOLOMMEN = {"fase", "respondent", "optimistisch",
                  "meest_waarschijnlijk", "pessimistisch"}


def lees_scenarios(pad_csv: str) -> dict[str, list[dict]]:
    """
    Leest het CSV-bestand in en groepeert de scenario-rijen per fase.

    Elke rij wordt een dict met:
        "respondent"  : str
        "opt"         : float
        "ml"          : float
        "pess"        : float
        "kenmerken"   : dict {factornaam: ruwe waarde}; ontbrekende/lege cellen
                        worden WEGGELATEN, zodat het model er de referentie-
                        waarde voor invult (= "zoals de referentiewoning")

    Returns
    -------
    dict {fase_code: [scenario_dict, ...]}
    """
    scenarios_per_fase: dict[str, list[dict]] = defaultdict(list)

    with open(pad_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Kon geen kolommen vinden in {pad_csv}")

        factor_kolommen = [k for k in reader.fieldnames if k not in VASTE_KOLOMMEN]

        for rij_nr, rij in enumerate(reader, start=2):  # rij 1 = header
            fase = rij["fase"].strip()
            if fase not in FASE_INSTELLINGEN:
                raise ValueError(
                    f"Rij {rij_nr}: onbekende fasecode '{fase}'. "
                    f"Verwacht één van {list(FASE_INSTELLINGEN.keys())}."
                )

            try:
                opt  = float(rij["optimistisch"])
                ml   = float(rij["meest_waarschijnlijk"])
                pess = float(rij["pessimistisch"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Rij {rij_nr} (fase {fase}): kon opt/ml/pess niet lezen "
                    f"als getal ({exc})."
                )

            kenmerken = {}
            for kol in factor_kolommen:
                waarde = (rij.get(kol) or "").strip()
                if waarde:
                    kenmerken[kol] = float(waarde)

            scenarios_per_fase[fase].append({
                "respondent": rij.get("respondent", "").strip(),
                "opt": opt, "ml": ml, "pess": pess,
                "kenmerken": kenmerken,
            })

    return scenarios_per_fase


def bouw_fase_model(fase_code: str) -> BayesiaansFaseModel:
    """
    Bouwt het startmodel voor één fase door BayesiaansFaseModel te hergebruiken
    (uit easicon_model_v2.py), zodat deze verwerker ALTIJD hetzelfde startpunt
    gebruikt als het hoofdmodel — inclusief de VLAKKE_PRIOR-schakelaar én de
    centrering rond de referentiewoning. Staat VLAKKE_PRIOR op True, dan
    starten de scenario's dus ook hier vanaf m=0 met brede onzekerheid, in
    plaats van vanaf de literatuurwaarden.
    Dit is het model dat de scenario-observaties gaat "absorberen".
    """
    return BayesiaansFaseModel(fase_code, FASE_INSTELLINGEN[fase_code])


def verwerk_scenario(model: BayesiaansFaseModel, scenario: dict) -> None:
    """
    Zet één scenario-antwoord om in een (kenmerken, duur)-observatie en voedt
    die aan model.update(), exact zoals een voltooid project. De ruwe
    kenmerken worden daarbij automatisch gecentreerd rond de referentiewoning
    (via BayesiaansFaseModel._naar_vector), net als bij echte projectdata.

    De drie-punts-schatting van dit ÉÉN scenario wordt eerst vertaald naar
    (mu, sigma) op log-schaal via dezelfde threepoint_to_lognormal() die ook
    de fase-brede baseline_threepoint vertaalt. mu = log(meest_waarschijnlijk),
    dus exp(mu) — de "geobserveerde duur" die het model te zien krijgt — is
    precies het meest-waarschijnlijke antwoord van de expert.
    """
    mu, _sigma_scenario = threepoint_to_lognormal(
        scenario["opt"], scenario["ml"], scenario["pess"]
    )
    duur_scenario = float(np.exp(mu))

    model.update(duur_scenario, scenario["kenmerken"])


def verwerk_alle_scenarios(pad_csv: str) -> dict[str, BayesiaansFaseModel]:
    """
    Leest het CSV-bestand en verwerkt, per fase, alle scenario's na elkaar
    door hetzelfde model. Retourneert het bijgewerkte model per fase.
    """
    scenarios_per_fase = lees_scenarios(pad_csv)
    bijgewerkte_modellen = {}

    for fase_code in FASE_INSTELLINGEN:
        model = bouw_fase_model(fase_code)
        for scenario in scenarios_per_fase.get(fase_code, []):
            verwerk_scenario(model, scenario)
        bijgewerkte_modellen[fase_code] = model

        n = len(scenarios_per_fase.get(fase_code, []))
        if n == 0:
            print(f"  Fase {fase_code}: geen scenario's gevonden in CSV, "
                  f"prior blijft ongewijzigd.")

    return bijgewerkte_modellen


def print_vergelijking(bijgewerkte_modellen: dict[str, BayesiaansFaseModel]) -> None:
    """Print per fase de prior naast de door scenario's bijgewerkte schatting,
    zodat je kunt zien hoeveel de interviews de prior hebben verschoven.
    Bij VLAKKE_PRIOR=True is de prior per definitie 0 (vlak) voor elke
    coëfficiënt; de literatuurwaarden in FASE_INSTELLINGEN worden dan niet
    als prior gebruikt en dus ook niet als zodanig getoond."""
    breedte = 92
    print("\n" + "=" * breedte)
    if VLAKKE_PRIOR:
        print("  Vergelijking: VLAKKE prior (alles op 0)  vs.  na verwerking expert-scenario's")
    else:
        print("  Vergelijking: literatuur-prior  vs.  na verwerking expert-scenario's")
    print("=" * breedte)

    for fase_code, model in bijgewerkte_modellen.items():
        config = FASE_INSTELLINGEN[fase_code]
        print(f"\n  Fase {fase_code}: {config['naam']}  "
              f"(n_scenario's verwerkt: {model.n_projecten})")

        samenvatting = model.coefficient_summary()
        namen = ["intercept"] + model.covariate_namen
        for naam in namen:
            if naam == "intercept":
                prior_waarde = 1.0 if VLAKKE_PRIOR else np.exp(config["mu_0"])
                nieuw = samenvatting[naam]
                print(f"    {'intercept (dagen)':<24} "
                      f"prior={prior_waarde:>7.1f}d   "
                      f"na scenario's={np.exp(nieuw['mean']):>7.1f}d "
                      f"(sd log-schaal={nieuw['sd']:.3f})")
            else:
                if VLAKKE_PRIOR:
                    prior_mean = 0.0
                elif naam in ALGEMENE_PRIOR_MEAN:
                    prior_mean = ALGEMENE_PRIOR_MEAN[naam]
                else:
                    prior_mean = config["fase_specifieke_factoren"][naam]["mean"]
                nieuw = samenvatting[naam]
                print(f"    {naam:<24} prior β={prior_mean:>+7.4f}   "
                      f"na scenario's β={nieuw['mean']:>+7.4f} "
                      f"(sd={nieuw['sd']:.4f})")

        print(f"    {'sigma (ruis)':<24} "
              f"prior={SIGMA_LOG_PRIOR:>7.3f}      "
              f"na scenario's={model.sigma_estimate():>7.3f}")

    print("\n" + "=" * breedte)


def schrijf_python_blok(bijgewerkte_modellen: dict[str, BayesiaansFaseModel],
                        bestandsnaam: str = "elicitatie_geupdatete_instellingen.py") -> None:
    """
    Schrijft een los Python-bestand met de bijgewerkte mean/sd-waarden per
    fase, in exact hetzelfde dict-formaat als FASE_INSTELLINGEN in
    easicon_model_v2.py. Dit is GEEN vervanging van FASE_INSTELLINGEN: het is
    een kant-en-klaar overzicht dat je zelf, met een blik erop, kunt
    overnemen in dat bestand (mean/sd per factor, en mu_0 per fase).
    """
    regels = [
        '"""',
        "Automatisch gegenereerd door elicitatie_verwerker.py.",
        "Bevat de coëfficiënten NA verwerking van de expert-scenario's uit de",
        "elicitatie-CSV. Kopieer de gewenste waarden over naar FASE_INSTELLINGEN",
        "in easicon_model_v2.py (mean/sd per factor, mu_0 per fase).",
        '"""',
        "",
        "import numpy as np",
        "",
        "GEUPDATETE_FASE_WAARDEN = {",
    ]

    for fase_code, model in bijgewerkte_modellen.items():
        samenvatting = model.coefficient_summary()
        regels.append(f'    "{fase_code}": {{')
        regels.append(f'        "n_scenarios": {model.n_projecten},')
        regels.append(f'        "mu_0_nieuw": np.log({np.exp(samenvatting["intercept"]["mean"]):.4f}),  '
                      f'# was {np.exp(FASE_INSTELLINGEN[fase_code]["mu_0"]):.1f}d')
        regels.append(f'        "sigma_nieuw": {model.sigma_estimate():.4f},  '
                      f'# was {SIGMA_LOG_PRIOR}')
        regels.append('        "factoren": {')
        for naam in model.covariate_namen:
            s = samenvatting[naam]
            regels.append(f'            "{naam}": {{"mean": {s["mean"]:.5f}, '
                          f'"sd": {s["sd"]:.5f}}},')
        regels.append("        },")
        regels.append("    },")

    regels.append("}")

    with open(bestandsnaam, "w", encoding="utf-8") as f:
        f.write("\n".join(regels) + "\n")

    print(f"\n  Python-blok met bijgewerkte waarden geschreven naar: {bestandsnaam}")


def main():
    if len(sys.argv) != 2:
        print("Gebruik: python elicitatie_verwerker.py <pad_naar_csv>")
        sys.exit(1)

    pad_csv = sys.argv[1]
    print(f"\nEaSiCon — Verwerking expert-elicitatie scenario's uit: {pad_csv}\n")

    bijgewerkte_modellen = verwerk_alle_scenarios(pad_csv)
    print_vergelijking(bijgewerkte_modellen)
    schrijf_python_blok(bijgewerkte_modellen)

    print("\nKlaar. Controleer elicitatie_geupdatete_instellingen.py en werk")
    print("FASE_INSTELLINGEN in easicon_model_v2.py bij met de waarden die je")
    print("na review wilt overnemen.\n")


if __name__ == "__main__":
    main()
