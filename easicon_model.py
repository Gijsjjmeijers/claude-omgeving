"""
EaSiCon Bayesiaans Duratie-Schattingsmodel
==========================================
Auteur  : Gijs Meijers (4957822) | TU Delft x Haskoning | 2026
Thesis  : Self-learning Construction Phase Estimation for a Digital Twin

Dit script implementeert een Bayesiaans lognormaal regressiemodel voor het
schatten van bouwfase-durations in het EaSiCon-project (aardbeving-renovaties
in Eemsdelta, uitgevoerd door VolkerWessels).

Theorie (zie thesis H6, Kim & Reinschmidt 2009, Gelman et al. 2013):
  - log(duur) ~ Normaal(mu, sigma²)       [lognormaal voor positieve durations]
  - Prior op mu: mu ~ Normaal(mu_0, tau_0²)
  - Update: Normal-Normal conjugate posterior

Gebruik : python easicon_model.py
Output  :
  1. Tabel — huidige schatting per fase in dagen, incl. covariaten
  2. TODO-lijst — openstaande punten na interviews
  3. easicon_resultaten.png  — convergentiegrafiek per fase (100 projecten)
  4. easicon_zekerheid.png   — zekerheidstoename per fase (tau + intervalsbreedte)

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

# Zorg dat de Windows-console UTF-8 uitvoert (box-tekens, β, enz.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# SECTIE 1: INSTELLINGEN
# Alle prior-waarden en coëfficiënten staan hier centraal.
# Na expert-interviews vervang je ALLEEN de waarden in dit blok.
# ==============================================================================

# --- Observatieruis ---
# SIGMA_LOG = standaardafwijking van log(duur) tussen projecten van hetzelfde type.
# 0.4 betekent dat ~68% van de durations binnen een factor exp(0.4) ≈ 1.5 van
# het gemiddelde valt — redelijk voor renovatieprojecten.
SIGMA_LOG = 0.4  # !! TODO: valideer na interviews / eerste EaSiCon-data

# --- Algemene covariate-coëfficiënten (alle fases) ---
# Effect op log(duur) per eenheid van de covariate.
# Bronnen literatuur: Lawal et al. 2023, Kim et al. 2019, Chan & Kumaraswamy 1999
# Let op: onderstaande zijn placeholders; eenheden en orde-van-grootte moeten
#          worden gevalideerd voor Eemsdelta-woningen (GFA ~80–150 m², 1–2 lagen).
BETA_GFA          = 0.002   # !! TODO: coëff. bruto vloeroppervlak (per m²)
BETA_VERDIEPINGEN = 0.08    # !! TODO: coëff. aantal verdiepingen
BETA_HOOGTE       = 0.015   # !! TODO: coëff. gebouwhoogte (per meter)

# --- Fase-instellingen ---
# Per fase:
#   mu_0    : startschatting op log-schaal = log(verwachte dagen), prior-gemiddelde
#   tau_0   : onzekerheid op mu_0 (std van de prior); 0.5 = grote onzekerheid
#   ware_mu : voor de simulatie/verificatiegrafiek (stel bewust iets anders
#             dan mu_0 om te testen of model een foute prior kan corrigeren)
#   kleur   : voor de grafiek
#   fase_specifieke_factoren : 3 factoren per fase (coëfficiënt 0.0 = placeholder)

FASE_INSTELLINGEN = {
    "A": {
        "naam":    "Voorbereiding",
        "mu_0":    np.log(15),    # !! TODO: vervang 15 met uitkomst expert-interview (dagen)
        "tau_0":   0.5,           # !! TODO: pas aan na interviews (drie-punts-methode)
        "ware_mu": np.log(18),    # simulatie: model moet van ~15 naar ~18 dagen corrigeren
        "kleur":   "#2196F3",
        "fase_specifieke_factoren": {
            # !! TODO: stel na interviews in welke 3 factoren duur het meest bepalen
            "schadegraad":    0.0,  # !! TODO: effect schadegraad (bijv. 0–10 schaal)
            "asbest":         0.0,  # !! TODO: effect aanwezigheid asbest (0/1)
            "bereikbaarheid": 0.0,  # !! TODO: effect bereikbaarheidsscore
        },
    },
    "B1": {
        "naam":    "Constructief: Fundering & Onderbouw",
        "mu_0":    np.log(25),    # !! TODO: vervang 25 met uitkomst expert-interview
        "tau_0":   0.5,
        "ware_mu": np.log(30),    # simulatie: onderschatting, model corrigeert omhoog
        "kleur":   "#F44336",
        "fase_specifieke_factoren": {
            "funderingsschade": 0.0,  # !! TODO: effect funderingsschade (0–10)
            "bodemtype":        0.0,  # !! TODO: effect bodemtype (categorisch)
            "onderbemaling":    0.0,  # !! TODO: effect onderbemaling noodzakelijk (0/1)
        },
    },
    "B2": {
        "naam":    "Constructief: Skelet & Casco",
        "mu_0":    np.log(35),    # !! TODO: vervang 35 met uitkomst expert-interview
        "tau_0":   0.5,
        "ware_mu": np.log(32),    # simulatie: lichte overschatting, model corrigeert
        "kleur":   "#FF9800",
        "fase_specifieke_factoren": {
            "casoschade":        0.0,  # !! TODO: effect casoschade (0–10)
            "bouwmethode":       0.0,  # !! TODO: effect bouwmethode (categorisch)
            "structuuringrepen": 0.0,  # !! TODO: effect aantal structuuringrepen
        },
    },
    "C1": {
        "naam":    "Afbouw: Gevel & Dak",
        "mu_0":    np.log(30),    # !! TODO: vervang 30 met uitkomst expert-interview
        "tau_0":   0.5,
        "ware_mu": np.log(28),    # simulatie: lichte overschatting
        "kleur":   "#4CAF50",
        "fase_specifieke_factoren": {
            "geveltype":  0.0,   # !! TODO: effect geveltype (categorisch)
            "daktype":    0.0,   # !! TODO: effect daktype (categorisch)
            "schade_m2":  0.0,   # !! TODO: effect schadeoppervlak gevel (m²)
        },
    },
    "C2": {
        "naam":    "Afbouw: Technische Installaties",
        "mu_0":    np.log(25),    # !! TODO: vervang 25 met uitkomst expert-interview
        "tau_0":   0.5,
        "ware_mu": np.log(27),    # simulatie: lichte onderschatting
        "kleur":   "#9C27B0",
        "fase_specifieke_factoren": {
            "elektra_pct":         0.0,  # !! TODO: % elektrische installaties nieuw
            "verwarmingssysteem":  0.0,  # !! TODO: effect type verwarming (categorisch)
            "n_onderaannemers":    0.0,  # !! TODO: effect aantal onderaannemers
        },
    },
    "C3": {
        "naam":    "Afbouw: Functionele Afwerking",
        "mu_0":    np.log(20),    # !! TODO: vervang 20 met uitkomst expert-interview
        "tau_0":   0.5,
        "ware_mu": np.log(22),    # simulatie: lichte onderschatting
        "kleur":   "#795548",
        "fase_specifieke_factoren": {
            "afwerkingsniveau": 0.0,  # !! TODO: effect afwerkingsniveau (1–5 schaal)
            "schade_pct":       0.0,  # !! TODO: % schade aan afwerking
            "n_wijzigingen":    0.0,  # !! TODO: aantal scopewijzigingen
        },
    },
}


# ==============================================================================
# SECTIE 2: HET MODEL
# ==============================================================================

class BayesianesFaseModel:
    """
    Bayesiaans lognormaal model voor één bouwfase.

    Het model houdt bij:
    - mu_n  : huidig geschat gemiddelde van log(duur)   [posterior mean]
    - tau_n : huidige onzekerheid over mu_n             [posterior std]

    Na elk voltooid project worden mu_n en tau_n bijgewerkt via de
    Normal-Normal conjugate update (zie thesis H6, vergelijkingen 6.5–6.6).
    """

    def __init__(self, naam, mu_0, tau_0, sigma, fase_config=None):
        """
        Parameters
        ----------
        naam       : str   — naam van de fase
        mu_0       : float — prior gemiddelde van log(duur)
        tau_0      : float — prior onzekerheid op mu (std)
        sigma      : float — observatieruis op log-schaal (spreiding tussen projecten)
        fase_config: dict  — volledige fase-config uit FASE_INSTELLINGEN
        """
        self.naam = naam
        self.mu_0 = mu_0
        self.tau_0 = tau_0
        self.sigma = sigma
        self.fase_config = fase_config or {}

        # Huidige posterior; worden bijgewerkt via update()
        self.mu_n = mu_0
        self.tau_n = tau_0
        self.n_projecten = 0

        # Geschiedenis voor convergentieplot (index 0 = prior)
        self.mu_geschiedenis  = [mu_0]
        self.tau_geschiedenis = [tau_0]

    # ------------------------------------------------------------------
    # Interne hulpfunctie: covariate-effect op log(duur)
    # ------------------------------------------------------------------

    def _covariate_effect(self, project_kenmerken=None):
        """
        Berekent het additieve effect van projectkenmerken op log(duur).

        Parameters
        ----------
        project_kenmerken : dict of None
            Mogelijke sleutels: 'GFA', 'verdiepingen', 'hoogte'
            plus fase-specifieke factoren uit FASE_INSTELLINGEN.

        Returns
        -------
        float — bijdrage aan log(verwachte duur)
        """
        if project_kenmerken is None:
            return 0.0

        effect = 0.0

        # Algemene covariaten (gelden voor alle fases)
        effect += BETA_GFA          * project_kenmerken.get("GFA", 0.0)
        effect += BETA_VERDIEPINGEN * project_kenmerken.get("verdiepingen", 0.0)
        effect += BETA_HOOGTE       * project_kenmerken.get("hoogte", 0.0)

        # Fase-specifieke covariaten (coëfficiënten zijn nu 0.0 — zie TODO's)
        # !! TODO: na interviews coëfficiënten invullen in FASE_INSTELLINGEN
        fase_factoren = self.fase_config.get("fase_specifieke_factoren", {})
        for factor, coeff in fase_factoren.items():
            effect += coeff * project_kenmerken.get(factor, 0.0)

        return effect

    # ------------------------------------------------------------------
    # Publieke methodes
    # ------------------------------------------------------------------

    def voorspel(self, project_kenmerken=None):
        """
        Geeft de huidige schatting: verwachte duur + 90% predictie-interval.

        De predictieve verdeling voor een nieuw project is:
          log(duur_nieuw) ~ Normaal(mu_n, tau_n² + sigma²)
        (onzekerheid in mu én natuurlijke spreiding tussen projecten)

        Verwachte duur (lognormaal gemiddelde):
          E[duur] = exp(mu_pred + (tau_n² + sigma²) / 2)

        Parameters
        ----------
        project_kenmerken : dict of None

        Returns
        -------
        tuple: (verwachte_duur, lower_90, upper_90) in DAGEN
        """
        mu_pred    = self.mu_n + self._covariate_effect(project_kenmerken)
        totale_var = self.tau_n**2 + self.sigma**2

        verwachte_duur = np.exp(mu_pred + totale_var / 2)
        lower = np.exp(stats.norm.ppf(0.05, mu_pred, np.sqrt(totale_var)))
        upper = np.exp(stats.norm.ppf(0.95, mu_pred, np.sqrt(totale_var)))

        return verwachte_duur, lower, upper

    def update(self, geobserveerde_duur, project_kenmerken=None):
        """
        Past mu en tau aan na een voltooid project.

        Normal-Normal conjugate update (Gelman et al. 2013, p. 43):
          tau_n²  =  1 / (1/tau_{n-1}² + 1/sigma²)
          mu_n    =  tau_n² × (mu_{n-1}/tau_{n-1}² + log(duur_adj)/sigma²)

        Parameters
        ----------
        geobserveerde_duur : float — werkelijke duur van het project (DAGEN)
        project_kenmerken  : dict of None — kenmerken van het voltooide project
        """
        # Verwijder het bekende covariate-effect voor een zuivere update op mu
        gecorrigeerde_log_duur = (
            np.log(geobserveerde_duur) - self._covariate_effect(project_kenmerken)
        )

        # Sla de oude waarden op vóór de update
        tau_oud_sq = self.tau_n**2
        mu_oud     = self.mu_n

        # Bereken nieuw posterior
        tau_nieuw_sq = 1.0 / (1.0 / tau_oud_sq + 1.0 / self.sigma**2)
        mu_nieuw     = tau_nieuw_sq * (
            mu_oud / tau_oud_sq + gecorrigeerde_log_duur / self.sigma**2
        )

        # Sla op en bewaar geschiedenis
        self.tau_n = np.sqrt(tau_nieuw_sq)
        self.mu_n  = mu_nieuw
        self.n_projecten += 1
        self.mu_geschiedenis.append(self.mu_n)
        self.tau_geschiedenis.append(self.tau_n)

    def reset(self):
        """
        Zet het model terug naar de beginprior.
        Gebruik dit om een simulatie-experiment te herhalen (SQ5-verificatie).
        """
        self.mu_n         = self.mu_0
        self.tau_n        = self.tau_0
        self.n_projecten  = 0
        self.mu_geschiedenis  = [self.mu_0]
        self.tau_geschiedenis = [self.tau_0]

    def __repr__(self):
        v, l, u = self.voorspel()
        return (
            f"BayesianesFaseModel('{self.naam}', "
            f"verwacht={v:.1f}d, 90%=[{l:.1f}–{u:.1f}]d, n={self.n_projecten})"
        )


# ==============================================================================
# SECTIE 3: SIMULATIE  (SQ5 — verificatie van het leergedrag)
# ==============================================================================

def simuleer_projecten(ware_mu, sigma, n, seed=42):
    """
    Genereert n gesimuleerde projectdurations voor verificatie.

    In de echte toepassing vervangt EaSiCon-data deze functie.
    Hier is ware_mu bekend (in tegenstelling tot de praktijk), zodat
    we kunnen controleren of het model convergeert.

    Parameters
    ----------
    ware_mu : float — de echte log(duur) (onbekend in praktijk)
    sigma   : float — spreiding van log(duur) tussen projecten
    n       : int   — aantal te simuleren projecten
    seed    : int   — voor reproduceerbaarheid

    Returns
    -------
    np.ndarray — durations in DAGEN
    """
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(loc=ware_mu, scale=sigma, size=n))


def voer_convergentie_experiment_uit(model, ware_mu, sigma, n_projecten=100, seed=42):
    """
    Voert het convergentie-experiment uit voor één fase.

    Reset het model, simuleert n_projecten, en slaat na elke update
    de schatting op. Hiermee toon je SQ5: convergeert het model naar
    de ware waarde als het een fout-prior heeft?

    Parameters
    ----------
    model       : BayesianesFaseModel
    ware_mu     : float — echte log(duur) voor de simulatie
    sigma       : float — observatieruis
    n_projecten : int
    seed        : int

    Returns
    -------
    dict met sleutels 'verwacht', 'lower', 'upper', 'tau', 'ware_duur'
    """
    model.reset()

    verwacht_lijst, lower_lijst, upper_lijst, tau_lijst = [], [], [], []

    # Schatting vóór enige observatie (= prior)
    v, l, u = model.voorspel()
    verwacht_lijst.append(v)
    lower_lijst.append(l)
    upper_lijst.append(u)
    tau_lijst.append(model.tau_n)

    # Update na elk gesimuleerd project
    for duur in simuleer_projecten(ware_mu, sigma, n_projecten, seed=seed):
        model.update(duur)
        v, l, u = model.voorspel()
        verwacht_lijst.append(v)
        lower_lijst.append(l)
        upper_lijst.append(u)
        tau_lijst.append(model.tau_n)

    return {
        "verwacht":  np.array(verwacht_lijst),
        "lower":     np.array(lower_lijst),
        "upper":     np.array(upper_lijst),
        "tau":       np.array(tau_lijst),
        "ware_duur": np.exp(ware_mu + sigma**2 / 2),
    }


# ==============================================================================
# SECTIE 4: GRAFIEKEN
# ==============================================================================

def maak_convergentiegrafiek(modellen, n_projecten=100, bestandsnaam="easicon_resultaten.png"):
    """
    Maakt convergentiegrafiek: 2×3 subplots, één per fase (100 projecten).

    Per subplot:
      - Gekleurde lijn    : schatting van het model na elk project
      - Gekleurd vlak     : 90% predictie-interval
      - Zwart gestippeld  : de 'ware' waarde (target voor convergentie)
      - Grijs gestippeld  : initiële prior-schatting
      - Tekstvak rechts   : covariaten van de fase met beta-waarden

    Parameters
    ----------
    modellen      : dict {fase_code: BayesianesFaseModel}
    n_projecten   : int — aantal projecten in de simulatie
    bestandsnaam  : str — pad voor de opgeslagen afbeelding
    """
    fig, assen = plt.subplots(2, 3, figsize=(16, 10))
    assen = assen.flatten()

    fig.suptitle(
        "EaSiCon — Bayesiaans Duratie-Schattingsmodel\n"
        f"Convergentie per bouwfase (simulatie ter verificatie, SQ5 — {n_projecten} projecten)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    milestones = [10, 25, 50]  # markeer deze projectaantallen met een streepje

    for i, (fase_code, model) in enumerate(modellen.items()):
        ax     = assen[i]
        config = FASE_INSTELLINGEN[fase_code]

        resultaat = voer_convergentie_experiment_uit(
            model, config["ware_mu"], SIGMA_LOG, n_projecten, seed=42 + i
        )

        x         = np.arange(n_projecten + 1)
        verwacht  = resultaat["verwacht"]
        lower     = resultaat["lower"]
        upper     = resultaat["upper"]
        ware_duur = resultaat["ware_duur"]
        prior_duur = verwacht[0]
        kleur      = config["kleur"]

        # 90%-interval als gekleurd vlak
        ax.fill_between(x, lower, upper,
                        alpha=0.20, color=kleur, label="90% interval")

        # Modelschatting
        ax.plot(x, verwacht, color=kleur, linewidth=2.5, label="Modelschatting")

        # Ware waarde (convergentiedoel)
        ax.axhline(ware_duur, color="black", linestyle="--",
                   linewidth=1.5, label=f"Ware waarde ({ware_duur:.0f}d)")

        # Prior
        ax.axhline(prior_duur, color="#888888", linestyle=":",
                   linewidth=1.2, alpha=0.8, label=f"Prior ({prior_duur:.0f}d)")

        # Milestone-markers
        for m in milestones:
            if m <= n_projecten:
                ax.axvline(m, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.text(m + 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1,
                        f"n={m}", fontsize=6, color="#888888", va="bottom")

        # Covariaten-tekstvak (linksboven)
        factoren = config.get("fase_specifieke_factoren", {})
        cov_regels = [
            f"GFA        β={BETA_GFA:.3f}/m²",
            f"Verdiep.   β={BETA_VERDIEPINGEN:.3f}",
            f"Hoogte     β={BETA_HOOGTE:.3f}/m",
            "─────────────────",
        ]
        for naam_f, coeff in factoren.items():
            markering = "?" if coeff == 0.0 else " "
            cov_regels.append(f"{naam_f:<14}{markering} β={coeff:.3f}")
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

        # Reset zodat teksttabel de prior toont, niet de simulatie-eindstand
        model.reset()

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafiek opgeslagen: {bestandsnaam}")


def maak_onzekerheidsgrafiek(modellen, n_projecten=100, bestandsnaam="easicon_zekerheid.png"):
    """
    Toont hoe de zekerheid per fase toeneemt na 0–100 projecten.

    Twee panelen:
      Links  — tau_n daalt richting 0: model leert het gemiddelde steeds beter.
               sigma (= SIGMA_LOG) is de onvermijdelijke projectvariabiliteit —
               die blijft altijd bestaan en wordt getoond als horizontale lijn.
      Rechts — 90%-intervalsbreedte in dagen. De gestippelde "sigma-vloer" toont
               de minimaal haalbare breedte: zelfs met oneindige data blijft die
               bestaan door inherente projectvariabiliteit (sigma).

    De vloer per fase = exp(ware_mu) * (exp(1.645*sigma) - exp(-1.645*sigma))
    = het interval bij tau_n=0, alleen nog bepaald door sigma.
    """
    fig, (ax_tau, ax_breedte) = plt.subplots(1, 2, figsize=(15, 6))

    fig.suptitle(
        f"EaSiCon — Zekerheidstoename per bouwfase ({n_projecten} projecten)\n"
        "Gekleurde lijn = model leert het gemiddelde  |  "
        "Gestippeld = onvermijdelijke projectvariabiliteit (sigma-vloer)",
        fontsize=12, fontweight="bold",
    )

    milestones = [10, 25, 50]

    for i, (fase_code, model) in enumerate(modellen.items()):
        config    = FASE_INSTELLINGEN[fase_code]
        kleur     = config["kleur"]
        ware_mu   = config["ware_mu"]
        label     = f"{fase_code}: {config['naam']}"

        resultaat = voer_convergentie_experiment_uit(
            model, ware_mu, SIGMA_LOG, n_projecten, seed=42 + i
        )

        x       = np.arange(n_projecten + 1)
        tau     = resultaat["tau"]
        breedte = resultaat["upper"] - resultaat["lower"]

        # Sigma-vloer: minimale intervalsbreedte bij tau_n = 0
        # (alleen nog inherente projectvariabiliteit)
        sigma_floor_breedte = (
            np.exp(ware_mu + 1.645 * SIGMA_LOG)
            - np.exp(ware_mu - 1.645 * SIGMA_LOG)
        )

        # --- Linkerpaneel: tau_n ---
        ax_tau.plot(x, tau, color=kleur, linewidth=2.2, label=label)
        # sigma als vaste horizontale lijn (het niveau dat tau nooit onderschrijdt)
        ax_tau.axhline(SIGMA_LOG, color="#CCCCCC", linestyle=":", linewidth=1.0)

        # Annoteer eindwaarde
        ax_tau.annotate(
            f"{tau[-1]:.3f}",
            xy=(n_projecten, tau[-1]),
            fontsize=6.5, color=kleur,
            xytext=(3, 0), textcoords="offset points", va="center",
        )

        # --- Rechterpaneel: intervalsbreedte ---
        ax_breedte.plot(x, breedte, color=kleur, linewidth=2.2, label=label)
        # Sigma-vloer als gestippelde horizontale lijn
        ax_breedte.axhline(
            sigma_floor_breedte, color=kleur, linestyle=":",
            linewidth=1.2, alpha=0.55,
        )
        # Annoteer vloer rechts van de grafiek
        ax_breedte.annotate(
            f"vloer {fase_code}: {sigma_floor_breedte:.0f}d",
            xy=(n_projecten, sigma_floor_breedte),
            fontsize=6.0, color=kleur, alpha=0.75,
            xytext=(3, 0), textcoords="offset points", va="center",
        )

        model.reset()

    # Sigma-label linkerpaneel
    ax_tau.text(
        1, SIGMA_LOG + 0.01,
        f"sigma = {SIGMA_LOG} (projectvariabiliteit, daalt niet)",
        fontsize=7, color="#999999", va="bottom",
    )

    for ax in (ax_tau, ax_breedte):
        for m in milestones:
            ax.axvline(m, color="#DDDDDD", linestyle="--", linewidth=0.9)
            ax.text(m + 0.5, 0, f"n={m}", fontsize=6.5, color="#AAAAAA", va="bottom")
        ax.set_xlabel("Voltooide projecten", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0, n_projecten)
        ax.set_ylim(bottom=0)

    ax_tau.set_title(
        "tau_n — leeronzekerheid op mu (log-schaal)\n"
        "daalt naar 0 naarmate meer projecten worden geobserveerd",
        fontsize=10, fontweight="bold",
    )
    ax_tau.set_ylabel("tau_n  (std posterior op log-schaal)", fontsize=10)

    ax_breedte.set_title(
        "90%-intervalsbreedte per fase (dagen)\n"
        "gestippeld = sigma-vloer: minimaal haalbaar, ook met oneindig veel data",
        fontsize=10, fontweight="bold",
    )
    ax_breedte.set_ylabel("Upper - Lower  (dagen)", fontsize=10)

    plt.tight_layout()
    plt.savefig(bestandsnaam, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafiek opgeslagen: {bestandsnaam}")


# ==============================================================================
# SECTIE 5: TEKSTOUTPUT
# ==============================================================================

def print_schattingstabel(modellen):
    """Print overzichtstabel met huidige schattingen + covariaten per fase."""
    breedte = 82
    print("\n" + "=" * breedte)
    print("  EaSiCon — Huidige faseschattingen  (prior, vóór projectdata)")
    print("=" * breedte)
    print(f"  {'Fase':<5} {'Naam':<36} {'Verwacht':>9} {'90%-interval':>20}")
    print("-" * breedte)

    for fase_code, model in modellen.items():
        config = FASE_INSTELLINGEN[fase_code]
        naam   = config["naam"]
        v, l, u = model.voorspel()
        print(f"  {fase_code:<5} {naam:<36} {v:>7.1f}d  [{l:>5.1f} – {u:>6.1f}d]")

        # Algemene covariaten
        print(f"  {'':>5}  ├─ Algemeen :  "
              f"GFA β={BETA_GFA:.3f}/m²  |  "
              f"Verdiep. β={BETA_VERDIEPINGEN:.2f}  |  "
              f"Hoogte β={BETA_HOOGTE:.3f}/m")

        # Fase-specifieke covariaten
        factoren = config.get("fase_specifieke_factoren", {})
        items    = list(factoren.items())
        for j, (naam_f, coeff) in enumerate(items):
            prefix  = "  └─" if j == len(items) - 1 else "  ├─"
            todo    = "  ← !! TODO" if coeff == 0.0 else ""
            print(f"  {'':>5} {prefix} {naam_f:<22}  β = {coeff:+.4f}{todo}")
        print()

    print("-" * breedte)
    print("  * Alle waarden zijn PLACEHOLDERS op basis van literatuurschattingen.")
    print("    Vervang mu_0 en tau_0 per fase na expert-interviews (zie TODO's).")
    print("    β = 0.0000 betekent: coëfficiënt nog niet ingesteld (TODO).")
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
                "         Zet mu_0 = log(meest_waarschijnlijk) in FASE_INSTELLINGEN."
            ),
            "locatie": "FASE_INSTELLINGEN -> mu_0 per fase  (nu: log(15), log(25), ...)",
        },
        {
            "nr": 2,
            "wat": "Onzekerheidsparameter tau_0 berekenen uit drie-punts-schatting",
            "hoe": (
                "Gebruik de drie-puntsmethode (PERT-achtig):\n"
                "         tau_0 = (log(pessimistisch) - log(optimistisch)) / (2 × 1.645)\n"
                "         Huidig: 0.5 voor alle fases (= grote onzekerheid)."
            ),
            "locatie": "FASE_INSTELLINGEN -> tau_0 per fase",
        },
        {
            "nr": 3,
            "wat": "Fase-specifieke covariaten invullen (3 per fase, nu alle β=0.0)",
            "hoe": (
                'Stel per fase: "Welke 3 projectkenmerken bepalen het meest\n'
                "         hoe lang fase X duurt?\" Stel coëfficiënten in op basis\n"
                "         van de interviews. Eenheden vastleggen en documenteren."
            ),
            "locatie": "FASE_INSTELLINGEN -> fase_specifieke_factoren per fase",
        },
        {
            "nr": 4,
            "wat": "Algemene covariate-coëfficiënten valideren voor Eemsdelta-woningen",
            "hoe": (
                "BETA_GFA (0.002/m²), BETA_VERDIEPINGEN (0.08), BETA_HOOGTE (0.015/m)\n"
                "         zijn literatuurplaceholders. Controleer orde van grootte voor\n"
                "         typische Eemsdelta-woning: ~100 m², 1–2 lagen, ~6–8 m hoog."
            ),
            "locatie": "BETA_GFA, BETA_VERDIEPINGEN, BETA_HOOGTE — bovenaan script",
        },
        {
            "nr": 5,
            "wat": "Observatieruis SIGMA_LOG valideren",
            "hoe": (
                "SIGMA_LOG = 0.4 -> ~40% variatie tussen vergelijkbare projecten.\n"
                "         Bespreek met Ragnar Klabbers of dit realistisch is voor\n"
                "         VolkerWessels-renovaties in Eemsdelta."
            ),
            "locatie": "SIGMA_LOG — bovenaan script",
        },
        {
            "nr": 6,
            "wat": "SQ5-verificatie afronden: drie experimenten controleren",
            "hoe": (
                "Voer uit nadat echte priors zijn ingesteld:\n"
                "         (a) Convergentie: daalt tau_n na elk project?\n"
                "         (b) Correctie: corrigeert model een bewust foute prior?\n"
                "         (c) Stabilisatie: wordt schatting stabieler na ~20–50 projecten?\n"
                "         -> easicon_resultaten.png + easicon_zekerheid.png tonen dit."
            ),
            "locatie": "voer_convergentie_experiment_uit()  /  beide PNG-bestanden",
        },
        {
            "nr": 7,
            "wat": "Echte EaSiCon-projectdata koppelen zodra beschikbaar",
            "hoe": (
                "Vervang simuleer_projecten() door een functie die data inleest\n"
                "         vanuit het EaSiCon datamanagementsysteem (VolkerWessels).\n"
                "         Roep daarna: model.update(duur_in_dagen, project_kenmerken) aan."
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
    N_PROJECTEN = 100  # simulatielengte voor beide grafieken

    print("\nEaSiCon Bayesiaans Duratie-Schattingsmodel")
    print("Gijs Meijers (4957822) | TU Delft x Haskoning | 2026")

    # Maak modelinstanties voor alle 6 fases
    modellen = {
        code: BayesianesFaseModel(
            naam=f"{code} — {cfg['naam']}",
            mu_0=cfg["mu_0"],
            tau_0=cfg["tau_0"],
            sigma=SIGMA_LOG,
            fase_config=cfg,
        )
        for code, cfg in FASE_INSTELLINGEN.items()
    }

    # 1. Tabel met huidige schattingen + covariaten (prior-stand)
    print_schattingstabel(modellen)

    # 2. Openstaande TODO's
    print_todo_lijst()

    # 3. Convergentiegrafiek met covariaat-annotaties (100 projecten)
    print("  Grafieken worden aangemaakt...")
    maak_convergentiegrafiek(modellen, n_projecten=N_PROJECTEN)

    # 4. Zekerheidstoename-grafiek (tau + intervalsbreedte over 100 projecten)
    maak_onzekerheidsgrafiek(modellen, n_projecten=N_PROJECTEN)

    print("\nKlaar.")
    print("  easicon_resultaten.png — convergentie per fase (incl. covariaten)")
    print("  easicon_zekerheid.png  — zekerheidstoename over 100 projecten")
    print()


if __name__ == "__main__":
    main()
