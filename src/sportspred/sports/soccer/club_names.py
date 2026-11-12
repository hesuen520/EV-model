"""Club names: one club, one participant, across the two club sources.

The results file (football-data.co.uk names such as ``Man City``) is the
source of record, so its names are the canonical ones.  The fixture files
(openfootball names such as ``Manchester City FC``) are mapped onto them with
the explicit table below - never by fuzzy matching.  Every pair was checked
against the data: for the 2025/26 season all fixtures of a mapped club agree
between the two sources on date and score (see docs/sports/soccer.md).  A
fixture whose club is not in the table is skipped with a warning; add the
club here when a newly promoted side appears.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["RESULT_NAME_VARIANTS", "FIXTURE_NAMES", "canonical_result_name"]

#: Two spellings of one club inside the results file -> the spelling kept.
RESULT_NAME_VARIANTS: Dict[str, str] = {
    "Nottm Forest": "Nott'm Forest",
    "MGladbach": "M'gladbach",
}

#: Fixture-file name -> results-file name, per league.
FIXTURE_NAMES: Dict[str, Dict[str, str]] = {
    "epl": {
        "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
        "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Burnley FC": "Burnley",
        "Chelsea FC": "Chelsea", "Coventry City FC": "Coventry", "Crystal Palace FC": "Crystal Palace",
        "Everton FC": "Everton", "Fulham FC": "Fulham", "Hull City AFC": "Hull", "Ipswich Town FC": "Ipswich",
        "Leeds United FC": "Leeds", "Liverpool FC": "Liverpool", "Manchester City FC": "Man City",
        "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
        "Nottingham Forest FC": "Nott'm Forest", "Sunderland AFC": "Sunderland",
        "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham",
        "Wolverhampton Wanderers FC": "Wolves",
    },
    "laliga": {
        "Athletic Club": "Ath Bilbao", "CA Osasuna": "Osasuna", "Club Atlético de Madrid": "Ath Madrid",
        "Deportivo Alavés": "Alaves", "Elche CF": "Elche", "FC Barcelona": "Barcelona", "Getafe CF": "Getafe",
        "Girona FC": "Girona", "Levante UD": "Levante", "Málaga CF": "Malaga", "RC Celta de Vigo": "Celta",
        "RC Deportivo La Coruña": "La Coruna", "RCD Espanyol de Barcelona": "Espanol", "RCD Mallorca": "Mallorca",
        "Rayo Vallecano de Madrid": "Vallecano", "Real Betis Balompié": "Betis", "Real Madrid CF": "Real Madrid",
        "Real Oviedo": "Oviedo", "Real Racing Club de Santander": "Santander",
        "Real Sociedad de Fútbol": "Sociedad", "Sevilla FC": "Sevilla", "Valencia CF": "Valencia",
        "Villarreal CF": "Villarreal",
    },
    "bundesliga": {
        "1. FC Heidenheim 1846": "Heidenheim", "1. FC Köln": "FC Koln", "1. FC Union Berlin": "Union Berlin",
        "1. FSV Mainz 05": "Mainz", "Bayer 04 Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
        "Borussia Mönchengladbach": "M'gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Augsburg": "Augsburg", "FC Bayern München": "Bayern Munich", "FC Schalke 04": "Schalke 04",
        "FC St. Pauli 1910": "St Pauli", "Hamburger SV": "Hamburg", "RB Leipzig": "RB Leipzig",
        "SC Freiburg": "Freiburg", "SC Paderborn 07": "Paderborn", "SV 07 Elversberg": "Elversberg",
        "SV Werder Bremen": "Werder Bremen", "TSG 1899 Hoffenheim": "Hoffenheim", "VfB Stuttgart": "Stuttgart",
        "VfL Wolfsburg": "Wolfsburg",
    },
    "seriea": {
        "AC Milan": "Milan", "AC Monza": "Monza", "AC Pisa 1909": "Pisa", "ACF Fiorentina": "Fiorentina",
        "AS Roma": "Roma", "Atalanta BC": "Atalanta", "Bologna FC 1909": "Bologna", "Cagliari Calcio": "Cagliari",
        "Como 1907": "Como", "FC Internazionale Milano": "Inter", "Frosinone Calcio": "Frosinone",
        "Genoa CFC": "Genoa", "Hellas Verona FC": "Verona", "Juventus FC": "Juventus",
        "Parma Calcio 1913": "Parma", "SS Lazio": "Lazio", "SSC Napoli": "Napoli", "Torino FC": "Torino",
        "US Cremonese": "Cremonese", "US Lecce": "Lecce", "US Sassuolo Calcio": "Sassuolo",
        "Udinese Calcio": "Udinese", "Venezia FC": "Venezia",
    },
    "ligue1": {
        "AJ Auxerre": "Auxerre", "AS Monaco FC": "Monaco", "Angers SCO": "Angers", "ES Troyes AC": "Troyes",
        "FC Lorient": "Lorient", "FC Metz": "Metz", "FC Nantes": "Nantes", "Le Havre AC": "Le Havre",
        "Le Mans FC": "Le Mans", "Lille OSC": "Lille", "OGC Nice": "Nice", "Olympique Lyonnais": "Lyon",
        "Olympique de Marseille": "Marseille", "Paris FC": "Paris FC", "Paris Saint-Germain FC": "Paris SG",
        "RC Strasbourg Alsace": "Strasbourg", "Racing Club de Lens": "Lens", "Stade Brestois 29": "Brest",
        "Stade Rennais FC 1901": "Rennes", "Toulouse FC": "Toulouse",
    },
}


def canonical_result_name(name: str) -> str:
    return RESULT_NAME_VARIANTS.get(name, name)
