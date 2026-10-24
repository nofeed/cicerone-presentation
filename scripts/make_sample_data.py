"""Genera un catalogo di film (rigorosamente inventati) per provare Cicerone.

Uso: python make_sample_data.py <cartella_di_output>

Scrive events.parquet, users.parquet e items.parquet con le colonne attese
da config/features.toml. Il seed è fisso: ogni esecuzione produce gli
stessi dati (le date sono relative a oggi).
"""

import random
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SEED = 42
N_USERS = 200
COUNTRIES = ["IT", "CH", "SM"]  # Italia, Svizzera, San Marino: diritti diversi
DECADES = ["1980s", "1990s", "2000s", "2010s", "2020s"]
DURATIONS = ["corto", "normale", "director's cut infinito"]

TITLES = {
    "commedia": [
        "Mia Suocera è un Vampiro",
        "Il Pranzo della Domenica (Director's Cut, 6 ore)",
        "Due Cuori e una Partita IVA",
        "Ferie d'Agosto in Tangenziale",
        "L'Ultimo Parcheggio Libero",
        "Zio Gino Scopre TikTok",
        "Benvenuti al Nord-Est",
        "Il Ragioniere che Sapeva Usare Excel",
    ],
    "horror": [
        "Lo Squalo del Lago di Como",
        "Il Ritorno della Lavatrice Assassina",
        "Non Aprite quel Frigo",
        "La Notte dei Commercialisti Viventi",
        "Riunione che Poteva Essere una Mail",
        "It: la Bolletta",
        "Paranormal Wi-Fi",
        "Il Gatto che Fissava il Muro",
    ],
    "fantascienza": [
        "Guerre Stellari in Ape Piaggio",
        "2001: Odissea in Tangenziale",
        "Ritorno al Lunedì",
        "Interstellar Pendolare",
        "Matrix: Aggiornamento Obbligatorio",
        "Alien contro l'Amministratore di Condominio",
        "Blade Runner a Rate",
        "Il Robot che Voleva le Ferie",
    ],
    "drammatico": [
        "La Vita è Bella ma il Mutuo No",
        "Nuovo Cinema Parcheggio",
        "Ladri di Monopattini",
        "La Grande Bruttezza",
        "Il Sorpasso in Doppia Fila",
        "Ultimo Tango a Pescara",
        "L'Attimo Fuggente del Regionale delle 7:42",
        "Il Postino Suona Solo Quando Non Ci Sei",
    ],
    "animazione": [
        "Il Re Leone da Tastiera",
        "Alla Ricerca del Telecomando",
        "Shrek all'Ikea",
        "Toy Story 7: il Ritorno del Ditale",
        "Kung Fu Panda in Smart Working",
        "Gli Incredibili Ritardatari",
        "Up: la Casa che Volò in Comune",
        "Frozen: Lascia Andare la Bolletta del Gas",
    ],
    "thriller": [
        "Il Silenzio degli Inquilini",
        "Seven: Sette Giorni Senza Caffè",
        "Il Codice Fiscale da Vinci",
        "Psycho Parcheggiatore",
        "L'Uomo che Sapeva Troppo (del Gruppo WhatsApp)",
        "Fuga dalla Riunione delle 18",
        "La Talpa in Ufficio",
        "Memento: Dove Ho Messo le Chiavi",
    ],
}

EVENT_TYPES = [
    # (tipo, probabilità relativa)
    ("view", 55),               # ha aperto la scheda del film
    ("trailer", 15),            # ha guardato il trailer
    ("watched", 15),            # l'ha visto (quantity = quante volte)
    ("watchlist", 8),           # l'ha salvato per dopo (mentendo a se stesso)
    ("review_positive", 5),
    ("review_negative", 2),
]


def slug(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")


def make_items(rng: random.Random, now: datetime) -> pd.DataFrame:
    rows = []
    for genre, titles in TITLES.items():
        for title in titles:
            rows.append(
                {
                    "item_id": slug(title),
                    "title": title,
                    "genre": genre,
                    "director_id": f"regista-{rng.randint(1, 12)}",
                    "decade": rng.choice(DECADES),
                    "duration_bucket": rng.choice(DURATIONS),
                    "published": True,
                    # Ogni tanto la licenza scade: il film sparisce dai suggerimenti.
                    "licensed": rng.random() > 0.1,
                    "published_at": now - timedelta(days=rng.randint(1, 365)),
                    # Colonne per le ricette di business (slide "Regole di business").
                    "available_countries": rng.sample(COUNTRIES, rng.randint(1, 3)),
                    "is_original": rng.random() < 0.2,
                }
            )
    return pd.DataFrame(rows)


def make_users(rng: random.Random) -> pd.DataFrame:
    genres = list(TITLES)
    rows = []
    for i in range(N_USERS):
        rows.append(
            {
                "user_id": f"u{i + 1:03d}",
                "favorite_genres": rng.sample(genres, rng.randint(1, 2)),
                "country": rng.choice(COUNTRIES),
            }
        )
    return pd.DataFrame(rows)


def make_events(rng: random.Random, users: pd.DataFrame, items: pd.DataFrame, now: datetime) -> pd.DataFrame:
    by_genre = items.groupby("genre")["item_id"].apply(list).to_dict()
    all_items = items["item_id"].tolist()
    types, weights = zip(*EVENT_TYPES)
    rows = []
    for user in users.itertuples():
        # Storie di lunghezza molto diversa: dal nuovo iscritto (cold start)
        # a chi ha finito il catalogo due volte.
        n_events = rng.choice([0, 1, 2, 3, 5, 8, 15, 30, 60])
        for _ in range(n_events):
            # 80% degli eventi sui generi preferiti, 20% "vediamo che c'è".
            if rng.random() < 0.8:
                item_id = rng.choice(by_genre[rng.choice(user.favorite_genres)])
            else:
                item_id = rng.choice(all_items)
            event_type = rng.choices(types, weights)[0]
            rows.append(
                {
                    "user_id": user.user_id,
                    "item_id": item_id,
                    "event_type": event_type,
                    "quantity": rng.choice([1, 1, 1, 2, 3, 7]) if event_type == "watched" else 1,
                    "occurred_at": now - timedelta(days=rng.uniform(0, 180)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("uso: python make_sample_data.py <cartella_di_output>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    now = datetime.now(timezone.utc)
    items = make_items(rng, now)
    users = make_users(rng)
    events = make_events(rng, users, items, now)

    events.to_parquet(out / "events.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    print(f"{len(events)} eventi, {len(users)} utenti, {len(items)} film scritti in {out}/")


if __name__ == "__main__":
    main()
