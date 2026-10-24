# Cicerone · la presentazione

Una presentazione per principianti su [Cicerone](https://cicerone.dev) 0.8.5,
più tutto quello che serve per rifare la demo dal vivo: configurazioni,
dataset di esempio (film, rigorosamente inventati) e istruzioni per Docker e
per pip.

```
.
├── cicerone.md                  # le slide (presenterm)
├── cicerone.pdf                 # le slide esportate
├── note-relatore.md             # cosa dire, slide per slide (non versionato)
├── docker-compose.yml           # recommender + serve + dashboard
├── .env.example                 # i segreti (token)
├── config/
│   ├── cicerone.toml            # job batch: dati, modelli, cron, trigger
│   ├── cicerone.serve.toml      # API di lettura (+ eventi in giornata, commentati)
│   ├── cicerone.dashboard.toml  # pagina di stato
│   └── features.toml            # pesi eventi, caratteristiche, regole
└── scripts/
    └── make_sample_data.py      # genera 200 utenti, 48 film, ~3200 eventi
```

Le stesse configurazioni funzionano con Docker e con pip:

- i percorsi sono **relativi a questa cartella** (`data/input`,
  `config/features.toml`). In Docker la cartella è montata in `/app`, che è
  la cartella di lavoro dell'immagine;
- i servizi ascoltano su **127.0.0.1** (il default). In Docker si usa
  `network_mode: host`, quindi anche lì sono raggiungibili solo dalla
  macchina stessa.

Dopo il primo giro la cartella `data/` contiene:

```
data/input/   events.parquet, users.parquet, items.parquet   (dataset di esempio)
data/output/  recommendations.parquet, manifest.json, items_snapshot.parquet
```

| Servizio  | Indirizzo                               | Accesso                               |
| --------- | --------------------------------------- | ------------------------------------- |
| serve     | http://127.0.0.1:8000/docs              | `Authorization: Bearer $SERVE_AUTH_TOKEN` |
| trigger   | http://127.0.0.1:8080/trigger/retrain   | `Authorization: Bearer $TRIGGER_AUTH_TOKEN` |
| dashboard | http://127.0.0.1:8090/dashboard         | utente e password (vedi sotto)        |

---

## Con Docker

Serve Docker con Compose v2 su Linux (su macOS / Windows attivate "host
networking" in Docker Desktop). Non serve clonare Cicerone: l'immagine viene
costruita dal tag `v0.8.5` su GitHub. La prima build richiede qualche minuto
perché compila LightFM.

```bash
cp .env.example .env          # e cambiate i due token
mkdir -p data                 # crearla voi: se la crea Docker è di root
docker compose build
```

Generate il dataset di esempio:

```bash
docker compose run --rm --no-deps --entrypoint python recommender \
  scripts/make_sample_data.py data/input
```

Create un utente per la dashboard (chiede la password due volte):

```bash
docker compose run --rm dashboard \
  users --config config/cicerone.dashboard.toml add alice
```

Avviate tutto:

```bash
docker compose up -d
docker compose logs -f recommender     # il primo giro parte subito
```

Provate:

```bash
set -a; source .env; set +a

# I 3 suggerimenti per l'utente u001
curl -s -H "Authorization: Bearer $SERVE_AUTH_TOKEN" \
  "http://127.0.0.1:8000/recommendations/u001?limit=3"

# Solo horror
curl -s -H "Authorization: Bearer $SERVE_AUTH_TOKEN" \
  "http://127.0.0.1:8000/recommendations/u001?limit=3&category=horror"

# Retrain subito, senza aspettare le 03:00 UTC
curl -s -X POST -H "Authorization: Bearer $TRIGGER_AUTH_TOKEN" \
  http://127.0.0.1:8080/trigger/retrain
```

Dopo aver modificato un file in `config/`:

- `features.toml` vale dal giro successivo: lanciate il trigger qui sopra;
- `cicerone.toml` si legge all'avvio: `docker compose restart recommender`
  (rifà anche un giro);
- `serve` e `dashboard` rileggono i risultati da soli
  (`refresh_interval_seconds`); un `restart` serve solo se cambiate i loro
  file di configurazione.

Per fermare tutto: `docker compose down`.

---

## Con pip

Serve **Python 3.11** (Cicerone 0.8.5 non supporta altre versioni). Su
alcuni sistemi LightFM va compilato: servono `gcc`/`g++` e OpenMP
(`libgomp`). Il pacchetto su PyPI si chiama `cicerone-recommender`: il
pacchetto `cicerone` è un altro progetto.

```bash
uv venv --python 3.11 .venv          # oppure: python3.11 -m venv .venv
source .venv/bin/activate
pip install cicerone-recommender==0.8.5   # con uv: uv pip install ...

cp .env.example .env                 # e cambiate i due token
set -a; source .env; set +a          # cicerone legge i segreti dall'ambiente
```

Tutti i comandi vanno lanciati **da questa cartella** (i percorsi nelle
config sono relativi).

```bash
python scripts/make_sample_data.py data/input
cicerone job --config config/cicerone.toml          # un giro ed esce
cicerone users --config config/cicerone.dashboard.toml add alice
```

Poi, in tre terminali diversi (in ognuno: `source .venv/bin/activate` e
`set -a; source .env; set +a`):

```bash
cicerone start --config config/cicerone.toml              # job + cron + trigger
cicerone start --config config/cicerone.serve.toml        # API
cicerone dashboard --config config/cicerone.dashboard.toml
```

I `curl` sono gli stessi della sezione Docker. Si ferma tutto con Ctrl-C.

---

## Provare le alternative delle slide

Ogni alternativa è già scritta, commentata, nei file di `config/`.
Decommentatela, rilanciate il giro e guardate cosa cambia. Tra parentesi,
quello che abbiamo visto noi sul dataset di esempio.

| Passo                    | Dove                                 | Cosa decommentare                    | Cosa succede |
| ------------------------ | ------------------------------------ | ------------------------------------ | ------------ |
| 7a · RRF                 | `cicerone.toml`                      | `rrf_k` e `[job.model_weights]`      | `source` diventa un'etichetta composta, es. `personalized+item_based` |
| 9 · AutoML               | `cicerone.toml`                      | `[job.automl]`                       | nei log il punteggio di ogni candidato e quello scelto (da noi una fusione di 7 strategie) |
| 7b · Blending            | `features.toml`                      | `[blending]`                         | compaiono `blended` e la lista per gli sconosciuti (`__cold_start__`); la strategia `latest` viene saltata (warning) |
| 8 · Regole di business   | `features.toml`                      | `[[eligibility]]` e/o `[[boost]]`    | nessun film senza diritti nel paese dell'utente; con il boost gli "original" passano da 0 a 185 righe |
| 12 · Eventi e online training | `cicerone.serve.toml` + `cicerone.toml` + `.env` | in serve `[events]`, `[events.incremental]`, `[events.online]` e `artifact_hmac_key`; nel job `save_model_artifact` e `artifact_hmac_key`; in `.env` `ARTIFACT_HMAC_KEY` | il job scrive anche `model.artifact`; `POST /events` risponde 202 e dopo il flush (al massimo 60 s) la lista dell'utente cambia: i film appena toccati ai primi 3 posti (`incremental`), il resto da LightFM riallenato. Con Docker passate `ARTIFACT_HMAC_KEY` nel `docker-compose.yml` |
| 9b · A/B test           | `cicerone.toml` + `cicerone.serve.toml` | `[experiment]` e le due `[[experiment.variants]]`, **in tutti e due i file** | il job scrive una lista per variante (`control` tutta `personalized`, `treatment` 1870 righe su 2000 `item_based`); `serve` divide i 200 utenti 100 e 100 e risponde con `experiment_id` e `variant`; l'online training è spento |
| 2 · Da dove leggo (S3)   | `cicerone.toml` + `.env`             | `[input.options]` / `[output.options]` S3 | stessi risultati, ma su un bucket. Copiate lo stesso `[output.options]` in serve e dashboard e, con Docker, passate le variabili S3 nel `docker-compose.yml` |

Per vedere i risultati senza API:

```bash
# pip
python -c "import pandas as pd; print(pd.read_parquet('data/output/recommendations.parquet').head(20))"
# Docker
docker compose run --rm --no-deps --entrypoint python recommender -c \
  "import pandas as pd; print(pd.read_parquet('data/output/recommendations.parquet').head(20))"
```

Cose da sapere, scoperte provando:

- **Utente sconosciuto → 404.** Con il combinatore a priorità tutti gli
  utenti del dataset hanno suggerimenti personalizzati, quindi non c'è una
  lista "popolare" da usare come ripiego. Con `[blending]` attivo, invece,
  il job scrive una lista per gli sconosciuti e l'API risponde con
  `"fallback": true`.
- **`${VARIABILE}` funziona solo nei segreti e nelle tabelle `options`**
  (`[input.options]`, `[output.options]`, token...). Percorsi come
  `feature_config_path` o `users_path` e `host` sono presi alla lettera.
- La dashboard non parte finché non esiste almeno un utente.
- Appena dopo `docker compose up`, `serve` può rispondere **404** per
  circa un minuto: è partito prima che il job scrivesse i risultati e li
  ricarica al prossimo `refresh_interval_seconds` (60 s).
- `?category=horror` filtra i `top_k` suggerimenti **già salvati**: se
  nessuno dei 10 è un horror, la lista è vuota. Non calcola nulla di nuovo.
- Rilanciando il job l'ordine dei suggerimenti può cambiare un po': il
  seme di LightFM è fisso, ma l'allenamento gira su 4 thread in parallelo.
- `source` per LightFM vale `personalized`, non `collaborative`.
- **Eventi in giornata:** un `202` da `POST /events` vuol dire "in coda",
  non "applicato". Con `kind = "webhook"` la coda è in memoria e un
  riavvio di `serve` la perde. Con `fit_min_events = 100` pochi eventi di
  prova non riallenano LightFM: per vederlo abbassatelo a 1 (e
  `batch_window_seconds` a 10 per non aspettare).
- Gli eventi ricevuti da `POST /events` **non** vengono aggiunti a
  `data/input/events.parquet`: il giro notturno li vede solo se l'app li
  scrive anche lì.
- I film appena visti finiscono ai primi 3 posti della lista (30% di
  `top_k`, `source = "incremental"`), anche se l'utente li ha già visti.
  Non c'è un'opzione per spegnerlo.

---

## Le slide

Servono [presenterm](https://github.com/mfontanini/presenterm) e, per il
PDF, `weasyprint` (`sudo pacman -S presenterm python-weasyprint`).

```bash
presenterm cicerone.md               # presentare (frecce per muoversi)
presenterm --export-pdf cicerone.md  # rigenera cicerone.pdf
```

L'export usa la dimensione del terminale. Il PDF incluso è stato generato
a 110 colonne × 32 righe; da uno script senza terminale:

```bash
script -qec "stty cols 110 rows 32; presenterm --export-pdf cicerone.md" /dev/null
```
