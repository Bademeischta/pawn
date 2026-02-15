## Was die neuen Fehler bedeuten
- **Dashboard: `Can't get attribute 'ReplayBuffer'`**
  - Dein vorhandener Checkpoint enthält ein pickled `ReplayBuffer`-Objekt (aus Training-Lauf vor den Änderungen). Beim Laden in Streamlit ist das Modul `main` effektiv `dashboard.py`, und dort gibt es keine Klasse `ReplayBuffer` → Unpickling scheitert.
- **Training: `fatal: Memory allocation failure` + danach `CUDA error: unknown error` in `save_checkpoint`**
  - Der eigentliche CUDA/OOM passiert im Training, und der nachfolgende NaN/Inf-Check in `save_checkpoint` ([train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L372-L401)) triggert dann die nächste CUDA-Operation und wirft die „unknown error“ Exception.

## Ziel
- Dashboard soll **auch alte Checkpoints** (mit ReplayBuffer) wieder laden können.
- Training soll bei **CUDA-OOM/Allocator-Problemen** stabiler reagieren (oder sauber auf CPU fallen), und beim Speichern nicht zusätzlich crashen.

## Änderungen
### 1) Dashboard: Legacy-Checkpoint laden
- In [dashboard.py](file:///c:/pawn/dashboard.py):
  - Eine minimale `ReplayBuffer`-Klasse hinzufügen (ohne Logik), damit das Unpickling alter Checkpoints `main.ReplayBuffer` auflösen kann.
  - Optional: zusätzlich eine „weights-only“-Checkpoint-Variante unterstützen (falls später vorhanden), damit Dashboard auch reine `state_dict` Dateien laden kann.

### 2) Training: Speichern OOM-sicher machen
- In [train_end_to_end.py](file:///c:/pawn/train_end_to_end.py):
  - Den NaN/Inf-Check in `save_checkpoint` so umbauen, dass er CUDA-Fehler nicht triggert (z.B. CPU-basierte Prüfung in `try/except` oder bei CUDA-Fehlern komplett überspringen).
  - In jedem Fall zusätzlich eine zweite Datei schreiben, die **nur** `model_state_dict` enthält (z.B. `checkpoints/latest_model.pt`). Diese Datei ist klein und Dashboard-freundlich.

### 3) Training: OOM/Allocator-Probleme abfangen
- In [train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L338-L370):
  - Um `forward/backward` einen Guard bauen, der `out of memory`/`Memory allocation failure` erkennt:
    - Gradients verwerfen, `torch.cuda.empty_cache()` aufrufen,
    - den Batch überspringen oder Training sauber abbrechen mit verständlicher Logmeldung.
  - Defaults sicherer machen:
    - `--batch-size` Default auf einen kleineren Wert (z.B. 16) setzen.
    - Optional `--device cpu|cuda` anbieten, um gezielt CPU zu erzwingen.

## Verifikation
- Dashboard starten und prüfen, dass ein vorhandener (alter) `latest.pt` wieder lädt (kein `ReplayBuffer`-Attributfehler).
- Training mit kleinerem Batch starten (z.B. Default/16) und prüfen:
  - kein Crash in `save_checkpoint` mehr,
  - falls OOM weiterhin auftritt: saubere, verständliche Fehlermeldung statt „unknown error“.
