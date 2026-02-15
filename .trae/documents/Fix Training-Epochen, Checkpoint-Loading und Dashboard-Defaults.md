## Ursache (was gerade passiert)
- Phase 1 endet frühzeitig, weil in [train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L469-L491) ein Early-Stop eingebaut ist: sobald `acc > 0.55`, wird `break` ausgeführt.
- Phase 2 startet immer bei Epoche **51**, weil `start_ep_rl = max(self.current_epoch + 1, 51)` gesetzt ist ([train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L513-L515)). Das lässt die Nummerierung so aussehen, als würden Epochen fehlen und sorgt außerdem dafür, dass der Stockfish-Scheduler direkt im höchsten Skill-Bereich landet.
- Das Dashboard lädt standardmäßig `checkpoints/latest_checkpoint.pt`, aber das Training schreibt `checkpoints/latest.pt` ([train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L395-L413)) → Pfad stimmt nicht.
- Der Dashboard-Fehler entsteht, weil `safe_load_checkpoint` immer mit `weights_only=True` lädt ([utils.py](file:///c:/pawn/utils.py#L46-L60)), aber deine Checkpoints enthalten ein pickled `ReplayBuffer`-Objekt (`data['replay_buffer'] = replay_buffer`) ([train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L371-L392)). PyTorch 2.6 blockt das standardmäßig.
- `distillzero_factory.py` ist kein Trainer, sondern ein **Dataset-Generator** (PGN + Stockfish → HDF5) ([distillzero_factory.py](file:///c:/pawn/distillzero_factory.py#L623-L688)).

## Änderungen (Code)
### 1) Training läuft wirklich „alle Epochen“
- In [train_end_to_end.py](file:///c:/pawn/train_end_to_end.py):
  - Early-Stop in Phase 1 konfigurierbar machen (z.B. `--supervised-early-stop-acc`, Default: deaktiviert oder sehr hoch), damit Training nicht nach ~Epoche 7 endet.
  - Phase-2-Startlogik ändern: nicht mehr hart auf 51 springen. Stattdessen Phase 2 bei `self.current_epoch + 1` starten, sodass bis `--epochs` wirklich durchtrainiert wird.
  - Stockfish-Scheduler-Epoche entkoppeln (für Phase 2 eine eigene RL-Epochenzählung), damit die Skill-Level wieder sinnvoll ansteigen.

### 2) Checkpoints kompatibel mit PyTorch 2.6 + Dashboard
- In [train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L371-L400):
  - Nicht mehr `ReplayBuffer` als Objekt speichern, sondern nur eine „einfache“ Struktur (z.B. `replay_buffer.buffer` als Liste). Das ist bereits kompatibel, weil der Loader den Legacy-List-Fall schon unterstützt ([train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L505-L512)).
- In [utils.py](file:///c:/pawn/utils.py):
  - `safe_load_checkpoint` erweitern: wenn `weights_only=True` mit dem typischen PyTorch-2.6-Fehler scheitert („Weights only load failed“/„Unsupported global“), dann für lokale Checkpoints automatisch einmal mit `weights_only=False` retryen (mit klarer Warnung), damit auch alte Checkpoints weiter ladbar sind.

### 3) Dashboard findet den richtigen Checkpoint
- In [dashboard.py](file:///c:/pawn/dashboard.py):
  - Default von `checkpoints/latest_checkpoint.pt` auf `checkpoints/latest.pt` ändern.
  - Zusätzlich eine Auto-Detection: wenn der angegebene Pfad nicht existiert, dann im `checkpoints/`-Ordner den neuesten `.pt`-Checkpoint wählen (z.B. nach `mtime`). Das fängt auch ungewöhnliche Dateinamen wie `latest @.pt` ab.

### 4) DistillZero-Factory Stabilität (optional, aber sinnvoll)
- In [distillzero_factory.py](file:///c:/pawn/distillzero_factory.py):
  - `_process_batch` sollte **immer** `(results, batch_idx)` zurückgeben. Aktuell kann es `[]` zurückgeben ([distillzero_factory.py](file:///c:/pawn/distillzero_factory.py#L382-L384)), was beim Unpacking in `for result_batch, batch_game_idx in ...` crashen kann.

### 5) AMP Deprecation-Warnings (optional)
- In [train_end_to_end.py](file:///c:/pawn/train_end_to_end.py#L312-L349):
  - Umstellen auf `torch.amp.GradScaler('cuda', ...)` und `torch.amp.autocast('cuda', ...)`, damit die Warnings verschwinden.

## Verifikation (nach den Änderungen)
- Ein kurzer Trainings-Run (wenige Epochen, wenige Spiele) prüft:
  - Phase 1 läuft nicht mehr automatisch nach Epoche ~7 aus.
  - Phase 2 startet ohne Sprung auf 51.
  - Checkpoint wird geschrieben und vom Dashboard geladen.
- Dashboard-Start prüft:
  - Default-Pfad zeigt auf einen existierenden Checkpoint.
  - Model-Load funktioniert ohne PyTorch-WeightsOnly-Fehler.

## Nebeninfo (PowerShell-Meldung)
- Die Meldung „Installieren Sie die neueste PowerShell … aka.ms/PSWindows“ erscheint bei Windows PowerShell 5.1 immer und lässt sich nicht abschalten. Sie ist unabhängig vom Training.
