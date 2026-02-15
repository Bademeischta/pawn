## Diagnose
Der Crash `ValueError: probabilities do not sum to 1` kommt aus [mcts.py](file:///c:/pawn/mcts.py) in `_select_move()`: dort werden aus Visit-Counts Wahrscheinlichkeiten gebaut. In deinem Log ist der Summenfall typisch **Summe=0** (z.B. wenn alle Child-Visits 0 sind oder numerisch zu 0 werden) oder **NaN/Inf** (bei sehr kleiner Temperatur).

Zusätzlich ist die Backprop aktuell so gebaut, dass **der Leaf-Node nicht mitgezählt wird** (es werden nur die Knoten in `path` geupdatet, der Leaf selbst nicht). Das kann dazu führen, dass Visit-Counts ungewöhnlich lange 0/1 bleiben.

## Änderungen
1. **Backprop fixen:** In `_simulate_batch()` den `leaf_node` explizit updaten (Visit++ und Value addieren), während Virtual-Loss nur bei den `path`-Nodes zurückgenommen wird.
2. **Robuste Move-Selection:** `_select_move()` so umbauen, dass:
   - nur aktuelle `board.legal_moves` berücksichtigt werden
   - bei `sum(visits) <= 0` auf Priors oder Uniform-Fallback gewechselt wird
   - NaN/Inf/negative Werte abgefangen werden
   - am Ende `probs` garantiert normalisiert ist (Summe=1)
3. **Mini-Verifikation:** Einen kurzen Smoke-Test laufen lassen (MCTS auf Startposition, `search()` mehrfach) und sicherstellen, dass kein `np.random.choice(..., p=...)` mehr wirft.

Danach sollte Self-Play nicht mehr an der Stelle abbrechen.