## Ursache
Die Datei [start.ps1](file:///c:/pawn/start.ps1) ist syntaktisch korrekt, aber sie enthält Unicode-Symbole (ℹ ✔ ⚠ ✖). Windows PowerShell 5.1 parst Skripte ohne UTF-8-BOM oft im aktuellen ANSI-Codepage-Encoding. Dadurch werden diese Zeichen ggf. „kaputt“ dekodiert und können Quotes/Token zerstören, was dann exakt zu den Parser-Fehlern bei `} else {` / `} catch {` führt.

## Fix
1. Generiere `start.ps1` erneut – diesmal **100% ASCII-only** (keine Unicode-Icons), damit es in PowerShell 5.1 unabhängig vom File-Encoding stabil parst.
2. Behalte ausschließlich `Write-Host -ForegroundColor ...` bei (keine Custom-Funktionen).
3. Halte die Struktur minimal/robust:
   - `Get-Command python` Check
   - `$pyVersionInfo = python --version 2>&1 | Out-String` und nur `-match "Python 3"`
   - Jeder `try` hat genau einen `catch`, alle `{}` sind sauber geschlossen.

## Verifikation
Nach dem Überschreiben führe ich einmal einen Parser-Check aus (`powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`) und stelle sicher, dass keine ParserError mehr auftreten.