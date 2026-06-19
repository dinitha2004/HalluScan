# HallKing — local launch (Windows). Forwards all args to run_local.py.
#   .\run_local.ps1            # build frontend + start backend, open http://localhost:8000
#   .\run_local.ps1 --dev      # Vite hot-reload + backend
#   .\run_local.ps1 --no-build # reuse existing frontend/dist
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)

$py = "D:/Github Repositories/semantic-entropy-probes/se_probes_env/Scripts/python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py "$PSScriptRoot/run_local.py" @Rest
