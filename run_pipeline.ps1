# Pipeline lengkap: data -> embedding konten -> eksperimen -> analisis.
# Jalankan: powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
#
# Tiap tahap punya progress bar sendiri. Perkiraan total 2-3 jam di RTX 4050.
# Aman diulang: tahap yang hasilnya sudah ada akan ditimpa, bukan ditumpuk.

$ErrorActionPreference = "Stop"

$PY = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PY)) {
    throw "Python 3.10 64-bit tidak ditemukan di $PY. Python 3.8 32-bit tidak bisa dipakai."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $root "src")
$started = Get-Date

function Step($num, $name, $script, $argList) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host "  TAHAP $num  $name" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    $t0 = Get-Date
    & $PY $script @argList
    if ($LASTEXITCODE -ne 0) { throw "$script gagal (exit $LASTEXITCODE)" }
    $mins = ((Get-Date) - $t0).TotalMinutes
    Write-Host ("  selesai dalam {0:N1} menit" -f $mins) -ForegroundColor Green
}

# Tahap 0 WAJIB lebih dulu. Daftar video berkonten dipakai tahap 1 untuk menyaring
# sebelum user disampel. Kalau dibalik, sebagian besar sequence jadi bolong.
Step "0/6" "Ekstrak teks konten semua video (tanpa encoding, ~10 detik)" `
    "build_content.py" @("--texts-only", "--items-from", "nonexistent")

Step "1/6" "Sampling user dari user-video.json 3,18 GB (~3 menit)" `
    "prepare_sequences.py" @("--n-users", "20000")

Step "2/6" "Encoding konten: multilingual-E5 + PCA (~3 menit)" `
    "build_content.py" @()

Step "3/6" "Smoke test: 1 model, 3 epoch (~1 menit)" `
    "train.py" @("--mode", "delta", "--seed", "42", "--max-epochs", "3", "--rebuild-splits")

Step "4/6" "Eksperimen utama: 4 model x 5 seed (~45 menit)" `
    "run_all.py" @()

Step "5/6" "Sweep delta_max: 8 nilai x 3 seed (~75 menit)" `
    "sweep_delta.py" @("--deltas", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "0.9")

Step "6/6" "Analisis Gambar 4 dan 5 (~5 menit)" `
    "analysis.py" @()

$total = ((Get-Date) - $started).TotalMinutes
Write-Host ""
Write-Host ("SELESAI dalam {0:N1} menit" -f $total) -ForegroundColor Green
Write-Host "Luaran:" -ForegroundColor Green
Write-Host "  results\table2.md                      tabel utama"
Write-Host "  results\delta_sweep.csv                sweep delta_max"
Write-Host "  results\fig4_by_input_warm_fraction.csv"
Write-Host "  results\fig5_by_gt_frequency.csv"
Write-Host ""
Write-Host "Setelah melihat delta_sweep.csv, jalankan ulang tabel utama pada nilai terbaik:" -ForegroundColor Yellow
Write-Host "  & `$PY run_all.py --max-delta-norm <nilai> --tag _terbaik"
