# build_and_run.ps1 — compile, analyze registers, and benchmark
# apply_to_buffer with [ForceUnroll] vs [MaxIters(N)].
#
# Usage:
#   .\build_and_run.ps1
#   .\build_and_run.ps1 -SlangC "C:\path\to\slangc.exe"
#
# Parameters:
#   -SlangC  path to slangc (default: C:/Github/slang/build/windows-vs2022-dev/Debug/bin/slangc.exe)
#   -Nvcc    path to nvcc   (default: searches PATH)
#   -Arch    GPU arch        (default: sm_89)
#   -Runs    benchmark runs  (default: 5)

param(
    [string]$SlangC = "C:/Github/slang/build/windows-vs2022-dev/Debug/bin/slangc.exe",
    [string]$Nvcc = "",
    [string]$Arch = "sm_89",
    [int]$Runs = 5
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Find nvcc if not specified
if (-not $Nvcc) {
    $Nvcc = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
    if (-not $Nvcc) {
        Write-Error "nvcc not found in PATH. Set -Nvcc parameter or add CUDA toolkit to PATH."
        exit 1
    }
}

$Src = Join-Path $ScriptDir "loop_bench.slang"
$Bench = Join-Path $ScriptDir "bench_launcher.cu"
$OutDir = Join-Path $env:TEMP "loop-strategy-bench"

$NvccFlags = @(
    "-arch=$Arch",
    "--use_fast_math",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== Configuration ==="
Write-Host "  slangc : $SlangC"
Write-Host "  nvcc   : $Nvcc"
Write-Host "  arch   : $Arch"
Write-Host "  runs   : $Runs"
Write-Host ""

# --- Phase 1: Compile both Slang variants ---
Write-Host "=== Phase 1: Slang compilation ==="

Write-Host -NoNewline "  [ForceUnroll]... "
& $SlangC $Src -O3 -target cuda -o "$OutDir/unroll.cu"
if ($LASTEXITCODE -ne 0) { Write-Error "slangc failed (ForceUnroll)"; exit 1 }
Write-Host "done"

Write-Host -NoNewline "  [MaxIters]...    "
& $SlangC $Src -O3 -target cuda -DUSE_MAXITERS -o "$OutDir/maxiters.cu"
if ($LASTEXITCODE -ne 0) { Write-Error "slangc failed (MaxIters)"; exit 1 }
Write-Host "done"
Write-Host ""

# --- Phase 2: Register analysis ---
Write-Host "=== Phase 2: Register counts (backward kernels) ==="

Write-Host ""
Write-Host "--- [ForceUnroll] ---"
& $Nvcc @NvccFlags --ptxas-options=-v -cubin "$OutDir/unroll.cu" -o "$OutDir/unroll.cubin" 2>&1 |
    Select-String "fused_apply.*_bwd_diff" -Context 0,2

Write-Host ""
Write-Host "--- [MaxIters] ---"
& $Nvcc @NvccFlags --ptxas-options=-v -cubin "$OutDir/maxiters.cu" -o "$OutDir/maxiters.cubin" 2>&1 |
    Select-String "fused_apply.*_bwd_diff" -Context 0,2
Write-Host ""

# --- Phase 3: Build benchmark launchers ---
Write-Host "=== Phase 3: Building benchmark launchers ==="

Write-Host -NoNewline "  [ForceUnroll]... "
& $Nvcc @NvccFlags "-DBENCH_CUDA_PATH=$OutDir/unroll.cu" $Bench -o "$OutDir/bench_unroll.exe" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Error "nvcc failed (ForceUnroll launcher)"; exit 1 }
Write-Host "done"

Write-Host -NoNewline "  [MaxIters]...    "
& $Nvcc @NvccFlags "-DBENCH_CUDA_PATH=$OutDir/maxiters.cu" $Bench -o "$OutDir/bench_maxiters.exe" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Error "nvcc failed (MaxIters launcher)"; exit 1 }
Write-Host "done"
Write-Host ""

# --- Phase 4: Run benchmarks ---
Write-Host "=== Phase 4: Runtime benchmark ($Runs runs, 1000 iters each) ==="
Write-Host ""

for ($run = 1; $run -le $Runs; $run++) {
    Write-Host "--- Run $run ---"
    Write-Host "[ForceUnroll]:"
    & "$OutDir/bench_unroll.exe"
    Write-Host "[MaxIters]:"
    & "$OutDir/bench_maxiters.exe"
    Write-Host ""
}
