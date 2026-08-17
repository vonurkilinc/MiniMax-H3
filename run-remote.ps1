#Requires -Version 5.1
<#
.SYNOPSIS
  MiniMax-H3 code-only remote runner.
  Pushes a job (config + prompt + inputs) to the Ubuntu server, runs inference
  there, pulls videos back to outputs\ and logs to logs\. The server only keeps
  the model (~/models/minimax-h3) and the python env (~/h3-env); the runtime
  job dir /dev/shm/h3-job is always deleted afterwards, success or failure.

.EXAMPLE
  .\run-remote.ps1                                  # run with default config + example prompt
  .\run-remote.ps1 -PromptFile prompts\my.txt -Config configs\fl2va.json -Inputs inputs\first.png
  .\run-remote.ps1 -Provision                       # one-time: create server env + install deps
  .\run-remote.ps1 -DownloadModel                   # one-time: download model (~134 GiB, resumable)
  .\run-remote.ps1 -DownloadModel -WithRef2VA       # also fetch transformer_ref (needs +62 GiB disk)
  .\run-remote.ps1 -ProvisionStatus                 # peek at server-side provision/download logs
#>
param(
  [string]$PromptFile = "prompts\example.txt",
  [string]$Prompt = "",
  [string]$Config = "configs\default.json",
  [string[]]$Inputs = @(),
  [string]$JobName = "",
  [string]$Server = $env:H3_SERVER,
  [switch]$Provision,
  [switch]$DownloadModel,
  [switch]$WithRef2VA,
  [switch]$ProvisionStatus
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$SshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new")
$JobDir = "/dev/shm/h3-job"
$ProvDir = "/dev/shm/h3-provision"

if ([string]::IsNullOrWhiteSpace($Server)) {
  throw "No GPU server configured. Pass -Server user@host or set H3_SERVER."
}

function Invoke-Ssh([string]$Cmd) {
  & ssh @SshOpts $Server $Cmd
  return $LASTEXITCODE
}

function Push-Job([string]$StagingDir) {
  # NOTE: no tar-through-PowerShell-pipe - PS 5.1 mangles binary native pipes.
  # Windows scp needs forward-slash local paths, otherwise it recreates the
  # whole "C:\..." path as directory names on the remote side.
  $fwd = $StagingDir -replace '\\', '/'
  Invoke-Ssh "rm -rf $JobDir && mkdir -p $JobDir" | Out-Null
  & scp @SshOpts -r "$fwd/in" "$fwd/code" "${Server}:${JobDir}/"
  if ($LASTEXITCODE -ne 0) { throw "push to ${JobDir} failed" }
}

function Push-Provision() {
  Invoke-Ssh "rm -rf $ProvDir && mkdir -p $ProvDir" | Out-Null
  & scp @SshOpts "$Root\src\provision.sh" "${Server}:${ProvDir}/"
  if ($LASTEXITCODE -ne 0) { throw "push of provision.sh failed" }
}

function Pull-Results([string]$Name) {
  # scp -r of the remote out/ dir, then flatten into outputs\<Name>.
  $dest = "$Root\outputs\$Name"
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  & scp @SshOpts -r "${Server}:${JobDir}/out" $dest 2>$null | Out-Null
  if (Test-Path "$dest\out") {
    $items = Get-ChildItem "$dest\out"
    if ($items) { Move-Item "$dest\out\*" $dest -Force }
    Remove-Item -Recurse -Force "$dest\out"
  }
}

# ---------- provisioning modes ----------
if ($ProvisionStatus) {
  Invoke-Ssh "tail -30 $ProvDir/env.log $ProvDir/download.log 2>/dev/null || echo 'no provision logs yet'" | Out-Null
  exit 0
}

if ($Provision -or $DownloadModel) {
  New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null
  Push-Provision
  if ($Provision) {
    Write-Host "== provisioning env on server (streaming) =="
    Invoke-Ssh "bash $ProvDir/provision.sh --env 2>&1 | tee $ProvDir/env.log"
    if ($LASTEXITCODE -ne 0) { throw "env provisioning failed; see output above" }
  }
  if ($DownloadModel) {
    $dlArgs = "--download"; if ($WithRef2VA) { $dlArgs += " --with-ref2va" }
    Write-Host "== starting model download on server (nohup; resumable) =="
    Invoke-Ssh "cd $ProvDir && nohup bash provision.sh $dlArgs > $ProvDir/download.log 2>&1 & echo download started, pid `$!"
    Write-Host "Download runs on the server even if you close this window. Check with: .\run-remote.ps1 -ProvisionStatus"
  }
  exit 0
}

# ---------- job mode ----------
if ($JobName -eq "") { $JobName = "h3-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
New-Item -ItemType Directory -Force -Path "$Root\outputs", "$Root\logs" | Out-Null

$cfgPath = Join-Path $Root $Config
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json

$staging = Join-Path $env:TEMP ("h3-stage-" + [guid]::NewGuid().ToString("n"))
New-Item -ItemType Directory -Force -Path "$staging\in", "$staging\code" | Out-Null
try {
  # code
  Copy-Item "$Root\src\generate.py", "$Root\src\remote_job.sh" "$staging\code\"
  Copy-Item $cfgPath "$staging\in\config.json"

  # prompt
  if ($Prompt -ne "") {
    [IO.File]::WriteAllText("$staging\in\prompt.txt", $Prompt)
    if (-not $cfg.prompt_file) { $cfg | Add-Member -NotePropertyName prompt_file -NotePropertyValue "prompt.txt"; $cfg | ConvertTo-Json | Set-Content "$staging\in\config.json" }
  }
  elseif ($cfg.prompt_file) {
    $src = Join-Path $Root $PromptFile
    if (-not (Test-Path $src)) { throw "prompt file not found: $src" }
    Copy-Item $src (Join-Path "$staging\in" $cfg.prompt_file)
  }

  # inputs named by the config must exist locally and be shipped
  # (deduped by file name: an absolute -Inputs path wins over a config basename)
  $needed = @()
  foreach ($p in @("image", "last_image", "continue_video")) { if ($cfg.$p) { $needed += $cfg.$p } }
  if ($cfg.references) { foreach ($r in $cfg.references) { $needed += $r.path } }
  if ($cfg.loras) { foreach ($l in $cfg.loras) { $needed += $l.path } }
  $shipped = @()
  foreach ($f in ($Inputs + $needed)) {
    if (-not $f) { continue }
    $leaf = Split-Path $f -Leaf
    if ($shipped -contains $leaf) { continue }
    $local = if (Test-Path $f) { $f } else { Join-Path $Root ("inputs\" + $f) }
    if (-not (Test-Path $local)) { throw "input file not found: $f (looked in cwd and $Root\inputs)" }
    Copy-Item $local (Join-Path "$staging\in" $leaf)
    $shipped += $leaf
  }

  Write-Host "== pushing job '$JobName' to $Server =="
  Push-Job $staging

  Write-Host "== running on server (live output below) =="
  Invoke-Ssh "bash $JobDir/code/remote_job.sh"
  $jobRc = $LASTEXITCODE

  Write-Host "== pulling results to outputs\$JobName =="
  Pull-Results $JobName
  $runLog = "$Root\outputs\$JobName\run.log"
  if (Test-Path $runLog) { Copy-Item $runLog "$Root\logs\$JobName.log" }

  if ($jobRc -ne 0) {
    Write-Warning "remote job failed (exit $jobRc)  -  see logs\$JobName.log"
    exit $jobRc
  }
  Write-Host "== done. Video: outputs\$JobName\  -  log: logs\$JobName.log =="
}
finally {
  # guaranteed cleanup of the server-side temp dir, success or failure
  Invoke-Ssh "rm -rf $JobDir" | Out-Null
  Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
}
