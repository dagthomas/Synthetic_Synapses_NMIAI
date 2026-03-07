# Upload all necessary files to RunPod server.
# Usage: .\PRE\upload_to_runpod.ps1 [-Host 103.196.86.219] [-Port 16075]

param(
    [string]$RemoteHost = "103.196.86.219",
    [int]$Port = 16075
)

$ErrorActionPreference = "Stop"

$Dest = "root@$RemoteHost"
$RemoteDir = "/workspace/AINM/PRE"
$Key = "$env:USERPROFILE\.ssh\id_ed25519"
$SshOpts = @("-o", "StrictHostKeyChecking=no", "-p", $Port, "-i", $Key)
$ScpOpts = @("-o", "StrictHostKeyChecking=no", "-P", $Port, "-i", $Key)

$PreDir = "$PSScriptRoot"

Write-Host "=== Upload to RunPod ===" -ForegroundColor Cyan
Write-Host "Host: ${Dest}:${Port}"
Write-Host "Remote dir: $RemoteDir"
Write-Host ""

# 1. Create remote directory structure
Write-Host "--- Creating remote directories ---" -ForegroundColor Yellow
ssh @SshOpts $Dest "mkdir -p $RemoteDir/grocery-bot-gpu/captures $RemoteDir/grocery-bot-gpu/cache $RemoteDir/grocery-bot-b200 $RemoteDir/replay/app/src/lib/components $RemoteDir/replay/app/src/routes"

# 2. grocery-bot-gpu/*.py
Write-Host "`n--- Uploading grocery-bot-gpu/*.py ---" -ForegroundColor Yellow
scp @ScpOpts "$PreDir\grocery-bot-gpu\*.py" "${Dest}:${RemoteDir}/grocery-bot-gpu/"

# 3. grocery-bot-gpu/captures/
Write-Host "`n--- Uploading grocery-bot-gpu/captures/ ---" -ForegroundColor Yellow
$captures = Get-ChildItem "$PreDir\grocery-bot-gpu\captures\*.json" -ErrorAction SilentlyContinue
if ($captures) {
    scp @ScpOpts $captures.FullName "${Dest}:${RemoteDir}/grocery-bot-gpu/captures/"
} else {
    Write-Host "  (no captures found, skipping)"
}

# 4. grocery-bot-gpu/cache/
Write-Host "`n--- Uploading grocery-bot-gpu/cache/ ---" -ForegroundColor Yellow
$cacheFiles = Get-ChildItem "$PreDir\grocery-bot-gpu\cache\*.npz" -ErrorAction SilentlyContinue
if ($cacheFiles) {
    scp @ScpOpts $cacheFiles.FullName "${Dest}:${RemoteDir}/grocery-bot-gpu/cache/"
} else {
    Write-Host "  (no cache files found, skipping)"
}

# 5. grocery-bot-b200/
Write-Host "`n--- Uploading grocery-bot-b200/ ---" -ForegroundColor Yellow
scp @ScpOpts "$PreDir\grocery-bot-b200\*.py" "${Dest}:${RemoteDir}/grocery-bot-b200/"
scp @ScpOpts "$PreDir\grocery-bot-b200\requirements.txt" "${Dest}:${RemoteDir}/grocery-bot-b200/"

# 6. replay/app config files
Write-Host "`n--- Uploading replay/app config ---" -ForegroundColor Yellow
foreach ($f in @("package.json", "svelte.config.js", "vite.config.js", ".env")) {
    $fp = "$PreDir\replay\app\$f"
    if (Test-Path $fp) {
        scp @ScpOpts $fp "${Dest}:${RemoteDir}/replay/app/"
    }
}

# 7. replay/app/src/ (recursive)
Write-Host "`n--- Uploading replay/app/src/ ---" -ForegroundColor Yellow
scp @ScpOpts -r "$PreDir\replay\app\src" "${Dest}:${RemoteDir}/replay/app/"

# 8. docker-compose.yml
Write-Host "`n--- Uploading docker-compose.yml ---" -ForegroundColor Yellow
scp @ScpOpts "$PreDir\replay\docker-compose.yml" "${Dest}:${RemoteDir}/replay/"

# 9. setup_runpod.sh
Write-Host "`n--- Uploading setup_runpod.sh ---" -ForegroundColor Yellow
scp @ScpOpts "$PreDir\setup_runpod.sh" "${Dest}:${RemoteDir}/"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  Upload complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps on RunPod:"
Write-Host "  ssh $($SshOpts -join ' ') $Dest"
Write-Host "  cd $RemoteDir && bash setup_runpod.sh"
