# Download pinned PMD, Checkstyle, and SpotBugs binaries into this folder.
# Re-run is a no-op when the expected files already exist.
$ErrorActionPreference = "Stop"

$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PmdVersion = "7.28.0"
$CheckstyleVersion = "10.23.1"
$SpotBugsVersion = "4.9.8"

$PmdUrl = "https://github.com/pmd/pmd/releases/download/pmd_releases%2F$PmdVersion/pmd-dist-$PmdVersion-bin.zip"
$CheckstyleUrl = "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-$CheckstyleVersion/checkstyle-$CheckstyleVersion-all.jar"
$SpotBugsUrl = "https://github.com/spotbugs/spotbugs/releases/download/$SpotBugsVersion/spotbugs-$SpotBugsVersion.zip"

function Expand-Zip($ZipPath, $Dest) {
    if (Test-Path $Dest) {
        Remove-Item -Recurse -Force $Dest
    }
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $Dest -Force
}

$pmdBat = Join-Path $ToolsDir "pmd\pmd-bin-$PmdVersion\bin\pmd.bat"
if (-not (Test-Path $pmdBat)) {
    Write-Host "Downloading PMD $PmdVersion..."
    $zip = Join-Path $env:TEMP "pmd-dist-$PmdVersion-bin.zip"
    Invoke-WebRequest -Uri $PmdUrl -OutFile $zip
    Expand-Zip $zip (Join-Path $ToolsDir "pmd")
    Remove-Item $zip -Force
} else {
    Write-Host "PMD $PmdVersion already present."
}

$checkstyleDir = Join-Path $ToolsDir "checkstyle"
$checkstyleJar = Join-Path $checkstyleDir "checkstyle-$CheckstyleVersion-all.jar"
if (-not (Test-Path $checkstyleJar)) {
    Write-Host "Downloading Checkstyle $CheckstyleVersion..."
    New-Item -ItemType Directory -Force -Path $checkstyleDir | Out-Null
    Invoke-WebRequest -Uri $CheckstyleUrl -OutFile $checkstyleJar
} else {
    Write-Host "Checkstyle $CheckstyleVersion already present."
}

$spotbugsJar = Join-Path $ToolsDir "spotbugs\spotbugs-$SpotBugsVersion\lib\spotbugs.jar"
if (-not (Test-Path $spotbugsJar)) {
    Write-Host "Downloading SpotBugs $SpotBugsVersion..."
    $zip = Join-Path $env:TEMP "spotbugs-$SpotBugsVersion.zip"
    Invoke-WebRequest -Uri $SpotBugsUrl -OutFile $zip
    Expand-Zip $zip (Join-Path $ToolsDir "spotbugs")
    Remove-Item $zip -Force
} else {
    Write-Host "SpotBugs $SpotBugsVersion already present."
}

@(
    "pmd=$PmdVersion",
    "checkstyle=$CheckstyleVersion",
    "spotbugs=$SpotBugsVersion"
) | Set-Content -Encoding utf8 (Join-Path $ToolsDir "VERSIONS.txt")

Write-Host "Tools ready under $ToolsDir"
