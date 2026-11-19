param([switch]$OpenBrowser)

$ErrorActionPreference = 'Stop'
$dashboardRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dashboardPython = Join-Path $dashboardRoot '.venv\Scripts\python.exe'
$dashboardServer = Join-Path $PSScriptRoot 'server.py'
$dashboardUrl = 'http://127.0.0.1:8000'

function Get-DashboardHealth {
    try {
        $dashboardResponse = Invoke-WebRequest -Uri "$dashboardUrl/api/health" -UseBasicParsing -TimeoutSec 2
        $dashboardHealth = $dashboardResponse.Content | ConvertFrom-Json
        if ($dashboardHealth.service -eq 'media_collector_dashboard' -and $dashboardHealth.status -eq 'ok') {
            return $dashboardHealth
        }
    } catch {
        return $null
    }
    return $null
}

try {
    $dashboardHealth = Get-DashboardHealth
    if (-not $dashboardHealth) {
        if (-not (Test-Path -LiteralPath $dashboardPython -PathType Leaf)) {
            throw 'The project virtual environment is missing. See tools\dashboard\README.md for setup.'
        }
        $dashboardLogs = Join-Path $dashboardRoot '.test_tmp\dashboard'
        New-Item -ItemType Directory -Path $dashboardLogs -Force | Out-Null
        $dashboardStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $dashboardStdout = Join-Path $dashboardLogs "$dashboardStamp.stdout.log"
        $dashboardStderr = Join-Path $dashboardLogs "$dashboardStamp.stderr.log"
        $env:PYTHONIOENCODING = 'utf-8'
        Start-Process -FilePath $dashboardPython -ArgumentList @('-u', ('"{0}"' -f $dashboardServer)) `
            -WorkingDirectory $dashboardRoot -WindowStyle Hidden `
            -RedirectStandardOutput $dashboardStdout -RedirectStandardError $dashboardStderr | Out-Null

        $dashboardDeadline = (Get-Date).AddSeconds(25)
        do {
            Start-Sleep -Milliseconds 400
            $dashboardHealth = Get-DashboardHealth
        } until ($dashboardHealth -or (Get-Date) -ge $dashboardDeadline)

        if (-not $dashboardHealth) {
            if (Test-Path -LiteralPath $dashboardStderr) {
                Get-Content -LiteralPath $dashboardStderr -Tail 15 | Write-Host
            }
            throw "Dashboard did not start. Check $dashboardStderr"
        }
    }
    Write-Output "Dashboard running at $dashboardUrl (PID $($dashboardHealth.process_id))."
    if ($OpenBrowser) {
        # Opening the dashboard is the launcher's interactive user interface.
        Start-Process -FilePath $dashboardUrl
    }
} catch {
    Write-Error $_
    exit 1
}
