param([string]$Root = (Split-Path -Parent $PSScriptRoot), [switch]$Offline, [switch]$SkipModel)
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath($Root)
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'source\run.py'))) { throw 'アプリのソースがありません。ZIPをフォルダーごと展開してください。' }
$taskLogDirectory = Join-Path $taskRoot 'logs'
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null
$taskLog = Join-Path $taskLogDirectory ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff') + '.log')
$taskTranscriptStarted = $false
try {
    Start-Transcript -LiteralPath $taskLog -ErrorAction Stop | Out-Null
    $taskTranscriptStarted = $true
    Write-Host '公式配布元から固定版の構成部品を取得し、このフォルダー内に配置します。'
    Write-Host '取得する部品にはそれぞれの利用条件が適用されます。THIRD_PARTY_NOTICES.mdをご確認ください。'
    Write-Host '初回はインターネット接続と25GiB以上の空き容量、対応するNVIDIA GPU・ドライバーが必要です。'
    & (Join-Path $taskRoot 'scripts\setup_python.ps1') -Root $taskRoot -ManifestPath (Join-Path $taskRoot 'assets\python-runtime.json') -Offline:$Offline
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) { throw 'Python構築に失敗しました。' }
    $taskPython = Join-Path $taskRoot 'runtime\python.exe'
    $taskArgs = @('-I','-B','-X','utf8',(Join-Path $taskRoot 'scripts\bootstrap.py'),'--root',$taskRoot)
    if ($Offline) { $taskArgs += '--offline' }
    if ($SkipModel) { $taskArgs += '--skip-model' }
    $taskSavedErrorPreference = $ErrorActionPreference
    try {
        # Send native output through the host so Transcript also retains it.
        $ErrorActionPreference = 'Continue'
        & $taskPython @taskArgs 2>&1 | ForEach-Object { Write-Host $_ }
        $taskResult = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $taskSavedErrorPreference
    }
    if ($taskResult -ne 0) { throw ('初期設定に失敗しました。終了コード: ' + $taskResult) }
} finally {
    if ($taskTranscriptStarted) { Stop-Transcript | Out-Null }
    Write-Host ('初期設定の記録: ' + $taskLog)
}
