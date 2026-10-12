[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [switch]$Offline,
    [switch]$FunctionsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Sha256 {
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}

function Write-AtomicJson {
    param([string]$Path, [object]$Value)
    $temporary = $Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding($false)))
    if (Test-Path -LiteralPath $Path) { [IO.File]::Replace($temporary, $Path, $null) }
    else { [IO.File]::Move($temporary, $Path) }
}

function Assert-LocalPath {
    param([string]$Parent, [string]$Path)
    $base = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) { throw "Path escapes the setup folder: $Path" }
    # Do not follow existing junctions/symlinks into another installation.
    $probe = $full
    while ($probe.Length -ge $base.TrimEnd('\').Length) {
        if (Test-Path -LiteralPath $probe) {
            $item = Get-Item -LiteralPath $probe -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point is not supported: $probe" }
        }
        $probe = [IO.Path]::GetDirectoryName($probe)
        if ([string]::IsNullOrEmpty($probe)) { break }
    }
    return $full
}

function Assert-Asset {
    param([string]$Path, [object]$Asset, [object]$Manifest)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Asset is missing: $($Asset.name)" }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Asset.size) { throw "Asset size mismatch: $($Asset.name)" }
    $actual = Get-Sha256 $Path
    if ($actual -ne $Asset.sha256) { throw "Asset SHA-256 mismatch: $($Asset.name)" }
    $signature = Get-AuthenticodeSignature -FilePath $Path
    if ($signature.Status -ne 'Valid' -or $null -eq $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -ne $Manifest.signer_subject -or
        $signature.SignerCertificate.Thumbprint -ne $Manifest.signer_thumbprint) {
        throw "Python Software Foundation signature validation failed: $($Asset.name) ($($signature.Status))"
    }
}

function Receive-PythonAsset {
    param([string]$Cache, [object]$Asset, [object]$Manifest, [bool]$NoNetwork)
    $destination = Assert-LocalPath $Cache (Join-Path $Cache $Asset.name)
    if (Test-Path -LiteralPath $destination) {
        # Fail visibly for a corrupt finished cache; never replace it silently.
        Assert-Asset $destination $Asset $Manifest
        return $destination
    }
    if ($Asset.url -notmatch '^https://www\.python\.org/ftp/python/3\.13\.15/amd64/[a-z]+\.msi$') { throw 'Unexpected Python download URL' }
    $partial = Assert-LocalPath $Cache ($destination + '.part')
    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -eq [long]$Asset.size) {
        Assert-Asset $partial $Asset $Manifest
        [IO.File]::Move($partial, $destination)
        return $destination
    }
    if ($NoNetwork) { throw "Offline cache is incomplete: $($Asset.name)" }
    Write-Host ("Python: downloading " + $Asset.name)
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (-not (Test-Path -LiteralPath $curl)) { throw 'Windows curl.exe is required.' }
    $arguments = @('--fail', '--location', '--retry', '3', '--connect-timeout', '30', '--proto', '=https', '--proto-redir', '=https', '--output', $partial)
    if (Test-Path -LiteralPath $partial) { $arguments += @('--continue-at', '-') }
    & $curl @arguments $Asset.url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($Asset.name), curl exit $LASTEXITCODE. Partial data was retained." }
    Assert-Asset $partial $Asset $Manifest
    [IO.File]::Move($partial, $destination)
    return $destination
}

function Initialize-MsiReader {
    if ('SimpleDrawingMsiReader' -as [type]) { return }
    # Read-only MSI database APIs. No installation/custom-action API is called.
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
public static class SimpleDrawingMsiReader {
    [DllImport("msi.dll", CharSet=CharSet.Unicode)] static extern uint MsiOpenDatabase(string path, IntPtr mode, out uint db);
    [DllImport("msi.dll", CharSet=CharSet.Unicode)] static extern uint MsiDatabaseOpenView(uint db, string query, out uint view);
    [DllImport("msi.dll")] static extern uint MsiViewExecute(uint view, uint record);
    [DllImport("msi.dll")] static extern uint MsiViewFetch(uint view, out uint record);
    [DllImport("msi.dll")] static extern uint MsiRecordGetFieldCount(uint record);
    [DllImport("msi.dll", CharSet=CharSet.Unicode)] static extern uint MsiRecordGetString(uint record, uint field, StringBuilder data, ref uint length);
    [DllImport("msi.dll")] static extern uint MsiRecordReadStream(uint record, uint field, byte[] data, ref uint length);
    [DllImport("msi.dll")] static extern uint MsiCloseHandle(uint handle);
    static void Check(uint status) { if (status != 0) throw new IOException("MSI read failed: " + status); }
    static string Value(uint record, uint field) {
        uint length = 0; uint status = MsiRecordGetString(record, field, null, ref length);
        if (status != 0 && status != 234) Check(status);
        var data = new StringBuilder((int)length + 1); length++;
        Check(MsiRecordGetString(record, field, data, ref length)); return data.ToString();
    }
    public static string[][] Query(string path, string sql) {
        uint db=0, view=0, record=0;
        try {
            Check(MsiOpenDatabase(path, IntPtr.Zero, out db));
            Check(MsiDatabaseOpenView(db, sql, out view)); Check(MsiViewExecute(view, 0));
            var rows = new List<string[]>(); uint status;
            while ((status=MsiViewFetch(view, out record)) == 0) {
                uint count = MsiRecordGetFieldCount(record); var row=new string[count];
                for (uint i=1; i<=count; i++) row[i-1]=Value(record,i);
                rows.Add(row); MsiCloseHandle(record); record=0;
            }
            if (status != 259) Check(status); return rows.ToArray();
        } finally { if(record!=0) MsiCloseHandle(record); if(view!=0) MsiCloseHandle(view); if(db!=0) MsiCloseHandle(db); }
    }
    public static void ExtractStream(string path, string name, string output) {
        uint db=0,view=0,record=0;
        try {
            if (name.IndexOf('\'') >= 0) throw new IOException("Invalid MSI stream name");
            Check(MsiOpenDatabase(path,IntPtr.Zero,out db));
            Check(MsiDatabaseOpenView(db,"SELECT `Data` FROM `_Streams` WHERE `Name`='"+name+"'",out view));
            Check(MsiViewExecute(view,0)); Check(MsiViewFetch(view,out record));
            using (var file = new FileStream(output,FileMode.CreateNew,FileAccess.Write)) {
                var buffer = new byte[65536]; uint length;
                do { length=(uint)buffer.Length; Check(MsiRecordReadStream(record,1,buffer,ref length)); file.Write(buffer,0,(int)length); } while(length != 0);
            }
        } finally { if(record!=0) MsiCloseHandle(record); if(view!=0) MsiCloseHandle(view); if(db!=0) MsiCloseHandle(db); }
    }
}
'@
}

function Get-MsiTargetDirectory {
    param([string]$Id, [hashtable]$Directories, [hashtable]$Resolved, [string[]]$Visiting = @())
    if ($Resolved.ContainsKey($Id)) { return $Resolved[$Id] }
    if ($Id -in $Visiting -or -not $Directories.ContainsKey($Id)) { throw "Invalid MSI directory: $Id" }
    $row = $Directories[$Id]
    $parent = Get-MsiTargetDirectory $row[1] $Directories $Resolved ($Visiting + $Id)
    $name = ($row[2] -split ':', 2)[0]
    $name = ($name -split '\|')[-1]
    if ($name -eq '.') { $result = $parent }
    else {
        if ($name -match '[\\/:]' -or $name -in @('', '..')) { throw 'Invalid MSI directory name' }
        $result = if ($parent) { Join-Path $parent $name } else { $name }
    }
    $Resolved[$Id] = $result
    return $result
}

function Expand-PythonMsi {
    param([string]$Msi, [string]$Work, [string]$Runtime)
    Initialize-MsiReader
    New-Item -ItemType Directory -Path $Work -Force | Out-Null
    $expanded = Join-Path $Work 'expanded'
    New-Item -ItemType Directory -Path $expanded -Force | Out-Null
    foreach ($media in [SimpleDrawingMsiReader]::Query($Msi, 'SELECT Cabinet FROM Media')) {
        $cabinet = $media[0]
        if (-not $cabinet.StartsWith('#')) { throw 'Only embedded official MSI cabinets are supported.' }
        $cabinetName = $cabinet.Substring(1)
        $cabinetPath = Assert-LocalPath $Work (Join-Path $Work $cabinetName)
        [SimpleDrawingMsiReader]::ExtractStream($Msi, $cabinetName, $cabinetPath)
        & (Join-Path $env:SystemRoot 'System32\expand.exe') '-F:*' $cabinetPath $expanded | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "CAB expansion failed: $Msi" }
    }
    $directories = @{}
    foreach ($row in [SimpleDrawingMsiReader]::Query($Msi, 'SELECT Directory,Directory_Parent,DefaultDir FROM Directory')) { $directories[$row[0]] = $row }
    $resolved = @{ 'TARGETDIR' = ''; 'InstallDirectory' = '' }
    $components = @{}
    foreach ($row in [SimpleDrawingMsiReader]::Query($Msi, 'SELECT Component,Directory_ FROM Component')) { $components[$row[0]] = $row[1] }
    $count = 0
    foreach ($file in [SimpleDrawingMsiReader]::Query($Msi, 'SELECT File,Component_,FileName,FileSize FROM File')) {
        $source = Assert-LocalPath $expanded (Join-Path $expanded $file[0])
        if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or (Get-Item -LiteralPath $source).Length -ne [long]$file[3]) { throw "MSI file payload mismatch: $($file[0])" }
        $directory = Get-MsiTargetDirectory $components[$file[1]] $directories $resolved
        $filename = ($file[2] -split '\|')[-1]
        if ($filename -match '[\\/:]' -or $filename -in @('', '.', '..')) { throw 'Invalid MSI filename' }
        $relative = if ($directory) { Join-Path $directory $filename } else { $filename }
        $destination = Assert-LocalPath $Runtime (Join-Path $Runtime $relative)
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
        if (Test-Path -LiteralPath $destination) {
            if ((Get-Sha256 $source) -ne (Get-Sha256 $destination)) { throw "MSI payload file collision: $relative" }
        } else { Copy-Item -LiteralPath $source -Destination $destination }
        $count++
    }
    return $count
}

function Test-PythonRuntime {
    param([string]$Runtime, [string]$ExpectedVersion)
    $python = Join-Path $Runtime 'python.exe'
    $probe = @'
import ensurepip, json, pathlib, ssl, struct, sys, tkinter
root = pathlib.Path(sys.executable).resolve().parent
assert sys.version.split()[0] == sys.argv[1], sys.version
assert struct.calcsize('P') == 8, '64-bit Python is required'
assert pathlib.Path(sys.prefix).resolve() == root, (sys.prefix, root)
assert sys.flags.isolated and sys.flags.ignore_environment
tcl = tkinter.Tcl()
assert tcl.eval('info library').lower().startswith(str(root).lower()), tcl.eval('info library')
window = tkinter.Tk()
window.withdraw()
window.update_idletasks()
tk_version = window.tk.eval('package require Tk')
assert window.tk.eval('set tk_library').lower().startswith(str(root).replace('\\', '/').lower()), window.tk.eval('set tk_library')
window.destroy()
import pip
print(json.dumps({'version': sys.version.split()[0], 'architecture': 64, 'prefix_relative': 'runtime', 'tcl': tcl.eval('info patchlevel'), 'tk': tk_version, 'ensurepip': ensurepip.version(), 'pip': pip.__version__, 'isolated': bool(sys.flags.isolated)}, ensure_ascii=True))
'@
    $output = & $python -I -B -X utf8 -c $probe $ExpectedVersion
    if ($LASTEXITCODE -ne 0) { throw 'Python/Tcl/Tk/pip runtime verification failed.' }
    return ($output | ConvertFrom-Json)
}

function Invoke-PythonSetup {
    param([string]$SetupRoot, [string]$ManifestFile, [bool]$NoNetwork)
    $setupFull = [IO.Path]::GetFullPath($SetupRoot).TrimEnd('\')
    if ($setupFull -eq [IO.Path]::GetPathRoot($setupFull).TrimEnd('\')) { throw 'The setup root must be a dedicated folder.' }
    New-Item -ItemType Directory -Path $setupFull -Force | Out-Null
    $manifest = Get-Content -LiteralPath $ManifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or $manifest.version -ne '3.13.15' -or $manifest.architecture -ne 'amd64') { throw 'Unsupported Python runtime manifest.' }
    $manifestSha = Get-Sha256 $ManifestFile
    $cache = Assert-LocalPath $setupFull (Join-Path $setupFull 'downloads\python')
    $data = Assert-LocalPath $setupFull (Join-Path $setupFull 'data')
    $temporary = Assert-LocalPath $setupFull (Join-Path $setupFull 'temp')
    foreach ($folder in @($cache, $data, $temporary)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
    $recordPath = Join-Path $data 'setup-python.json'
    $pendingPath = Join-Path $data 'setup-python.pending.json'
    $runtime = Assert-LocalPath $setupFull (Join-Path $setupFull 'runtime')
    $lockPath = Assert-LocalPath $setupFull (Join-Path $data 'setup-python.lock')
    $operationPath = Assert-LocalPath $setupFull (Join-Path $data 'operation.lock')
    $lock = $null; $operationLock = $null; $operationLocked = $false
    $oldTemporary = $env:TEMP; $oldTmp = $env:TMP
    try {
        try {
            # Windows byte-range locks interoperate with Python msvcrt.locking/FileLock.
            $operationLock = [IO.File]::Open($operationPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite)
            $operationLock.Lock(0, 1)
            $operationLocked = $true
        } catch { throw 'Another operation is using this folder.' }
        try { $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
        catch { throw 'Another Python setup is using this folder.' }
        $env:TEMP = $temporary; $env:TMP = $temporary
        if (Test-Path -LiteralPath $runtime) {
            $existingRecord = if (Test-Path -LiteralPath $recordPath) { $recordPath } elseif (Test-Path -LiteralPath $pendingPath) { $pendingPath } else { $null }
            if (-not $existingRecord) { throw 'An unregistered runtime already exists. It was preserved; use a new setup folder.' }
            $record = Get-Content -LiteralPath $existingRecord -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($record.manifest_sha256 -ne $manifestSha) { throw 'The existing Python runtime belongs to another manifest. It was preserved.' }
            foreach ($file in $record.files) {
                $path = Assert-LocalPath $runtime (Join-Path $runtime $file.path)
                if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Sha256 $path) -ne $file.sha256) { throw "Python runtime integrity mismatch: $($file.path)" }
            }
            $check = Test-PythonRuntime $runtime $manifest.version
            if ($existingRecord -eq $pendingPath) { Write-AtomicJson $recordPath $record }
            Write-Host ('Python: verified existing runtime ' + $check.version)
            return
        }
        $assets = @()
        foreach ($asset in $manifest.assets) { $assets += Receive-PythonAsset $cache $asset $manifest $NoNetwork }
        $build = Assert-LocalPath $setupFull (Join-Path $setupFull ('runtime-staging-' + [Guid]::NewGuid().ToString('N')))
        $buildRuntime = Join-Path $build 'runtime'
        New-Item -ItemType Directory -Path $buildRuntime -Force | Out-Null
        Write-Host 'Python: extracting official MSI payloads (no OS installation)'
        $counts = @{}
        foreach ($msi in $assets) {
            $name = [IO.Path]::GetFileNameWithoutExtension($msi)
            $counts[$name] = Expand-PythonMsi $msi (Join-Path $build $name) $buildRuntime
        }
        $python = Join-Path $buildRuntime 'python.exe'
        & $python -I -B -X utf8 -m ensurepip --upgrade --default-pip
        if ($LASTEXITCODE -ne 0) { throw 'Bundled ensurepip failed.' }
        # The launcher executables contain an absolute build path. Supported commands
        # use python -m pip, which remains valid after moving this folder.
        foreach ($launcher in @('pip.exe', 'pip3.exe', 'pip3.13.exe')) {
            $launcherPath = Assert-LocalPath $buildRuntime (Join-Path $buildRuntime ('Scripts\' + $launcher))
            if (Test-Path -LiteralPath $launcherPath -PathType Leaf) { Remove-Item -LiteralPath $launcherPath }
        }
        $check = Test-PythonRuntime $buildRuntime $manifest.version
        # Record immutable MSI payloads, excluding generated pip launchers with staging paths.
        $files = @(Get-ChildItem -LiteralPath $buildRuntime -File -Recurse | Where-Object {
            $relative = $_.FullName.Substring($buildRuntime.Length + 1)
            $relative -notlike 'Scripts\*' -and $relative -notlike 'Lib\site-packages\*'
        } | ForEach-Object { @{ path = $_.FullName.Substring($buildRuntime.Length + 1); sha256 = Get-Sha256 $_.FullName } })
        $record = @{ schema_version=1; manifest_sha256=$manifestSha; version=$manifest.version; architecture=$manifest.architecture; created_utc=[DateTime]::UtcNow.ToString('o'); files=$files; payload_counts=$counts; probe=$check }
        Write-AtomicJson $pendingPath $record
        # Only a completely verified runtime becomes visible to the application.
        [IO.Directory]::Move($buildRuntime, $runtime)
        Test-PythonRuntime $runtime $manifest.version | Out-Null
        Write-AtomicJson $recordPath $record
        Write-Host ('Python: ready ' + $check.version + ' (Tcl/Tk ' + $check.tcl + ', pip ' + $check.pip + ')')
    } finally {
        $env:TEMP=$oldTemporary; $env:TMP=$oldTmp
        if ($null -ne $lock) { $lock.Dispose() }
        if ($null -ne $operationLock) {
            if ($operationLocked) { $operationLock.Unlock(0, 1) }
            $operationLock.Dispose()
        }
    }
}

if (-not $FunctionsOnly) { Invoke-PythonSetup $Root $ManifestPath ([bool]$Offline) }
