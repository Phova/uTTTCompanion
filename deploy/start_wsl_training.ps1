# Run this on the Windows host that owns the WSL distribution. WMI creates the
# foreground wsl.exe outside an OpenSSH session's inherited process/job tree.
[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu",
    [string]$LinuxUser = "root",
    [string]$Workspace = "/root/uTTT",
    [string]$RunDirectory = "/root/uTTT",
    [string]$Session = "uttt_train",
    [string]$LogPath = "/root/uTTT/logs/train_5090.log",
    [string]$OwnerScript = "",
    [string]$LinuxPython = "/usr/bin/python3",
    [Parameter(Mandatory = $true)]
    [AllowEmptyString()]
    [string[]]$TrainingCommand,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "Run this launcher in Windows PowerShell on the machine hosting WSL."
}
if ($Session -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') {
    throw "Session must contain 1-64 letters, digits, underscores or hyphens."
}
if (-not $OwnerScript) { $OwnerScript = "$($Workspace.TrimEnd('/'))/deploy/wsl_training_owner.py" }
foreach ($LinuxPath in @($Workspace, $RunDirectory, $LogPath, $OwnerScript, $LinuxPython, $TrainingCommand[0])) {
    if (-not $LinuxPath.StartsWith('/') -or $LinuxPath.Contains([char]0)) {
        throw "Workspace, executable and script paths must be absolute Linux paths."
    }
}

function ConvertTo-NativeArgument([string]$Value) {
    if ($Value.Contains([char]0)) { throw "Arguments cannot contain a NUL character." }
    # Some WSL entry points treat a quoted first option ("-d") as a Linux
    # command instead of a WSL option. Leave ordinary tokens unquoted.
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    # CommandLineToArgvW / CRT quoting: double backslashes preceding a quote
    # and at the end of a quoted argument. No cmd.exe or PowerShell eval is used.
    $EscapedValue = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $EscapedValue = [regex]::Replace($EscapedValue, '(\\+)$', '$1$1')
    return '"' + $EscapedValue + '"'
}

$WslExecutable = Join-Path $env:SystemRoot "System32\wsl.exe"
if (-not (Test-Path -LiteralPath $WslExecutable -PathType Leaf)) {
    throw "Cannot locate wsl.exe at $WslExecutable"
}
$WslArguments = @(
    $WslExecutable, "-d", $Distribution, "-u", $LinuxUser, "-e", $LinuxPython,
    $OwnerScript, "run", "--workspace", $Workspace, "--session", $Session,
    "--cwd", $RunDirectory, "--log", $LogPath, "--"
) + $TrainingCommand
$NativeCommandLine = ($WslArguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
if ($NativeCommandLine.Length -gt 30000) { throw "Training command is too long for a Windows process command line." }
if ($DryRun) {
    $NativeCommandLine
    return
}

$Startup = New-CimInstance -ClassName Win32_ProcessStartup -Namespace root/cimv2 -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$Created = Invoke-CimMethod -ClassName Win32_Process -Namespace root/cimv2 -MethodName Create -Arguments @{
    CommandLine = $NativeCommandLine
    ProcessStartupInformation = $Startup
}
if ($Created.ReturnValue -ne 0) {
    throw "Windows refused to create the independent WSL owner (Win32_Process.Create status $($Created.ReturnValue))."
}
Write-Output "Started independent Windows WSL client PID $($Created.ProcessId)."
Write-Output "Linux startup/running state: $Workspace/logs/.wsl_owners/$Session.json"
Write-Output "Owner diagnostics: $Workspace/logs/.wsl_owners/$Session.owner.log"
Write-Output "Creation success does not confirm training startup; check the owner record and log."
