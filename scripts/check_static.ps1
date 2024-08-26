param(
    [Parameter(Mandatory=$true)]
    [string]$ExecutablePath
)

function Check-StaticBinary {
    param(
        [string]$ExePath
    )

    if (-not (Test-Path $ExePath)) {
        Write-Error "Executable not found at $ExePath"
        return
    }

    Write-Output "Checking if the binary is static: $ExePath"

    try {
        $dependencies = & objdump -p $ExePath | Select-String "DLL Name"
        
        if ($dependencies) {
            Write-Output "The following dependencies were found:"
            $dependencies | ForEach-Object { Write-Output $_.Line }
            Write-Output "The binary might not be fully static."
        } else {
            Write-Output "No DLL dependencies found. The binary appears to be static."
        }
    }
    catch {
        Write-Error "Error occurred while running objdump: $_"
    }
}

# Run the function with the provided executable path
Check-StaticBinary -ExePath $ExecutablePath
