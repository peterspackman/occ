$ARCH = "x86_64"
$NAME = "windows-mingw"

if ($args.Count -gt 0) {
    $ARCH = $args[0]
}
if ($args.Count -gt 1) {
    $NAME = $args[1]
}

# Choose the configure preset from the requested name. "windows-mingw" maps to
# the default preset; any other name is used directly as a preset name.
$PRESET = $NAME

# Must match the binaryDir of the configure preset (see CMakePresets.json base).
$BUILD_DIR = "build/$PRESET"

# MSVC-family presets compile resources with rc.exe through CMake's cmcldeps.exe
# helper. MSYS2's CMake does not ship cmcldeps.exe, which produces a broken RC
# rule ("fatal error RC1107"), so prefer the Visual Studio CMake for those.
function Get-OccCMake {
    param([string]$Preset)
    if ($Preset -notmatch 'msvc') {
        return "cmake"
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -property installationPath 2>$null
        if ($vs) {
            $candidate = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
            if (Test-Path $candidate) { return $candidate }
        }
    }
    $fallback = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path $fallback) { return $fallback }
    Write-Warning "Visual Studio CMake (cmcldeps.exe) not found; falling back to 'cmake' on PATH."
    return "cmake"
}

$CMAKE = Get-OccCMake -Preset $PRESET
$CPACK = if ($CMAKE -eq "cmake") { "cpack" } else { Join-Path (Split-Path $CMAKE) "cpack.exe" }

if (!(Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "Ninja not found. Please install Ninja and add it to your PATH."
    exit 1
}
if ($PRESET -match "mingw" -and !(Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Error "GCC not found. Please install MinGW-w64 and add it to your PATH."
    exit 1
}

# Configure using the chosen preset (sets compiler, flags, build dir, ...)
& $CMAKE --preset "$PRESET"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit $LASTEXITCODE
}

# Build the project
& $CMAKE --build "$BUILD_DIR" --target occ

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit $LASTEXITCODE
}

# Package the project
Push-Location $BUILD_DIR
& $CPACK -G TXZ
if ($LASTEXITCODE -ne 0) {
    Write-Error "Packaging failed."
    exit $LASTEXITCODE
}
Pop-Location

Write-Output "Build and packaging completed successfully."
