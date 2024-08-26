$BUILD_DIR = "build"
$ARCH = "x86_64"
$NAME = "windows"

if ($args.Count -gt 0) {
    $ARCH = $args[0]
}
if ($args.Count -gt 1) {
    $NAME = $args[1]
}

if (!(Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Error "GCC not found. Please install MinGW-w64 and add it to your PATH."
    exit 1
}

if (!(Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "Ninja not found. Please install Ninja and add it to your PATH."
    exit 1
}

# Get GCC path
$GCC_PATH = (Get-Command gcc).Source
$GPP_PATH = (Get-Command g++).Source

if (!(Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Force -Path $BUILD_DIR
}

$STATIC_FLAGS = "-static -static-libgcc -static-libstdc++"

cmake . -B"$BUILD_DIR" `
    -DCMAKE_BUILD_TYPE=Release `
    -DENABLE_HOST_OPT=OFF `
    -GNinja `
    -DCMAKE_C_COMPILER="$GCC_PATH" `
    -DCMAKE_CXX_COMPILER="$GPP_PATH" `
    -DCMAKE_CXX_FLAGS="-O2 $STATIC_FLAGS" `
    -DCMAKE_C_FLAGS="-O2 $STATIC_FLAGS" `
    -DCMAKE_EXE_LINKER_FLAGS="$STATIC_FLAGS" `
    -DBUILD_SHARED_LIBS=OFF `
    -DUSE_OPENMP=OFF `
    -DCPACK_SYSTEM_NAME="$NAME" `
    -DGG_NO_PRAGMA=ON

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit $LASTEXITCODE
}

# Build the project
cmake --build "$BUILD_DIR" --target occ

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit $LASTEXITCODE
}

# Package the project
Push-Location $BUILD_DIR
cpack -G TXZ
if ($LASTEXITCODE -ne 0) {
    Write-Error "Packaging failed."
    exit $LASTEXITCODE
}
Pop-Location

Write-Output "Build and packaging completed successfully."
