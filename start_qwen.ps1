# Start the exact model used in the paired experiment.
$ErrorActionPreference = 'Stop'
$modelDir = Join-Path $PSScriptRoot 'models'
$toolsDir = Join-Path $PSScriptRoot 'tools'
New-Item -ItemType Directory -Force $modelDir, $toolsDir | Out-Null
$model = Join-Path $modelDir 'qwen2.5-3b-instruct-q4_k_m.gguf'
$zip = Join-Path $toolsDir 'llama.zip'

function Get-VerifiedFile($Path, $Url, $ExpectedHash) {
    if (Test-Path -LiteralPath $Path) {
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $ExpectedHash) {
            throw "Checksum mismatch. Please inspect the existing file: $Path"
        }
        return
    }
    $part = "$Path.part-$PID"
    & curl.exe --fail --location --retry 3 --output $part $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    if ((Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash -ne $ExpectedHash) {
        throw "Checksum mismatch: $part"
    }
    Move-Item -LiteralPath $part -Destination $Path
}

Get-VerifiedFile $model 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf' '626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'
Get-VerifiedFile $zip 'https://github.com/ggml-org/llama.cpp/releases/download/b7999/llama-b7999-bin-win-vulkan-x64.zip' 'f0c801c2941592b8b31f1e1a9cedfbc81984d206c1411c36e4e67a7eb14467db'
$binaryDir = Join-Path $toolsDir 'llama'
$server = Join-Path $binaryDir 'llama-server.exe'
if (!(Test-Path -LiteralPath $server)) {
    Expand-Archive -LiteralPath $zip -DestinationPath $binaryDir -Force
}
& $server -m $model --host 127.0.0.1 --port 18091
exit $LASTEXITCODE
