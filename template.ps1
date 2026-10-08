$lifFile = Read-Host "Full path to the .lif file"
$outputFolder = Read-Host "Output root folder"

$bfconvert = "C:\tools\bftools\bfconvert.bat"

python .\lif_pipeline.py `
  --input $lifFile `
  --output $outputFolder `
  --scale-bar-um 100 `
  --bfconvert $bfconvert