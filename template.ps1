$lifFile = Read-Host "Full path to the .lif file"
$outputFolder = Read-Host "Output root folder"

python .\lif_pipeline.py `
  --input $lifFile `
  --output $outputFolder `
  --scale-bar-um 100