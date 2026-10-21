$ErrorActionPreference = "Stop"
$python = "C:/Users/CHARAN SUBHASH/AppData/Local/Python/bin/python.exe"

Write-Output "Running protein_dta.py..."
& $python preprocessing/protein_pretrain/protein_dta.py --root_data_path data --dataset pdb

Write-Output "Running protein_structure.py..."
& $python preprocessing/protein_pretrain/protein_structure.py --root_data_path data --dataset pdb

Write-Output "Running main.py..."
& $python main.py --root_data_path data --dataset pdb --max_epochs 1 --batch_size 16 --num_workers 0
