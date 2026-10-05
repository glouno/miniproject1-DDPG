"""Regenerate pilot figures and tables from immutable run data.

Invocation from Sorbonne with the locked RL interpreter:
  environment/.venv/bin/python lunarlander_pilot.py \
    --input outputs/pilot-20261005 \
    --output generated/lunarlander-pilot-20261005
Inputs: 12 completed runs + supplemental final calibration. Training seeds 0,1,2.
Aggregation: sampled pairs -> episode -> training run; figures show run-level
means ±1 sample SD across 3 seeds. Complete-only bias curves exclude censored
episodes and are flagged; final comparisons use all long-return diagnostics.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyse import load_pilot,make_figures,write_analysis

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
manifest,runs=load_pilot(args.input)
make_figures(runs,args.output)
rows,effects=write_analysis(runs,args.output)
for row in rows:
    print(row)
print('Figures and analysis written to',args.output)
