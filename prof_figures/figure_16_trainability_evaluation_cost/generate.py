"""Generate paper figure 16 from saved experiment outputs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser, trainability

sample = HERE / 'data/results_all.csv'
p = parser(sample, 'Trainability evaluation cost', 'Historical May 2026 data | target 10 only')
p.set_defaults(output=HERE / "figure.png")
args = p.parse_args()
trainability(args, sample)
