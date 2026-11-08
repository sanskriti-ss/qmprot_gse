"""Generate paper figure 15 from saved experiment outputs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser, finals

sample = HERE / 'data'
p = parser(sample, 'Final energies by initialization', 'Hydrogen pilot only | Bayesian | one trial per strategy')
p.set_defaults(output=HERE / "figure.png")
args = p.parse_args()
finals(args, sample)
