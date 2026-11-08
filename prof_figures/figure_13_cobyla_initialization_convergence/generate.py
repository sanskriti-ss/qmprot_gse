"""Generate paper figure 13 from saved experiment outputs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser, traces

sample = HERE / 'data'
p = parser(sample, 'COBYLA initialization convergence', 'Historical May 2026 data | 16 molecules | COBYLA')
p.set_defaults(output=HERE / "figure.png")
args = p.parse_args()
traces(args, sample, 'COBYLA initialization convergence')
