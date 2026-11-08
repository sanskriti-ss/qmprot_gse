"""Generate the paired trainability energy and runtime view from saved CSV data."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser
from additional_views import combined_trainability

sample = HERE.parent / "figure_16_trainability_evaluation_cost/data/results_all.csv"
p = parser(sample, "Paired trainability energy and runtime", "Historical May 2026 | verify molecule IDs")
p.set_defaults(output=HERE / "figure.png")
p.add_argument("--target", type=int, help="Requested parameter target; required when input has multiple targets")
args = p.parse_args()
combined_trainability(args, sample)
