"""Plot matched ADAPT energy and evaluation-cost trajectories from saved results."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser
from additional_views import combined_adapt

sample = HERE.parent / "additional_04_adapt_operator_cost/data/batch_all_results.csv"
p = parser(sample, "ADAPT energy and cost", "Historical May 2026 | verify molecule IDs")
p.set_defaults(output=HERE / "figure.png")
p.add_argument("--molecule", default="gly", help="Molecule ID to select from the batch CSV")
args = p.parse_args()
combined_adapt(args, sample)
