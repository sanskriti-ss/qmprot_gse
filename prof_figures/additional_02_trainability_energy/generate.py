"""Plot trainability energy from saved experiment results."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser
from additional_views import trainability_metric

sample = HERE.parent / 'figure_16_trainability_evaluation_cost/data/results_all.csv'
p = parser(sample, 'Trainability energy', 'Historical May 2026 | verify molecule IDs')
p.set_defaults(output=HERE / "figure.png")
p.add_argument("--target", type=int, help="Requested parameter target; required for multiple targets")
args = p.parse_args()

trainability_metric(args, sample, 'energy')
