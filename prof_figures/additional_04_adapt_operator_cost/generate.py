"""Plot adapt cost versus operators from saved experiment results."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser
from additional_views import adapt_cost

sample = HERE.parent / 'additional_04_adapt_operator_cost/data/batch_all_results.csv'
p = parser(sample, 'ADAPT cost versus operators', 'Historical May 2026 | targets 1–3 | verify molecule IDs')
p.set_defaults(output=HERE / "figure.png")
args = p.parse_args()

adapt_cost(args, sample)
