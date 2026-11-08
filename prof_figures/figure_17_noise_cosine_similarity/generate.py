"""Generate paper figure 17 from saved experiment outputs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser, noise

sample = HERE / 'data/batch_all_results.csv'
p = parser(sample, 'Noise cosine similarity', 'Historical May 2026 data | input IDs require independent verification')
p.set_defaults(output=HERE / "figure.png")
args = p.parse_args()
noise(args, sample)
