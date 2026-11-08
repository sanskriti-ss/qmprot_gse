"""Plot noise energy comparison from saved experiment results."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import parser
from additional_views import noise_energy

sample = HERE.parent / 'figure_17_noise_cosine_similarity/data/batch_all_results.csv'
p = parser(sample, 'Noise energy comparison', 'Historical May 2026 noise batch | molecule IDs require verification')
p.set_defaults(output=HERE / "figure.png")

p.add_argument("--molecule", help="Molecule ID to select from a batch CSV")
args = p.parse_args()

if args.molecule is None and args.input.resolve() == sample.resolve():
    args.molecule = "gly"
noise_energy(args, sample)
