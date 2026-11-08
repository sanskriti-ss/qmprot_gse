# ABOUTME: Reads Figure 4 and Figure 6 directly from the October QMProt manuscript.
# ABOUTME: Measures a conservative plotted trial-energy bound and the relative heights of methionine's endpoint bars.
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from pypdf import PdfReader


def measure(path):
    if sha256(path.read_bytes()).hexdigest() != "8c8c1a825b1755c8a8119567c3a4c8c412f174190395ad52dc752c2c4045194b":
        raise ValueError("Expected the October manuscript with the calibrated plot axes")
    pdf = PdfReader(path)
    figure = np.asarray(pdf.pages[8].images[0].image.convert("RGB"), dtype=int)
    bars = np.asarray(pdf.pages[9].images[1].image.convert("RGB"))
    assert figure.shape == (750, 1200, 3) and bars.shape == (1811, 3401, 3)
    # Tick centres y=64,627 correspond to −786.4,−788.0 Ha in Figure 4.
    def energy(y):
        return -786.4 - (y-64)*1.6/563

    colorful = np.ptp(figure, axis=2) > 1
    assert not colorful[440:500, 186:1136].any()  # Neutral gap above the legend.
    bound = np.inf
    for x in range(189, 1136):
        rows = np.where(colorful[55:501, x-2:x+3].any(axis=1))[0] + 55
        if rows.size:
            lower = energy(rows.max() + 10)
            upper = energy(rows.min() - 10)
            bound = min(bound, (3*lower - upper)/2)
    # In Figure 6, grid lines for 0.05 and 0.35 Ha are at x=834 and x=3299.
    spacing = (3299 - 834)/6
    zero = 834 - spacing
    deltas = {}
    for label, y, color in (("random_uniform", 1522, (68, 125, 182)),
                            ("small_random", 1536, (220, 132, 70)),
                            ("near_identity", 1550, (94, 164, 82))):
        pixels = np.where(np.all(bars[y, :1000] == color, axis=1))[0]
        deltas[label] = float((pixels.max() + 0.5 - zero)*0.05/spacing) if pixels.size else 0.
    return {"figure4_conditional_lower_bound_ha": float(bound),
            "figure6_approximate_mean_ha_above_best": deltas}


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1])), indent=2))
