# Licensing

## QMProt GSE software

The MIT License below applies to the original software and associated software
documentation in this repository. Third-party material retains its upstream
terms. The QMProt dataset attribution follows the software license.

### MIT License

Copyright (c) 2025–2026 QMProt GSE contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## QMProt dataset attribution

The upstream **QMProt dataset** was created by **Laia Coronas Sala and Parfait
Atchade-Adelomou**. The dataset distribution lists the Creative Commons
Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0).
[PennyLane QMProt collection](https://pennylane.ai/datasets/collection/qmprot) ·
[Dataset license listing](https://pennylane.ai/datasets/pro) ·
[CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)

The pinned geometry files in `framework/smoke/geometries/` select hydrogen and
19 amino acids from the upstream data and repackage the published geometry
fields as per-molecule HDF5 files. `framework/smoke/tiny_geometries/gly/gly.h5`
copies the pinned glycine file. The asparagine source snapshot in
`framework/datasets2/asn/` preserves the upstream JSON. These copies and the
repackaged geometry data are distributed under CC BY-SA 4.0.
[Pinned upstream geometry source](https://github.com/pifparfait/qmprot_strategy/blob/71260c18bb341de12dd3dd23a86234254f3ed48c/QMProt/QMProt.json) ·
[Repository geometry notes](framework/smoke/README.md#geometry-inputs)

Please credit the dataset authors and source, retain the license notice, and
identify changes when sharing the QMProt data. Share adaptations in accordance
with the ShareAlike terms. The dataset paper is *QMProt: A Comprehensive Dataset
of Quantum Properties for Proteins*, Electronics 14(14), 2825 (2025).
[Dataset paper](https://doi.org/10.3390/electronics14142825)
