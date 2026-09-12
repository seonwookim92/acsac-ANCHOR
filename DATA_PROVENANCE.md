# Data and ontology provenance

## CTINexus report text

- Project: https://ctinexus.github.io/
- Repository: https://github.com/peng-gao-lab/ctinexus
- Snapshot: `0c688536d85eae72f6055723492b573b0a1ff865`
- Upstream source: `ctinexus/data/annotation/*.json`, field `text`
- Packaged location: `dataset/ctinexus_*.txt`

The 149 packaged files contain the upstream report text converted from a JSON
field to UTF-8 text files. CTINexus distributes its repository under MIT terms;
original report publishers retain applicable attribution and rights.

## ANCHOR annotations

The class and property labels under `ground_truth/{uco,stix,malont}/` were newly
authored for ANCHOR; they are not CTINexus labels. These annotations are
released under CC BY 4.0. The embedded `text` fields remain CTINexus material
and are not relicensed by the ANCHOR authors.

## Ontologies

- UCO 1.4.0: https://github.com/ucoProject/UCO at
  `8335d32d600f717b7b4f43d74cf40efdf07aa340` (Apache-2.0).
- STIX 2.1 OWL: https://github.com/oasis-open/tac-ontology at
  `d93cdd9b025b0c17a4ee047e29cdd762439409a3` (BSD-3-Clause).
- MALOnt: https://github.com/aiforsec/MALOnt at
  `967ce06c09e262678432f4eba5e844f95bd2cf51` (MIT).

The deterministic preflight verifies the packaged counts, identifier alignment,
JSON syntax, and RDF syntax.

