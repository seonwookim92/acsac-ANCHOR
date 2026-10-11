EXPECTED OUTPUT -- CLAIM 3

Deterministic. No LLM endpoint, GPU, credentials, or network access required.

ACCEPTANCE CRITERION -- ./claims/claim3/run.sh must print these lines and exit 0:

    PASS: ACSAC layout present (3 claim directories)
    PASS: no generated files outside ignored runtime directories
    PASS: <N> Python source files parse
    PASS: 149 aligned inputs and 149 records for each schema
    PASS: uco RDF parses (<n> files)
    PASS: stix RDF parses (<n> files)
    PASS: malont RDF parses (<n> files)
    PASS: SHACL scope explicit (uco=<n> shapes; stix=0; malont=0)
    PASS: artifact preflight completed
    PASS: graph construction did not initialize an LLM provider
    PASS: semantic fallbacks allowed and operational fallbacks rejected
    PASS: deterministic functional checks completed

The <N> and <n> counts are printed for information. The file counts are stable
for a given release of this package, but they are not themselves the acceptance
criterion; any nonzero count that clears the checks is acceptable. The exact
required values that ARE enforced, and that will fail the run if violated, are:

  - exactly 149 CTINexus report inputs in artifact/dataset/;
  - exactly 149 author-created records in each of
    artifact/ground_truth/{uco,stix,malont}/;
  - the record identifiers in all three schemas match the input identifiers
    one-to-one, with no missing and no extra records;
  - every ground-truth file is a JSON object;
  - every packaged UCO .ttl and every packaged STIX/MALOnt .owl file parses
    with rdflib;
  - packaged UCO contains SHACL shapes, while the packaged STIX and MALOnt
    sources are explicitly verified as OWL-only;
  - every distributed Python source parses;
  - `main.py --help` exposes --schema and `main.py --graph` renders the
    documented pipeline graph including the mapper_init node without
    initializing an LLM provider or requiring provider credentials; and
  - the output validator accepts a mixture of ontology-specific mappings and
    intentional generic semantic fallbacks, but rejects both all-generic and
    operational-fallback outputs; and
  - the required ACSAC top-level layout is present, with an executable run.sh,
    a claim.txt, and an expected/ directory in every claims/claimN folder; and
  - submission/metadata.toml is present in the package.

Any failure aborts with a single line beginning "ERROR:" that names the failing
file or the failing condition.

TROUBLESHOOTING

  "Python environment not found"
      Run ./install.sh from the package root first.

  "rdflib is not installed"
      The virtual environment is incomplete; re-run ./install.sh.
