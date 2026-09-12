EXPECTED OUTPUT -- CLAIM 2

The run writes artifact/output/functional-stix.json and
artifact/output/functional-malont.json, validating each in turn.

ACCEPTANCE CRITERION -- ./claims/claim2/run.sh must print, in order:

    PASS: valid stix ANCHOR output
    PASS: valid malont ANCHOR output
    PASS: schema switch validated for stix and malont

Each per-schema validation applies the same structural contract described in
claims/claim1/expected/README.txt, with one difference that is the actual point
of this claim: the "ontology" field of each output must equal the schema
selected on the command line ("stix" and "malont" respectively), and the entity
classes and predicate URIs must come from that schema's vocabulary rather than
UCO's.

What demonstrates the claim is that a single unmodified executable and a single
extraction path produced ontology-aligned output against three different
OWL/SHACL schemas, selected only by --schema. Combined with claim1 (UCO), this
covers all three packaged ontologies.

NOT AN ACCEPTANCE CRITERION

As in claim1, exact names, counts, ordering, and URI choices are
nondeterministic and are not compared against a reference file. The per-schema
accuracy numbers in the paper are outside the scope of this claim.

MODEL CAPABILITY IS A PREREQUISITE

As in claim1: the mapper depends on multi-step tool calling, so a model weak at
it yields a poor or empty graph for one or both schemas. check_endpoint.py warns
about this before each run, and claims/claim3 isolates package integrity from
model capability.

TROUBLESHOOTING

See claims/claim1/expected/README.txt. Run claim1 first so the embedding model
is already cached; otherwise the first schema here absorbs the download time.
