import os
import sys
import glob
import hashlib
import pickle
import rdflib
import numpy as np
from rdflib.namespace import RDF, RDFS, OWL
from sentence_transformers import SentenceTransformer
import logging

# Prevent noisy logs from interfering with MCP stdout
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from huggingface_hub import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

SHACL = rdflib.Namespace("http://www.w3.org/ns/shacl#")

# Explicit URIs for SHACL terms that clash with Python keywords or aren't in the namespace
SHACL_CLASS             = rdflib.URIRef("http://www.w3.org/ns/shacl#class")
SHACL_OR                = rdflib.URIRef("http://www.w3.org/ns/shacl#or")
SHACL_QUALIFIED_VALUE   = rdflib.URIRef("http://www.w3.org/ns/shacl#qualifiedValueShape")

# Bump this whenever the cache format or schema parsing logic changes
_CACHE_VERSION = "3"

# Keywords that identify composition/grouping properties (generalised from UCO's hasFacet)
_COMPOSITION_PROP_KEYWORDS = {"hasfacet", "hascomponent", "haspart", "hasextension", "hasfeature"}
# Keywords that identify subordinate grouping classes (Facet, Component, Extension…)
_COMPOSITION_CLASS_KEYWORDS = {"facet", "component", "extension"}

# Format labels for load summary
import re as _re
_RE_DATETIME = _re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$")
_RE_INTEGER  = _re.compile(r"^-?\d+$")

_FORMAT_DISPLAY = {
    "ttl":     "TTL",
    "xml":     "OWL/RDF-XML",
    "n3":      "N3",
    "nt":      "NT",
    "trig":    "TriG",
    "json-ld": "JSON-LD",
}

# OWL property-type → normalized string name.
# Defined at module level so it's shared by _build_schema_cache and any future callers.
_OWL_DATA_PROPERTY = rdflib.URIRef("http://www.w3.org/2002/07/owl#DataProperty")
_PROP_TYPE_NAMES = {
    OWL.ObjectProperty:   "ObjectProperty",
    OWL.DatatypeProperty: "DatatypeProperty",
    _OWL_DATA_PROPERTY:   "DatatypeProperty",  # normalize DataProperty → DatatypeProperty
    RDF.Property:         "Property",
}


def _local_name(uri: str) -> str:
    """Returns the local name of a URI (fragment after # or last path segment)."""
    return uri.split('/')[-1].split('#')[-1]


class OntologyEngine:
    """Core ontology management engine with semantic search capabilities."""
    def __init__(self, ontology_path, model_name="all-MiniLM-L6-v2"):
        self.graph = rdflib.Graph()
        self.ontology_path = ontology_path
        self.classes = {}
        self.properties = {}
        self.shapes = {}
        self.instance_base_uri = "http://example.org/entities/"
        self._ontology_files = []  # [(filepath, format), ...] — used for cache key

        emb_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        sys.stderr.write(f"Initializing Semantic Search Model ({model_name}) on {emb_device}...\n")
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            self.model = SentenceTransformer(model_name, device=emb_device)
        self.load_ontology()

    @staticmethod
    def _detect_format(filepath):
        """Detects rdflib parse format from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        return {
            ".ttl":    "ttl",
            ".owl":    "xml",
            ".rdf":    "xml",
            ".n3":     "n3",
            ".nt":     "nt",
            ".trig":   "trig",
            ".jsonld": "json-ld",
        }.get(ext, None)

    def load_ontology(self):
        """Loads all supported ontology files and parses the schema."""
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            files = []

            if os.path.isdir(self.ontology_path):
                for pattern, fmt in [
                    ("**/*.ttl",    "ttl"),
                    ("**/*.owl",    "xml"),
                    ("**/*.rdf",    "xml"),
                    ("**/*.n3",     "n3"),
                    ("**/*.nt",     "nt"),
                    ("**/*.trig",   "trig"),
                    ("**/*.jsonld", "json-ld"),
                ]:
                    for f in glob.glob(os.path.join(self.ontology_path, pattern), recursive=True):
                        files.append((f, fmt))
            elif os.path.isfile(self.ontology_path):
                fmt = self._detect_format(self.ontology_path)
                if fmt:
                    files = [(self.ontology_path, fmt)]
                else:
                    sys.stderr.write(f"Warning: unrecognized file extension for {self.ontology_path}, trying ttl\n")
                    files = [(self.ontology_path, "ttl")]

            self._ontology_files = files

            counts = {"errors": 0}
            for f, fmt in files:
                try:
                    self.graph.parse(f, format=fmt)
                    counts[fmt] = counts.get(fmt, 0) + 1
                except Exception as e:
                    counts["errors"] += 1
                    sys.stderr.write(f"Error loading {f}: {e}\n")

            total = sum(v for k, v in counts.items() if k != "errors")
            detail = ", ".join(
                f"{v} {_FORMAT_DISPLAY.get(k, k.upper())}"
                for k, v in counts.items()
                if k != "errors" and v > 0
            )
            sys.stderr.write(
                f"Ontology loaded: {total} files ({detail})"
                + (f", {counts['errors']} errors" if counts['errors'] else "") + "\n"
            )
            self._check_owl_imports()
            self._build_schema_cache()

    def _check_owl_imports(self):
        """Logs a single-line summary if owl:imports declarations are present."""
        count = sum(1 for _ in self.graph.triples((None, OWL.imports, None)))
        if count:
            sys.stderr.write(f"Note: {count} owl:imports declaration(s) found (imported URIs not fetched).\n")

    # ── Schema Parsing Helpers ────────────────────────────────────────────────

    def _resolve_class_refs(self, node) -> list[str]:
        """Resolves a domain/range RDF node to a flat list of class URI strings.

        Handles:
        - Plain URIRef  → [uri]
        - owl:unionOf blank node → [uri, uri, ...]
        """
        if isinstance(node, rdflib.URIRef):
            return [str(node)]
        if isinstance(node, rdflib.BNode):
            union = self.graph.value(node, OWL.unionOf)
            if union:
                return [
                    str(item)
                    for item in self.graph.items(union)
                    if isinstance(item, rdflib.URIRef)
                ]
        return []

    def _get_shacl_target_classes(self, shape) -> list[str]:
        """Returns all target class URIs for a SHACL NodeShape.

        Supports:
        - sh:targetClass   (most common)
        - sh:targetSubjectsOf  (infer from property rdfs:domain)
        - sh:targetObjectsOf   (infer from property rdfs:range)
        """
        targets = []

        # sh:targetClass
        for tc in self.graph.objects(shape, SHACL.targetClass):
            if isinstance(tc, rdflib.URIRef):
                targets.append(str(tc))

        # sh:targetSubjectsOf — infer class from rdfs:domain of the given property
        for prop in self.graph.objects(shape, SHACL.targetSubjectsOf):
            for d_node in self.graph.objects(prop, RDFS.domain):
                targets.extend(self._resolve_class_refs(d_node))

        # sh:targetObjectsOf — infer class from rdfs:range of the given property
        for prop in self.graph.objects(shape, SHACL.targetObjectsOf):
            for r_node in self.graph.objects(prop, RDFS.range):
                targets.extend(self._resolve_class_refs(r_node))

        return list(set(targets))

    def _get_shacl_node_targets(self, prop_shape) -> list[str]:
        """Resolves all node target class URIs from a SHACL property shape.

        Supports:
        - sh:node → sh:targetClass
        - sh:class (direct class constraint)
        - sh:qualifiedValueShape → sh:class / sh:targetClass
        - sh:or  (list of alternative sub-shapes, resolved recursively)
        """
        targets = []

        # sh:node → sh:targetClass
        for node_ref in self.graph.objects(prop_shape, SHACL.node):
            tc = self.graph.value(node_ref, SHACL.targetClass)
            if tc and isinstance(tc, rdflib.URIRef):
                targets.append(str(tc))
            # sh:node itself may carry sh:class
            for cls in self.graph.objects(node_ref, SHACL_CLASS):
                if isinstance(cls, rdflib.URIRef):
                    targets.append(str(cls))

        # sh:class — direct class constraint on the value
        for cls in self.graph.objects(prop_shape, SHACL_CLASS):
            if isinstance(cls, rdflib.URIRef):
                targets.append(str(cls))

        # sh:qualifiedValueShape → sh:class or sh:targetClass
        for qvs in self.graph.objects(prop_shape, SHACL_QUALIFIED_VALUE):
            for cls in self.graph.objects(qvs, SHACL_CLASS):
                if isinstance(cls, rdflib.URIRef):
                    targets.append(str(cls))
            tc = self.graph.value(qvs, SHACL.targetClass)
            if tc and isinstance(tc, rdflib.URIRef):
                targets.append(str(tc))

        # sh:or — list of alternative property shapes, resolved recursively
        for or_list in self.graph.objects(prop_shape, SHACL_OR):
            for alt_shape in self.graph.items(or_list):
                targets.extend(self._get_shacl_node_targets(alt_shape))

        return list(set(targets))

    def _build_schema_cache(self):
        """Organises ontology items into memory. Embeddings are computed/loaded separately."""
        # 1. Extract Class Information (named URIs only — skip OWL anonymous blank nodes)
        named_classes = {
            s for s in
            set(self.graph.subjects(RDF.type, OWL.Class)) | set(self.graph.subjects(RDF.type, RDFS.Class))
            if isinstance(s, rdflib.URIRef)
        }
        for s in named_classes:
            uri = str(s)
            name = str(self.graph.value(s, RDFS.label) or _local_name(uri))
            comment = str(self.graph.value(s, RDFS.comment) or "No description")
            self.classes[uri] = {
                'name': name,
                'comment': comment,
                'superclasses': [],
                'subclasses': [],
                # Embeddings populated by _build_embeddings()
            }

        # 2. Build Hierarchy
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, None)):
            s_uri, o_uri = str(s), str(o)
            if s_uri in self.classes and o_uri in self.classes:
                self.classes[s_uri]['superclasses'].append(o_uri)
                self.classes[o_uri]['subclasses'].append(s_uri)

        # 3. Extract Property Information (named URIs only)
        # Query by known property types directly to avoid missing multi-typed properties
        # (e.g. owl:ObjectProperty + owl:FunctionalProperty declared together).
        # Domain/range resolved via _resolve_class_refs to handle owl:unionOf blank nodes.
        seen_props = set()
        for prop_type, type_name in _PROP_TYPE_NAMES.items():
            for p_uri in self.graph.subjects(RDF.type, prop_type):
                if not isinstance(p_uri, rdflib.URIRef):
                    continue
                uri_str = str(p_uri)
                if uri_str in seen_props:
                    continue
                seen_props.add(uri_str)

                name = str(self.graph.value(p_uri, RDFS.label) or _local_name(uri_str))
                comment = str(self.graph.value(p_uri, RDFS.comment) or "")

                domain = []
                for d_node in self.graph.objects(p_uri, RDFS.domain):
                    domain.extend(self._resolve_class_refs(d_node))

                range_ = []
                for r_node in self.graph.objects(p_uri, RDFS.range):
                    range_.extend(self._resolve_class_refs(r_node))

                self.properties[uri_str] = {
                    'name': name,
                    'comment': comment,
                    'type': type_name,
                    'domain': list(set(domain)),
                    'range': list(set(range_)),
                    # Embeddings populated by _build_embeddings()
                }

        # 4. Extract SHACL Shape Constraints and augment property domains
        # Supports: sh:targetClass, sh:targetSubjectsOf, sh:targetObjectsOf,
        #           sh:node, sh:class, sh:qualifiedValueShape, sh:or
        for shape in self.graph.subjects(RDF.type, SHACL.NodeShape):
            target_uris = self._get_shacl_target_classes(shape)
            if not target_uris:
                continue

            for target_uri in target_uris:
                if target_uri not in self.shapes:
                    self.shapes[target_uri] = []

                for prop_shape in self.graph.objects(shape, SHACL.property):
                    path = self.graph.value(prop_shape, SHACL.path)
                    if not path:
                        continue
                    path_uri = str(path)

                    # Augment property domain from SHACL shape
                    if path_uri in self.properties and target_uri not in self.properties[path_uri]['domain']:
                        self.properties[path_uri]['domain'].append(target_uri)

                    # Resolve all node/class targets from this property shape
                    node_targets = self._get_shacl_node_targets(prop_shape)

                    self.shapes[target_uri].append({
                        'path': path_uri,
                        'minCount':    self.graph.value(prop_shape, SHACL.minCount),
                        'maxCount':    self.graph.value(prop_shape, SHACL.maxCount),
                        'datatype':    self.graph.value(prop_shape, SHACL.datatype),
                        'node_targets': node_targets,  # list; replaces single node_target
                    })

        # 5. Compute or restore embeddings (with on-disk caching)
        self._build_embeddings()

    # ── Embedding Cache ──────────────────────────────────────────────────────

    def _get_cache_path(self) -> str:
        """Returns the path for the embedding cache file."""
        if os.path.isdir(self.ontology_path):
            return os.path.join(self.ontology_path, ".embedding_cache.pkl")
        return self.ontology_path + ".embedding_cache.pkl"

    def _compute_cache_key(self) -> str:
        """Computes a cache key from the loaded ontology files' paths, sizes, and mtimes."""
        h = hashlib.md5()
        for filepath, _ in sorted(self._ontology_files):
            try:
                stat = os.stat(filepath)
                h.update(f"{filepath}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            except OSError:
                pass
        return h.hexdigest()

    def _load_embedding_cache(self, cache_path: str, cache_key: str) -> bool:
        """Tries to restore embeddings from cache. Returns True on cache hit."""
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('cache_key') != cache_key or cached.get('version') != _CACHE_VERSION:
                return False
            for uri, embs in cached.get('class_embeddings', {}).items():
                if uri in self.classes:
                    self.classes[uri]['name_embedding'] = embs['name']
                    self.classes[uri]['comment_embedding'] = embs['comment']
            for uri, embs in cached.get('property_embeddings', {}).items():
                if uri in self.properties:
                    self.properties[uri]['name_embedding'] = embs['name']
                    self.properties[uri]['comment_embedding'] = embs['comment']
            # Reject partial cache (e.g., schema changed but mtime didn't)
            all_entries = list(self.classes.values()) + list(self.properties.values())
            if any('name_embedding' not in v for v in all_entries):
                return False
            return True
        except Exception:
            return False

    def _save_embedding_cache(self, cache_path: str, cache_key: str):
        """Persists current embeddings to disk."""
        try:
            cached = {
                'version': _CACHE_VERSION,
                'cache_key': cache_key,
                'class_embeddings': {
                    uri: {'name': info['name_embedding'], 'comment': info['comment_embedding']}
                    for uri, info in self.classes.items()
                },
                'property_embeddings': {
                    uri: {'name': info['name_embedding'], 'comment': info['comment_embedding']}
                    for uri, info in self.properties.items()
                },
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)
            sys.stderr.write(f"Embedding cache saved to {cache_path}\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Could not save embedding cache: {e}\n")

    def _build_embeddings(self):
        """Loads embeddings from cache, or batch-encodes and saves them."""
        cache_path = self._get_cache_path()
        cache_key = self._compute_cache_key()

        if self._load_embedding_cache(cache_path, cache_key):
            sys.stderr.write("Embedding cache hit — skipping re-encoding.\n")
            return

        sys.stderr.write("Embedding cache miss — batch-encoding all terms...\n")

        class_uris = list(self.classes.keys())
        prop_uris = list(self.properties.keys())

        # Collect all texts and batch-encode in a single pass (much faster than one-by-one)
        texts = (
            [self.classes[u]['name']    for u in class_uris]
            + [self.classes[u]['comment'] for u in class_uris]
            + [self.properties[u]['name']    for u in prop_uris]
            + [self.properties[u]['comment'] for u in prop_uris]
        )

        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=False) if texts else []

        n_c = len(class_uris)
        n_p = len(prop_uris)

        for i, uri in enumerate(class_uris):
            self.classes[uri]['name_embedding']    = embeddings[i]
            self.classes[uri]['comment_embedding'] = embeddings[n_c + i]

        for i, uri in enumerate(prop_uris):
            self.properties[uri]['name_embedding']    = embeddings[2 * n_c + i]
            self.properties[uri]['comment_embedding'] = embeddings[2 * n_c + n_p + i]

        self._save_embedding_cache(cache_path, cache_key)

    # ── Schema Queries ───────────────────────────────────────────────────────

    def get_properties_for_class(self, class_uri: str, property_type: str = None, include_facets: bool = False) -> list[str]:
        """Returns all valid properties for a given class, including inherited ones.

        Args:
            class_uri: The URI of the class.
            property_type: Optional filter ('ObjectProperty', 'DatatypeProperty').
            include_facets: If True, also includes properties of related Facet/Component classes.
        """
        relevant_classes = {class_uri} | set(self.get_transitive_superclasses(class_uri))
        if include_facets:
            for facet_uri in self.get_related_facets(class_uri):
                relevant_classes.add(facet_uri)
                relevant_classes.update(self.get_transitive_superclasses(facet_uri))

        valid_props = []
        for p_uri, info in self.properties.items():
            if property_type and info['type'] != property_type:
                continue
            if not info['domain'] or any(d in relevant_classes for d in info['domain']):
                valid_props.append(p_uri)
        return list(set(valid_props))

    def _is_composition_prop(self, prop_uri: str) -> bool:
        """Returns True if the property looks like a composition/grouping property."""
        return any(kw in _local_name(prop_uri).lower() for kw in _COMPOSITION_PROP_KEYWORDS)

    def _is_composition_class(self, class_uri: str) -> bool:
        """Returns True if the class name suggests it is a subordinate grouping class."""
        return any(kw in _local_name(class_uri).lower() for kw in _COMPOSITION_CLASS_KEYWORDS)

    def get_related_facets(self, class_uri: str) -> list[str]:
        """
        Finds related Facet/Component classes for a given class.

        Strategies (applied in order, results merged):
        1. Direct composition property range  — ObjectProperty whose local name matches
           composition keywords (hasFacet, hasComponent, hasPart, hasExtension, hasFeature)
           and whose domain includes this class.
        2. rdfs:subPropertyOf chain  — sub-properties of any composition-like property.
        3. SHACL sh:node / sh:class / sh:qualifiedValueShape  — shape-defined node targets
           for composition-like paths.
        4. SHACL path name heuristic  — shape entries whose path name matches composition
           keywords; falls back to property range.
        5. Range class name heuristic  — any ObjectProperty whose range class name contains
           Facet/Component/Extension keywords (catches ontologies with no standard naming).
        """
        facets = set()
        relevant_classes = {class_uri} | set(self.get_transitive_superclasses(class_uri))

        # Identify all composition-like root properties
        composition_prop_uris = {
            p_uri for p_uri in self.properties
            if self._is_composition_prop(p_uri)
        }

        # 1. Direct range of composition properties whose domain includes this class
        for p_uri in composition_prop_uris:
            info = self.properties[p_uri]
            if not info['domain'] or any(d in relevant_classes for d in info['domain']):
                facets.update(info['range'])

        # 2. rdfs:subPropertyOf — sub-properties of any composition-like property
        for p_uri, info in self.properties.items():
            p_ref = rdflib.URIRef(p_uri)
            for super_prop in self.graph.objects(p_ref, RDFS.subPropertyOf):
                if str(super_prop) in composition_prop_uris:
                    if not info['domain'] or any(d in relevant_classes for d in info['domain']):
                        facets.update(info['range'])

        # 3 & 4. SHACL shape entries for this class and its superclasses
        for target in relevant_classes:
            if target not in self.shapes:
                continue
            for shape_entry in self.shapes[target]:
                path = shape_entry['path']
                is_composition_path = self._is_composition_prop(path)

                # Strategy 3: sh:node / sh:class / sh:qualifiedValueShape resolved targets
                if shape_entry.get('node_targets') and is_composition_path:
                    facets.update(shape_entry['node_targets'])

                # Strategy 4: fallback to property range via path name
                elif is_composition_path:
                    p_info = self.properties.get(path)
                    if p_info:
                        facets.update(p_info['range'])

        # 5. Range class name heuristic — catches ontologies with non-standard naming
        for p_uri, info in self.properties.items():
            if info['type'] != 'ObjectProperty':
                continue
            if info['domain'] and not any(d in relevant_classes for d in info['domain']):
                continue
            for range_uri in info['range']:
                if self._is_composition_class(range_uri):
                    facets.add(range_uri)

        return list(facets)

    def infer_datatype(self, value: str) -> str | None:
        """Heuristic to infer XSD datatype from a string value."""
        if not value:
            return None
        if _RE_DATETIME.match(value):
            return "http://www.w3.org/2001/XMLSchema#dateTime"
        if _RE_INTEGER.match(value):
            return "http://www.w3.org/2001/XMLSchema#integer"
        if value.lower() in ("true", "false"):
            return "http://www.w3.org/2001/XMLSchema#boolean"
        return "http://www.w3.org/2001/XMLSchema#string"

    def get_candidate_relations(self, subject_types: list, object_types: list) -> list:
        """
        Finds all possible relations between two sets of types.
        Returns list of dicts: {uri, path_type: 'direct'|'facet'|'inverse', bridge: facet_uri_or_none}
        """
        candidates = []

        all_s_types = set()
        for st in subject_types:
            all_s_types.add(st)
            all_s_types.update(self.get_transitive_superclasses(st))

        all_o_types = set()
        for ot in object_types:
            all_o_types.add(ot)
            all_o_types.update(self.get_transitive_superclasses(ot))

        # 1. Direct & Facet-based Forward Relations
        for st in subject_types:
            direct_props = self.get_properties_for_class(st, "ObjectProperty", include_facets=False)
            for p_uri in direct_props:
                p_range = self.properties[p_uri]['range']
                if not p_range or any(r in all_o_types for r in p_range):
                    candidates.append({'uri': p_uri, 'path_type': 'direct', 'bridge': None})

            facets = self.get_related_facets(st)
            for f_uri in facets:
                f_props = self.get_properties_for_class(f_uri, "ObjectProperty", include_facets=False)
                for p_uri in f_props:
                    p_range = self.properties[p_uri]['range']
                    if not p_range or any(r in all_o_types for r in p_range):
                        candidates.append({'uri': p_uri, 'path_type': 'facet', 'bridge': f_uri})

        # 2. Inverse Relations (Object -> Subject)
        seen_uris = {c['uri'] for c in candidates}
        for ot in object_types:
            inv_props = self.get_properties_for_class(ot, "ObjectProperty", include_facets=True)
            for p_uri in inv_props:
                if p_uri in seen_uris:
                    continue
                p_range = self.properties[p_uri]['range']
                if any(r in all_s_types for r in p_range):
                    candidates.append({'uri': p_uri, 'path_type': 'inverse', 'bridge': None})
                    seen_uris.add(p_uri)

        return candidates

    def semantic_similarity(self, query, embeddings):
        """Calculates cosine similarity."""
        query_vec = self.model.encode(query, show_progress_bar=False)
        norm_q = np.linalg.norm(query_vec)
        scores = []
        for emb in embeddings:
            norm_e = np.linalg.norm(emb)
            if norm_q < 1e-9 or norm_e < 1e-9:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(query_vec, emb) / (norm_q * norm_e)))
        return scores

    def weighted_similarity(self, query, name_embeddings, comment_embeddings, weights=(0.5, 0.5)):
        """Calculates weighted similarity between query and (name, comment) pairs."""
        query_vec = self.model.encode(query, show_progress_bar=False)
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-9:
            return [0.0] * len(name_embeddings)

        scores = []
        for n_emb, c_emb in zip(name_embeddings, comment_embeddings):
            n_norm = np.linalg.norm(n_emb)
            c_norm = np.linalg.norm(c_emb)
            n_sim = np.dot(query_vec, n_emb) / (q_norm * n_norm) if n_norm > 1e-9 else 0.0
            c_sim = np.dot(query_vec, c_emb) / (q_norm * c_norm) if c_norm > 1e-9 else 0.0
            scores.append(float(n_sim * weights[0] + c_sim * weights[1]))
        return scores

    def get_transitive_superclasses(self, class_uri):
        """Returns the full inheritance tree upwards."""
        superclasses = set()
        stack = [class_uri]
        while stack:
            curr = stack.pop()
            parents = self.classes.get(curr, {}).get('superclasses', [])
            for p in parents:
                if p not in superclasses:
                    superclasses.add(p)
                    stack.append(p)
        return list(superclasses)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect the parsed ontology schema from a directory or file.")
    parser.add_argument("ontology_path", help="Path to an ontology directory or file")
    parser.add_argument("--classes",    action="store_true", help="Show classes (default: all sections shown)")
    parser.add_argument("--properties", action="store_true", help="Show properties")
    parser.add_argument("--shapes",     action="store_true", help="Show SHACL shapes")
    parser.add_argument("--filter", metavar="KEYWORD", help="Filter output by keyword (case-insensitive)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    show_all = not (args.classes or args.properties or args.shapes)

    engine = OntologyEngine(args.ontology_path)
    kw = args.filter.lower() if args.filter else None

    def matches(text):
        return kw is None or kw in text.lower()

    if args.as_json:
        output = {}
        if show_all or args.classes:
            output["classes"] = {
                uri: {k: v for k, v in info.items() if not hasattr(v, 'tolist')}
                for uri, info in engine.classes.items()
                if matches(uri) or matches(info.get('name', '')) or matches(info.get('comment', ''))
            }
        if show_all or args.properties:
            output["properties"] = {
                uri: {k: v for k, v in info.items() if not hasattr(v, 'tolist')}
                for uri, info in engine.properties.items()
                if matches(uri) or matches(info.get('name', '')) or matches(info.get('comment', ''))
            }
        if show_all or args.shapes:
            output["shapes"] = {
                uri: entries for uri, entries in engine.shapes.items()
                if matches(uri)
            }
        print(json.dumps(output, indent=2, default=str))

    else:
        if show_all or args.classes:
            classes = [
                (uri, info) for uri, info in sorted(engine.classes.items())
                if matches(uri) or matches(info.get('name', '')) or matches(info.get('comment', ''))
            ]
            print(f"\n{'='*80}")
            print(f"  CLASSES  ({len(classes)} shown / {len(engine.classes)} total)")
            print(f"{'='*80}")
            for uri, info in classes:
                print(f"\n  [{info['name']}]")
                print(f"  URI     : {uri}")
                print(f"  Comment : {info.get('comment', '')}")
                if info.get('superclasses'):
                    print(f"  Parents : {', '.join(info['superclasses'])}")
                if info.get('subclasses'):
                    print(f"  Children: {', '.join(info['subclasses'])}")
            print()

        if show_all or args.properties:
            props = [
                (uri, info) for uri, info in sorted(engine.properties.items())
                if matches(uri) or matches(info.get('name', '')) or matches(info.get('comment', ''))
            ]
            print(f"\n{'='*80}")
            print(f"  PROPERTIES  ({len(props)} shown / {len(engine.properties)} total)")
            print(f"{'='*80}")
            for uri, info in props:
                print(f"\n  [{info['name']}]  ({info['type']})")
                print(f"  URI     : {uri}")
                if info.get('comment'):
                    print(f"  Comment : {info['comment']}")
                if info.get('domain'):
                    print(f"  Domain  : {', '.join(info['domain'])}")
                if info.get('range'):
                    print(f"  Range   : {', '.join(info['range'])}")
            print()

        if show_all or args.shapes:
            shapes = [
                (uri, entries) for uri, entries in sorted(engine.shapes.items())
                if matches(uri)
            ]
            print(f"\n{'='*80}")
            print(f"  SHACL SHAPES  ({len(shapes)} shown / {len(engine.shapes)} total)")
            print(f"{'='*80}")
            for uri, entries in shapes:
                print(f"\n  Target: {uri}")
                for e in entries:
                    parts = [f"path={e['path']}"]
                    if e.get('minCount') is not None:
                        parts.append(f"min={e['minCount']}")
                    if e.get('maxCount') is not None:
                        parts.append(f"max={e['maxCount']}")
                    if e.get('datatype'):
                        parts.append(f"datatype={e['datatype']}")
                    if e.get('node_targets'):
                        parts.append(f"node_targets={', '.join(e['node_targets'])}")
                    print(f"    - {', '.join(parts)}")
            print()

        print(f"Summary: {len(engine.classes)} classes, {len(engine.properties)} properties, {len(engine.shapes)} shapes")
