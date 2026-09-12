#!/usr/bin/env python3
"""
visualize.py — Visualize extracted entities and triples from ontology extraction output.

Usage:
    python visualize.py --file output/sample.json --out output/sample.png
    python visualize.py --file output/sample.json --out output/sample.png --show-text --show-schema
"""

import argparse
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path


WELL_KNOWN_NS = {
    "http://www.w3.org/2002/07/owl#":                  "owl",
    "http://www.w3.org/2000/01/rdf-schema#":           "rdfs",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#":     "rdf",
    "http://www.w3.org/2001/XMLSchema#":               "xsd",
}

NODE_COLORS = [
    "#8ecae6", "#a8dadc", "#caffbf", "#ffd6a5", "#ffadad",
    "#bde0fe", "#ffc8dd", "#e9c46a", "#b5e48c", "#d0d1ff",
]


def _shorten_uri(uri: str) -> str:
    """Compact a URI to a readable namespace:path form."""
    if not uri:
        return ""
    # UCO: https://ontology.unifiedcyberontology.org/uco/module/Name
    m = re.match(r"https?://ontology\.unifiedcyberontology\.org/(uco/.+)", uri)
    if m:
        return m.group(1).replace("/", ":")
    # Well-known (owl, rdfs, rdf, xsd)
    for ns, prefix in WELL_KNOWN_NS.items():
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):]}"
    # Generic: last two path segments
    parts = [p for p in re.split(r"[/#]", uri.rstrip("/#")) if p]
    return ":".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else uri)


def _prop_prefix(prop_type: str) -> str:
    return {"DatatypeProperty": "[DP]", "ObjectProperty": "[OP]"}.get(prop_type, "")


def _wrap_label(name: str, max_chars: int = 16) -> str:
    """Word-wrap a node name so long labels don't bleed over neighboring nodes."""
    if len(name) <= max_chars:
        return name
    words = name.split()
    lines, current = [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > max_chars:
            lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


def _repulse(pos: dict, min_dist: float = 0.35, iterations: int = 30) -> dict:
    """Simple post-processing: push nodes apart when they are too close."""
    import numpy as np
    nodes = list(pos)
    coords = {n: list(pos[n]) for n in nodes}
    for _ in range(iterations):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                ax_, ay_ = coords[a]; bx, by = coords[b]
                ddx, ddy = ax_ - bx, ay_ - by
                d = (ddx ** 2 + ddy ** 2) ** 0.5
                if d < min_dist and d > 1e-6:
                    push = (min_dist - d) / 2.0
                    nx_ = ddx / d * push; ny_ = ddy / d * push
                    coords[a][0] += nx_; coords[a][1] += ny_
                    coords[b][0] -= nx_; coords[b][1] -= ny_
                    moved = True
        if not moved:
            break
    return {n: tuple(coords[n]) for n in nodes}


def _bezier_point(t: float, p1, pc, p2) -> tuple:
    """Quadratic bezier position at parameter t."""
    mt = 1.0 - t
    return (
        mt * mt * p1[0] + 2 * t * mt * pc[0] + t * t * p2[0],
        mt * mt * p1[1] + 2 * t * mt * pc[1] + t * t * p2[1],
    )


def visualize(
    data: dict,
    out_path: Path,
    show_text: bool = False,
    show_schema: bool = False,
    fig_width: float = 16,
    fig_height: float = 10,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    entities = data.get("extracted_entities", [])
    triples  = data.get("extracted_triplets", [])

    entity_class: dict[str, str] = {e["name"]: e.get("class", "") for e in entities}

    # Collect all node names
    node_names: set[str] = {e["name"] for e in entities}
    for t in triples:
        if t.get("subject"): node_names.add(t["subject"])
        if t.get("object"):  node_names.add(t["object"])

    # Assign colors by class (deterministic order)
    unique_classes = list(dict.fromkeys(
        entity_class.get(n, "") for n in sorted(node_names)
    ))
    class_color: dict[str, str] = {
        cls: NODE_COLORS[i % len(NODE_COLORS)]
        for i, cls in enumerate(unique_classes)
    }

    # Build directed multigraph
    G = nx.MultiDiGraph()
    for name in node_names:
        G.add_node(name)
    for t in triples:
        s, o = t.get("subject", ""), t.get("object", "")
        if s and o:
            G.add_edge(s, o, **t)

    # ── Layout ──────────────────────────────────────────────────────────────────
    n = len(node_names)
    # kamada_kawai gives more uniform spacing for small-medium graphs
    try:
        pos = nx.kamada_kawai_layout(G, scale=1.8)
    except Exception:
        k = max(2.0, 4.0 / max(1, n ** 0.35))
        pos = nx.spring_layout(G, seed=42, k=k, iterations=300, scale=1.8)

    # Push nodes apart if still too close
    pos = _repulse(pos, min_dist=0.40, iterations=50)

    # ── Figure ──────────────────────────────────────────────────────────────────
    if show_text:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs  = fig.add_gridspec(2, 1, height_ratios=[0.22, 0.78], hspace=0.03)
        tax = fig.add_subplot(gs[0])
        ax  = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        tax = None

    # ── Text panel ──────────────────────────────────────────────────────────────
    if tax is not None:
        tax.set_axis_off()
        lines = []
        if data.get("file"):
            lines.append(f"File: {data['file']}")
        if data.get("text"):
            if lines:
                lines.append("")
            lines.extend(textwrap.wrap(data["text"], width=145))
        tax.text(
            0.005, 0.97, "\n".join(lines),
            transform=tax.transAxes, va="top", ha="left",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f9fb", edgecolor="#cccccc"),
        )

    ax.set_axis_off()

    # ── Nodes ───────────────────────────────────────────────────────────────────
    node_list    = list(G.nodes())
    node_color_list = [
        class_color.get(entity_class.get(nd, ""), "#dddddd")
        for nd in node_list
    ]
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list, ax=ax,
        node_color=node_color_list,
        node_size=2000, alpha=0.92,
        edgecolors="#444444", linewidths=0.9,
    )

    if show_schema:
        node_labels = {
            nd: f"{_wrap_label(nd)}\n{_shorten_uri(entity_class.get(nd, ''))}"
            for nd in node_list
        }
    else:
        node_labels = {nd: _wrap_label(nd) for nd in node_list}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=7)

    # ── Edges ───────────────────────────────────────────────────────────────────
    # Group by (u, v) to stagger parallel edges
    uv_groups: dict = defaultdict(list)
    for u, v, _k, d in G.edges(keys=True, data=True):
        uv_groups[(u, v)].append(d)

    for (u, v), edge_list in uv_groups.items():
        n_parallel = len(edge_list)
        for idx, d in enumerate(edge_list):
            # Stagger arc curvature for parallel / self-loop edges
            if u == v:
                rad = 0.5 + idx * 0.25
            else:
                rad = (idx - (n_parallel - 1) / 2) * 0.30

            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                connectionstyle=f"arc3,rad={rad}",
                arrowsize=15, edge_color="#555555",
                width=1.2, alpha=0.65, arrows=True,
                min_source_margin=28, min_target_margin=28,
            )

            # ── Accurate label position via quadratic bezier ─────────────────
            # matplotlib arc3 control point: mid + rad * dist * perp_unit
            x1, y1 = pos[u]; x2, y2 = pos[v]
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ddx, ddy = x2 - x1, y2 - y1
            dist = (ddx ** 2 + ddy ** 2) ** 0.5 or 1e-6
            # perpendicular unit: rot90 of (ddx, ddy)/dist
            px, py = -ddy / dist, ddx / dist
            # control point
            pcx = mx + rad * dist * px
            pcy = my + rad * dist * py
            # Place label at t∈[0.35, 0.65] spread across parallel edges
            if n_parallel == 1:
                t_label = 0.50
            else:
                t_label = 0.35 + 0.30 * idx / (n_parallel - 1)
            lx, ly = _bezier_point(t_label, (x1, y1), (pcx, pcy), (x2, y2))

            relation = d.get("relation", "")
            label_lines = [relation]
            if show_schema:
                prop_uri  = d.get("property", "")
                prop_type = d.get("property_type", "")
                pfx       = _prop_prefix(prop_type)
                short     = _shorten_uri(prop_uri)
                if pfx or short:
                    label_lines.append(f"{pfx} {short}".strip())

            ax.text(
                lx, ly, "\n".join(label_lines),
                fontsize=6, ha="center", va="center", zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="white",
                    alpha=0.90, edgecolor="#bbbbbb", linewidth=0.6,
                ),
            )

    # ── Legend ──────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=class_color[cls], label=_shorten_uri(cls) or "(unknown)")
        for cls in unique_classes
        if cls in class_color
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches, loc="best",
            fontsize=7, title="Entity Classes", title_fontsize=7,
            framealpha=0.92, edgecolor="#cccccc",
        )

    title = Path(data.get("file", "")).stem or "Ontology Graph"
    if data.get("ontology"):
        title += f"  [{data['ontology']}]"
    ax.set_title(title, fontsize=10, pad=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ontology extraction output as a knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file",        required=True,       help="Input JSON file")
    parser.add_argument("--out",         required=True,       help="Output image (png / svg / pdf)")
    parser.add_argument("--show-text",   action="store_true", help="Show file & text fields at the top")
    parser.add_argument("--show-schema", action="store_true", help="Show ontology schema URIs on nodes/edges")
    parser.add_argument("--width",       type=float, default=16, help="Figure width in inches (default: 16)")
    parser.add_argument("--height",      type=float, default=10, help="Figure height in inches (default: 10)")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visualize(
        data, out_path,
        show_text=args.show_text,
        show_schema=args.show_schema,
        fig_width=args.width,
        fig_height=args.height,
    )


if __name__ == "__main__":
    main()
