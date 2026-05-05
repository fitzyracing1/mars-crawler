"""
3XB Entity Graph API
REST interface into the live 3XB entity database.
"""

import math
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "3xb_internet.db"))
PORT = int(os.environ.get("PORT", 8000))

app = FastAPI(
    title="3XB Entity Graph API",
    description="Weighted entity tagging across the web — by Community Plan, Inc.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def decay_weight(weight: float, last_seen: float, half_life_days: float = 30.0) -> float:
    days = (time.time() - last_seen) / 86400
    factor = math.exp(-math.log(2) * days / half_life_days)
    return round(weight * factor, 4)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html><head><title>3XB API</title>
    <style>body{font-family:monospace;background:#0a0a0f;color:#7df7ff;padding:40px}
    a{color:#ff6b9d} h1{color:#fff} li{margin:6px 0}</style></head>
    <body>
    <h1>3XB Entity Graph API v2.0</h1>
    <p>Weighted entity tagging across the web — Community Plan, Inc.</p>
    <ul>
      <li><a href="/docs">Interactive API Docs (Swagger)</a></li>
      <li><a href="/stats">GET /stats</a> — crawl & entity stats</li>
      <li><a href="/entities?limit=20">GET /entities</a> — top entities</li>
      <li>GET /entity/{name} — single entity detail</li>
      <li>GET /entity/{name}/neighbors — connected entities</li>
      <li>GET /entity/{name}/loan-score — loan risk score</li>
      <li><a href="/graph">GET /graph</a> — full node+edge graph</li>
      <li>GET /search?q= — search by name</li>
      <li>GET /tag/{tag} — entities by type</li>
    </ul>
    </body></html>
    """


@app.get("/stats")
def stats():
    """Overall crawl and entity stats."""
    with db() as conn:
        pages   = conn.execute("SELECT COUNT(*) FROM crawled_urls").fetchone()[0]
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        edges   = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        tags    = conn.execute(
            "SELECT tag, COUNT(*) as n FROM entities GROUP BY tag ORDER BY n DESC"
        ).fetchall()
        top5    = conn.execute(
            "SELECT name, tag, weight, frequency FROM entities ORDER BY weight DESC, frequency DESC LIMIT 5"
        ).fetchall()
    return {
        "pages_crawled": pages,
        "unique_entities": entities,
        "relationship_edges": edges,
        "entities_by_type": {r["tag"]: r["n"] for r in tags},
        "top_5": [dict(r) for r in top5],
    }


@app.get("/entities")
def list_entities(
    limit: int = Query(50, le=500),
    min_weight: float = Query(0.0, ge=0.0, le=1.0),
    tag: Optional[str] = None,
    sort_by: str = Query("weight", enum=["weight", "frequency"]),
):
    """List top entities, optionally filtered by type and minimum weight."""
    order = "weight DESC, frequency DESC" if sort_by == "weight" else "frequency DESC, weight DESC"
    with db() as conn:
        if tag:
            rows = conn.execute(
                f"SELECT * FROM entities WHERE tag=? AND weight>=? ORDER BY {order} LIMIT ?",
                (tag, min_weight, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM entities WHERE weight>=? ORDER BY {order} LIMIT ?",
                (min_weight, limit)
            ).fetchall()
    return [dict(r) for r in rows]


@app.get("/entity/{name}")
def get_entity(name: str):
    """Full detail for a single entity including live (decayed) weight."""
    with db() as conn:
        row = conn.execute("SELECT * FROM entities WHERE name=?", (name,)).fetchone()
        if not row:
            raise HTTPException(404, f"Entity '{name}' not found")
        r = dict(row)
        r["live_weight"] = decay_weight(r["weight"], r["last_seen"])
        r["highlighted"] = r["live_weight"] >= 0.75

        neighbors = conn.execute(
            "SELECT target as entity, relation, strength FROM edges WHERE source=? "
            "UNION SELECT source as entity, '←'||relation, strength FROM edges WHERE target=? LIMIT 50",
            (name, name)
        ).fetchall()
        r["neighbors"] = [dict(n) for n in neighbors]
    return r


@app.get("/entity/{name}/neighbors")
def get_neighbors(name: str, depth: int = Query(1, ge=1, le=3)):
    """BFS neighbor traversal up to `depth` hops."""
    with db() as conn:
        row = conn.execute("SELECT name FROM entities WHERE name=?", (name,)).fetchone()
        if not row:
            raise HTTPException(404, f"Entity '{name}' not found")

        visited = {name: 1.0}
        frontier = [name]
        result = {}

        for hop in range(depth):
            next_frontier = []
            for node in frontier:
                edges = conn.execute(
                    "SELECT target, strength FROM edges WHERE source=?", (node,)
                ).fetchall()
                for e in edges:
                    nb, strength = e["target"], e["strength"]
                    if nb not in visited:
                        propagated = round(visited[node] * strength, 4)
                        visited[nb] = propagated
                        result[nb] = {"propagated_score": propagated, "hop": hop + 1}
                        next_frontier.append(nb)
            frontier = next_frontier

    return {"entity": name, "depth": depth, "neighbors": result}


@app.get("/entity/{name}/loan-score")
def loan_score(name: str):
    """
    Composite loan risk score for an entity.
    Combines own weight + 2-hop neighbor graph risk.
    Returns APPROVE / REVIEW / DECLINE recommendation.
    """
    with db() as conn:
        row = conn.execute(
            "SELECT weight, last_seen FROM entities WHERE name=?", (name,)
        ).fetchone()
        if not row:
            raise HTTPException(404, f"Entity '{name}' not found")

        own_weight = decay_weight(row["weight"], row["last_seen"])

        edges = conn.execute(
            "SELECT target, strength FROM edges WHERE source=?", (name,)
        ).fetchall()

        neighbor_scores = []
        for e in edges:
            nb_row = conn.execute(
                "SELECT weight, last_seen FROM entities WHERE name=?", (e["target"],)
            ).fetchone()
            if nb_row:
                nb_w = decay_weight(nb_row["weight"], nb_row["last_seen"])
                neighbor_scores.append(nb_w * e["strength"])

        neighbor_avg = round(sum(neighbor_scores) / len(neighbor_scores), 4) if neighbor_scores else own_weight
        composite = round(own_weight * 0.7 + neighbor_avg * 0.3, 4)

    return {
        "borrower": name,
        "own_weight": own_weight,
        "neighbor_risk_avg": neighbor_avg,
        "composite_score": composite,
        "recommendation": (
            "APPROVE" if composite >= 0.75 else
            "REVIEW"  if composite >= 0.55 else
            "DECLINE"
        ),
        "neighbor_count": len(neighbor_scores),
    }


@app.get("/search")
def search(q: str = Query(..., min_length=2), limit: int = Query(20, le=100)):
    """Search entities by name (case-insensitive substring)."""
    with db() as conn:
        rows = conn.execute(
            "SELECT name, tag, weight, frequency FROM entities "
            "WHERE name LIKE ? ORDER BY weight DESC, frequency DESC LIMIT ?",
            (f"%{q}%", limit)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/tag/{tag}")
def by_tag(tag: str, limit: int = Query(50, le=200)):
    """All entities of a specific type, sorted by weight."""
    with db() as conn:
        rows = conn.execute(
            "SELECT name, tag, weight, frequency FROM entities "
            "WHERE tag=? ORDER BY weight DESC, frequency DESC LIMIT ?",
            (tag, limit)
        ).fetchall()
    if not rows:
        raise HTTPException(404, f"No entities found for tag '{tag}'")
    return [dict(r) for r in rows]


@app.get("/tags")
def list_tags():
    """All entity types and their counts."""
    with db() as conn:
        rows = conn.execute(
            "SELECT tag, COUNT(*) as count, AVG(weight) as avg_weight "
            "FROM entities GROUP BY tag ORDER BY count DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/graph")
def graph(
    top_n: int = Query(200, le=1000),
    min_weight: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Full node + edge graph for visualization.
    top_n limits nodes; only edges between those nodes are returned.
    """
    with db() as conn:
        nodes = conn.execute(
            "SELECT name, tag, weight, frequency FROM entities "
            "WHERE weight>=? ORDER BY weight DESC, frequency DESC LIMIT ?",
            (min_weight, top_n)
        ).fetchall()
        node_ids = {r["name"] for r in nodes}

        all_edges = conn.execute("SELECT source, target, relation, strength FROM edges").fetchall()
        edges = [dict(e) for e in all_edges
                 if e["source"] in node_ids and e["target"] in node_ids]

    return {
        "nodes": [dict(n) for n in nodes],
        "edges": edges[:2000],
        "meta": {"node_count": len(nodes), "edge_count": len(edges)},
    }


@app.post("/entity/{name}/signal")
def inject_signal(name: str, signal_strength: float = Query(..., ge=0.0, le=1.0),
                  signal_type: str = "x_engagement"):
    """
    Inject a live signal (e.g. from X/Twitter) to boost an entity's weight.
    Resets the decay clock.
    """
    with db() as conn:
        row = conn.execute("SELECT * FROM entities WHERE name=?", (name,)).fetchone()
        if not row:
            raise HTTPException(404, f"Entity '{name}' not found")

        current_weight = row["weight"]
        new_weight = min(0.9999, round(current_weight + signal_strength * 0.1, 4))
        conn.execute(
            "UPDATE entities SET weight=?, last_seen=? WHERE name=?",
            (new_weight, time.time(), name)
        )
        conn.commit()

    return {
        "entity": name,
        "signal_type": signal_type,
        "signal_strength": signal_strength,
        "weight_before": current_weight,
        "weight_after": new_weight,
    }


# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("3XB Entity Graph API")
    print(f"Docs → http://localhost:{PORT}/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=True)
