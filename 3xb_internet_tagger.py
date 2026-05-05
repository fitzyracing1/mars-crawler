"""
3XB Internet Tagger
Crawls the web, extracts named entities via spaCy NLP, and scores
them using the 3XB weighted tagging model. Persists results to SQLite.
"""

import asyncio
import aiohttp
import json
import math
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import spacy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 3XB WEIGHT MATH (v2)
# ─────────────────────────────────────────────

def sigmoid_blend(base: float, boosts: Dict[str, float]) -> float:
    total_boost = sum(boosts.values())
    raw = base + total_boost
    return round(1 / (1 + math.exp(-6 * (raw - 0.5))), 4)

def decay_weight(weight: float, last_updated: float,
                 half_life_days: float = 30.0) -> float:
    days_elapsed = (time.time() - last_updated) / 86400
    decay_factor = math.exp(-math.log(2) * days_elapsed / half_life_days)
    return round(weight * decay_factor, 4)


# ─────────────────────────────────────────────
# 3XB TAGGER CORE
# ─────────────────────────────────────────────

class ThreeXBTagger:
    HIGHLIGHT_THRESHOLD = 0.75

    def __init__(self, decay_half_life_days: float = 30.0):
        self._entities: Dict[str, dict] = {}
        self._edges: List[Tuple[str, str, str, float]] = []
        self._decay_half_life = decay_half_life_days

    def add_entity(self, entity: str, tag: str,
                   base_weight: float = 0.5,
                   boost_factors: Optional[Dict[str, float]] = None,
                   metadata: Optional[dict] = None) -> dict:
        if boost_factors is None:
            boost_factors = {}
        weight = sigmoid_blend(base_weight, boost_factors)
        record = {
            "tag": tag,
            "base_weight": base_weight,
            "boost_factors": boost_factors,
            "weight": weight,
            "highlighted": weight >= self.HIGHLIGHT_THRESHOLD,
            "last_updated": time.time(),
            "metadata": metadata or {},
        }
        if entity in self._entities:
            # Merge: boost frequency count, refresh timestamp
            existing = self._entities[entity]
            freq = existing["metadata"].get("frequency", 1) + 1
            record["metadata"]["frequency"] = freq
            # Boost weight slightly for repeated appearances
            record["boost_factors"]["frequency_signal"] = min(0.15, freq * 0.01)
            record["weight"] = sigmoid_blend(base_weight, record["boost_factors"])
            record["highlighted"] = record["weight"] >= self.HIGHLIGHT_THRESHOLD
        else:
            record["metadata"]["frequency"] = 1
        self._entities[entity] = record
        return record

    def add_relationship(self, entity_a: str, entity_b: str,
                         relation: str, strength: float = 0.5):
        self._edges.append((entity_a, entity_b, relation, strength))

    def _live_weight(self, entity: str) -> float:
        rec = self._entities.get(entity)
        if not rec:
            return 0.0
        return decay_weight(rec["weight"], rec["last_updated"],
                            self._decay_half_life)

    def get_weighted_tags(self, min_weight: float = 0.0,
                          apply_decay: bool = True) -> Dict:
        results = {}
        for entity, rec in self._entities.items():
            w = self._live_weight(entity) if apply_decay else rec["weight"]
            if w >= min_weight:
                results[entity] = {**rec, "live_weight": w}
        return dict(sorted(results.items(),
                            key=lambda x: x[1]["live_weight"], reverse=True))

    def export_graph(self) -> dict:
        return {
            "nodes": [
                {"id": e, "tag": d["tag"], "weight": self._live_weight(e),
                 "highlighted": d["highlighted"],
                 "frequency": d["metadata"].get("frequency", 1)}
                for e, d in self._entities.items()
            ],
            "edges": [
                {"source": a, "target": b, "relation": r, "strength": s}
                for a, b, r, s in self._edges
            ],
        }


# ─────────────────────────────────────────────
# NLP ENTITY EXTRACTOR
# ─────────────────────────────────────────────

# Acronyms / noise to filter out entirely
ENTITY_BLOCKLIST = {
    "AI", "API", "CEO", "CFO", "CTO", "COO", "VP", "ML", "UI", "UX",
    "URL", "DNS", "MIN", "MAX", "FAQ", "TOS", "RSS", "SDK", "NLP",
    "Q&A", "OK", "US", "U.S.", "EU", "UK", "UN", "NATO", "GDP",
    "IPO", "VC", "PE", "SaaS", "B2B", "B2C", "iOS", "macOS",
}

# spaCy label → 3XB tag mapping
LABEL_TO_TAG = {
    "PERSON":   "Person",
    "ORG":      "Organization",
    "GPE":      "Location_Political",
    "LOC":      "Location_Physical",
    "PRODUCT":  "Product",
    "WORK_OF_ART": "CreativeWork",
    "LAW":      "LegalEntity",
    "MONEY":    "FinancialFigure",
    "DATE":     "TemporalRef",
    "EVENT":    "Event",
    "FAC":      "Facility",
    "NORP":     "Group_NationalOrReligious",
    "LANGUAGE": "Language",
}

# Base weights by entity type
BASE_WEIGHTS = {
    "PERSON": 0.65,
    "ORG": 0.70,
    "GPE": 0.60,
    "LOC": 0.55,
    "PRODUCT": 0.60,
    "WORK_OF_ART": 0.50,
    "LAW": 0.65,
    "MONEY": 0.58,
    "DATE": 0.40,
    "EVENT": 0.62,
    "FAC": 0.55,
    "NORP": 0.58,
    "LANGUAGE": 0.45,
}


class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str, source_url: str) -> List[dict]:
        doc = self.nlp(text[:100_000])  # cap at 100k chars
        seen = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in LABEL_TO_TAG:
                continue
            name = ent.text.strip()
            if len(name) < 3 or len(name) > 120:
                continue
            if name in ENTITY_BLOCKLIST or name.upper() in ENTITY_BLOCKLIST:
                continue
            if re.match(r'^[A-Z]{2,5}$', name):  # skip pure acronyms
                continue
            if name in seen:
                seen[name]["count"] += 1
            else:
                seen[name] = {
                    "entity": name,
                    "spacy_label": label,
                    "tag": LABEL_TO_TAG[label],
                    "base_weight": BASE_WEIGHTS.get(label, 0.5),
                    "count": 1,
                    "source_url": source_url,
                }
        return list(seen.values())


# ─────────────────────────────────────────────
# SQLITE PERSISTENCE
# ─────────────────────────────────────────────

class EntityStore:
    def __init__(self, db_path: str = "3xb_internet.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                name        TEXT PRIMARY KEY,
                tag         TEXT,
                weight      REAL,
                frequency   INTEGER DEFAULT 1,
                last_seen   REAL,
                metadata    TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                source      TEXT,
                target      TEXT,
                relation    TEXT,
                strength    REAL,
                PRIMARY KEY (source, target, relation)
            );
            CREATE TABLE IF NOT EXISTS crawled_urls (
                url         TEXT PRIMARY KEY,
                crawled_at  REAL,
                entity_count INTEGER
            );
        """)
        self.conn.commit()

    def upsert_entity(self, name: str, tag: str, weight: float,
                      frequency: int, metadata: dict):
        self.conn.execute("""
            INSERT INTO entities (name, tag, weight, frequency, last_seen, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                weight    = MAX(weight, excluded.weight),
                frequency = frequency + excluded.frequency,
                last_seen = excluded.last_seen,
                metadata  = excluded.metadata
        """, (name, tag, weight, frequency, time.time(), json.dumps(metadata)))
        self.conn.commit()

    def upsert_edge(self, source: str, target: str, relation: str, strength: float):
        self.conn.execute("""
            INSERT OR REPLACE INTO edges (source, target, relation, strength)
            VALUES (?, ?, ?, ?)
        """, (source, target, relation, strength))
        self.conn.commit()

    def mark_crawled(self, url: str, entity_count: int):
        self.conn.execute("""
            INSERT OR REPLACE INTO crawled_urls (url, crawled_at, entity_count)
            VALUES (?, ?, ?)
        """, (url, time.time(), entity_count))
        self.conn.commit()

    def top_entities(self, limit: int = 50) -> List[dict]:
        cur = self.conn.execute("""
            SELECT name, tag, weight, frequency FROM entities
            ORDER BY weight DESC, frequency DESC LIMIT ?
        """, (limit,))
        return [{"name": r[0], "tag": r[1], "weight": r[2], "frequency": r[3]}
                for r in cur.fetchall()]

    def export_json(self, path: str = "3xb_internet_graph.json"):
        nodes = self.conn.execute(
            "SELECT name, tag, weight, frequency FROM entities ORDER BY weight DESC"
        ).fetchall()
        edges = self.conn.execute(
            "SELECT source, target, relation, strength FROM edges"
        ).fetchall()
        graph = {
            "nodes": [{"id": r[0], "tag": r[1], "weight": r[2], "frequency": r[3]}
                      for r in nodes],
            "edges": [{"source": r[0], "target": r[1], "relation": r[2], "strength": r[3]}
                      for r in edges],
        }
        with open(path, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"Graph exported → {path}  ({len(nodes)} nodes, {len(edges)} edges)")


# ─────────────────────────────────────────────
# ASYNC WEB CRAWLER
# ─────────────────────────────────────────────

class InternetTagger:
    def __init__(self,
                 seed_urls: List[str],
                 max_pages: int = 1000,
                 max_depth: int = 3,
                 concurrency: int = 10,
                 same_domain_only: bool = False,
                 db_path: str = None):

        self.seed_urls = seed_urls
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency
        self.same_domain_only = same_domain_only

        self.tagger = ThreeXBTagger()
        self.extractor = EntityExtractor()
        self.store = EntityStore(db_path or os.path.join(BASE_DIR, "3xb_internet.db"))

        self._visited: Set[str] = set()
        self._queue: deque = deque()   # (url, depth)
        self._pages_crawled = 0
        self._seed_domains = {urlparse(u).netloc for u in seed_urls}
        self._load_visited()

    def _load_visited(self):
        """Load already-crawled URLs from DB. Seeds are excluded so they
        re-crawl and generate fresh unvisited child links on resume."""
        try:
            rows = self.store.conn.execute("SELECT url FROM crawled_urls").fetchall()
            seed_set = set(self.seed_urls)
            self._visited = {r[0] for r in rows if r[0] not in seed_set}
            self._pages_crawled = len(self._visited)
            if self._pages_crawled:
                print(f"Resuming — {self._pages_crawled} pages already done, re-seeding for new links")
        except Exception:
            pass

    def _is_allowed(self, url: str, depth: int) -> bool:
        if depth > self.max_depth:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if self.same_domain_only and parsed.netloc not in self._seed_domains:
            return False
        return True

    async def _fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            headers = {"User-Agent": "3XBTagger/2.0 (entity-research-bot)"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10),
                                   allow_redirects=True, ssl=False) as resp:
                if resp.status == 200 and "text/html" in resp.headers.get("Content-Type", ""):
                    return await resp.text(errors="ignore")
        except Exception:
            pass
        return None

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = urljoin(base_url, tag["href"])
            parsed = urlparse(href)
            clean = parsed._replace(fragment="").geturl()
            links.append(clean)
        return links

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())

    async def _process_page(self, session: aiohttp.ClientSession,
                             url: str, depth: int):
        if url in self._visited or self._pages_crawled >= self.max_pages:
            return
        self._visited.add(url)

        html = await self._fetch(session, url)
        if not html:
            return

        self._pages_crawled += 1
        text = self._extract_text(html)
        entities = self.extractor.extract(text, url)

        # Feed into 3XB tagger
        entity_names = [e["entity"] for e in entities]
        for ent in entities:
            freq_boost = min(0.15, ent["count"] * 0.02)
            self.tagger.add_entity(
                ent["entity"], ent["tag"],
                base_weight=ent["base_weight"],
                boost_factors={"page_frequency": freq_boost},
                metadata={"source_url": url},
            )
            self.store.upsert_entity(
                ent["entity"], ent["tag"],
                weight=self.tagger._entities[ent["entity"]]["weight"],
                frequency=ent["count"],
                metadata={"source_url": url},
            )

        # Co-occurrence edges: entities on the same page are related
        for i, a in enumerate(entity_names[:20]):
            for b in entity_names[i+1:21]:
                if a != b:
                    self.tagger.add_relationship(a, b, "co_occurs_with", strength=0.3)
                    self.store.upsert_edge(a, b, "co_occurs_with", 0.3)

        self.store.mark_crawled(url, len(entities))
        print(f"[{self._pages_crawled}/{self.max_pages}] {url} → {len(entities)} entities")

        # Enqueue discovered links
        if depth < self.max_depth:
            for link in self._extract_links(html, url)[:30]:
                if link not in self._visited and self._is_allowed(link, depth + 1):
                    self._queue.append((link, depth + 1))

    async def run(self):
        for url in self.seed_urls:
            self._queue.append((url, 0))

        connector = aiohttp.TCPConnector(limit=self.concurrency, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            while self._queue and self._pages_crawled < self.max_pages:
                batch = []
                while self._queue and len(batch) < self.concurrency:
                    batch.append(self._queue.popleft())
                await asyncio.gather(*[
                    self._process_page(session, url, depth)
                    for url, depth in batch
                ])

        print(f"\n=== Done. {self._pages_crawled} pages crawled ===")
        print(f"Total unique entities: {len(self.tagger._entities)}")

        # Final export
        self.store.export_json("3xb_internet_graph.json")

        print("\nTop 20 entities by weight:")
        for e in self.store.top_entities(20):
            print(f"  [{e['tag']:25s}] {e['name']:40s} weight={e['weight']:.4f}  freq={e['frequency']}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Seed URLs — broad coverage across news, finance, gov, tech, science
    SEEDS = [
        # encyclopedic
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://en.wikipedia.org/wiki/Portal:Current_events",
        # tech / startup
        "https://news.ycombinator.com",
        "https://techcrunch.com",
        "https://theverge.com",
        "https://wired.com",
        "https://arstechnica.com",
        "https://venturebeat.com",
        # news
        "https://reuters.com",
        "https://apnews.com",
        "https://bbc.com/news",
        "https://npr.org/sections/news",
        "https://politico.com",
        "https://thehill.com",
        # finance / business
        "https://finance.yahoo.com",
        "https://marketwatch.com",
        "https://bloomberg.com",
        "https://wsj.com",
        "https://fortune.com",
        "https://sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K",
        # science / research
        "https://nature.com/news",
        "https://sciencedaily.com",
        "https://arxiv.org/list/cs.AI/recent",
        # government / public records
        "https://congress.gov",
        "https://whitehouse.gov/news",
        "https://federalregister.gov",
        # people / companies
        "https://crunchbase.com/discover/organization",
        "https://pitchbook.com",
        "https://forbes.com/lists/billionaires",
    ]

    tagger = InternetTagger(
        seed_urls=SEEDS,
        max_pages=5000,
        max_depth=3,
        concurrency=15,
        same_domain_only=False,
        db_path=os.path.join(BASE_DIR, "3xb_internet.db"),
    )

    asyncio.run(tagger.run())
