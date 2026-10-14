#!/usr/bin/env python3
"""Global schedule and lifecycle verifier for the schedule invariants (I1)-(I3).

Supplemental Lemmas "Paths between primitives" and "Resource-boundary
separation" hold for every schedule that satisfies three invariants: cell
locality (I1), padding (I2) and dedication (I3).  This script builds one
complete schedule by the stated rules, exports it, and checks the invariants
and their graph consequences on it.

The schedule
------------
A serial schedule for one level of magic-state distillation feeding one data
T gate, on a fixed layout of cells:

  * 16 Steane 7-to-1 runs produce |Y> resources (112 raw |Y> candidates);
  * one punctured Reed-Muller 15-to-1 run produces one |A> (15 raw |A>
    candidates), each T teleportation in it followed by an adaptive S slot
    that either consumes a distilled |Y> or idles;
  * the |A> output is moved to the data block and consumed by a T
    teleportation, followed by an adaptive S slot.

Every raw candidate is an injection site with its own courier chain: a bare
qubit, a courier block B0 of distance d0 and a courier block B1 of distance
d1, raw injection into B0, transfer B0 -> B1, terminal transfer B1 -> factory
block.  Two courier lanes per factory alternate, so courier cells are reused
by later sites and courier chains run while the factory blocks work.  Births
and discards are primitives too.  Every primitive starts at the earliest
round allowed by the padding of its blocks, so independent primitives run at
the same time.

The checks
----------
From the exported trace alone the verifier generates detector rows by the
catalog templates (T0 consecutive rounds of a live block, T3 entry, deformed
coupling and grid rows, T4 exit, T5 birth and discard rows), one vertex per
owner (block, bare qubit or grid) and round, and checks:

  lifecycle   each cell hosts at most one live block per round; primitives
              act only on live owners; grids live inside their windows;
  geometry    cells are disjoint boxes; every routing ball (radius r_star)
              of an idle block stays in its cell and every grid box with its
              routing ball stays in the union of the participating cells;
  slots       a factory slot holds at most one resource at a time, and every
              terminal transfer and consumption serves exactly one site;
  (I1)        every row lies in one block or in the region of one primitive,
              for every branch assignment tested and for the union graph;
  (I2)        windows of distinct primitives on a block are at least
              R_pad(B) apart, and R_pad(B) > tau;
  (I3)        no primitive involves dedicated owners of two sites;
  SV.19       in the union graph, the region of every primitive P is at
              distance at least min R_pad(B)/tau (B in P) from every other
              primitive region;
  SV.25       neighborhoods A_iota of distinct sites are at distance at least
              R/tau; the exact minimum is reported.

Nine negative controls each break one rule and must be detected.

Outputs: schedule_trace.json (the exported schedule) and
schedule_verifier_report.json.  Scope: the rows are coarse (one vertex per
owner and round); the checks certify this schedule and the rules that produce
it, not the arbitrary-size theorems.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ======================================================================= trace
class Trace:
    def __init__(self, R, dcour, S=12, margin=2):
        self.R, self.dcour, self.S, self.margin = R, list(dcour), S, margin
        self.cells = {}            # name -> (x, y, z)
        self.owners = {}           # name -> dict(kind, cell, site, rpad, birth, death)
        self.prims = []            # dicts
        self.last_end = {}         # block -> end of its last window
        self.cell_free = defaultdict(int)
        self.slots = defaultdict(dict)   # block -> {slot: occupant}
        self.n = 0

    # ----------------------------------------------------------- building
    def cell(self, name, xyz):
        self.cells[name] = tuple(xyz)

    def _new(self, kind, blocks, s, e, site=None, served=None, grid=True,
             branch=False, label=""):
        pid = f"P{len(self.prims)}"
        g = f"g{pid}" if grid else None
        if g:
            self.owners[g] = dict(kind="grid", cell=None, site=None, rpad=None,
                                  birth=s + 1, death=e - 1)
        p = dict(id=pid, kind=kind, blocks=list(blocks), window=[s, e],
                 grid=g, site=site, served=served, branch=branch, label=label)
        self.prims.append(p)
        for b in blocks:
            if self.owners[b]["kind"] == "block":
                self.last_end[b] = e
        return p

    def birth(self, b, cell, rpad, site=None, after=0, kind="full"):
        s = max(after, self.cell_free[cell])
        self.owners[b] = dict(kind="block", cell=cell, site=site, rpad=rpad,
                              birth=s, death=None, role=kind)
        self.cell_free[cell] = 10 ** 9            # busy until the discard
        return self._new("birth", [b], s, s + 1, site=site, grid=False)

    def discard(self, b, after=0):
        o = self.owners[b]
        s = max(after, self.last_end[b] + o["rpad"])
        p = self._new("discard", [b], s, s + 1, site=o["site"], grid=False)
        o["death"] = s + 1
        self.cell_free[o["cell"]] = s + 2
        return p

    def gauge(self, kind, blocks, rounds, site=None, served=None, after=0,
              branch=False, label=""):
        s = after
        for b in blocks:
            o = self.owners[b]
            if o["kind"] == "block":
                s = max(s, self.last_end[b] + o["rpad"])
        return self._new(kind, blocks, s, s + rounds + 1, site=site,
                         served=served, branch=branch, label=label)

    def bare(self, q, cell, site, at):
        self.owners[q] = dict(kind="bare", cell=cell, site=site, rpad=None,
                              birth=at, death=None)

    # ------------------------------------------------------------- slots
    def put(self, block, item):
        sl = self.slots[block]
        k = 0
        while k in sl:
            k += 1
        sl[k] = item
        return k

    def take(self, block, item):
        sl = self.slots[block]
        for k, v in list(sl.items()):
            if v == item:
                del sl[k]
                return k
        raise KeyError(item)


def courier_chain(T, site, lane, factory):
    """Raw injection, one courier transfer, terminal transfer into `factory`."""
    d0, d1 = T.dcour
    q, b0, b1 = f"q{site}", f"B0_{site}", f"B1_{site}"
    T.birth(b0, lane[0], d0, site=site, kind="courier")
    s_inj = T.last_end[b0] + d0
    T.bare(q, lane[0], site, s_inj)
    p = T.gauge("raw_injection", [q, b0], d0, site=site)
    T.owners[q]["death"] = p["window"][1]
    T.birth(b1, lane[1], d1, site=site, kind="courier")
    T.gauge("courier_transfer", [b0, b1], d0, site=site)
    T.discard(b0)
    p = T.gauge("terminal_transfer", [b1, factory], T.R, site=site)
    slot = T.put(factory, site)
    p["slot"] = slot
    T.discard(b1)
    return slot


def build(R=5, dcour=(2, 3), n_y_runs=16, n_cliff=3, S=12, margin=2):
    T = Trace(R, dcour, S, margin)
    # ---------------------------------------------------------------- layout
    lay = {"FY": (0, 0, 0), "GY": (1, 0, 0), "FA": (1, 1, 0), "GA": (2, 1, 0),
           "D": (3, 1, 0),
           "Y1c0": (-2, 0, 0), "Y1c1": (-1, 0, 0),
           "Y2c0": (0, -2, 0), "Y2c1": (0, -1, 0),
           "A1c0": (1, 3, 0), "A1c1": (1, 2, 0),
           "A2c0": (-1, 1, 0), "A2c1": (0, 1, 0)}
    for k, v in lay.items():
        T.cell(k, v)
    for b in ("FY", "GY", "FA", "GA", "D"):
        T.birth(b, b, R)
    ylanes = [("Y1c0", "Y1c1"), ("Y2c0", "Y2c1")]
    alanes = [("A1c0", "A1c1"), ("A2c0", "A2c1")]
    site = 0
    # ------------------------------------------------- Steane 7-to-1 runs
    for run in range(n_y_runs):
        for _ in range(n_cliff):
            T.gauge("clifford", ["FY", "GY"], R, label=f"Y{run} encoder")
        for c in range(7):
            courier_chain(T, site, ylanes[site % 2], "FY")
            p = T.gauge("consume", ["FY", "GY"], R, served=site,
                        label=f"Y{run} teleport")
            p["slot"] = T.take("FY", site)
            T.gauge("readout", ["GY"], R, label=f"Y{run} X readout")
            site += 1
        for _ in range(n_cliff):
            T.gauge("clifford", ["FY", "GY"], R, label=f"Y{run} decoder")
        T.gauge("move", ["FY", "GY"], R, label=f"Y{run} output to GY")
        T.put("GY", f"Y{run}")
    # the sixteenth |Y> waits in GA for the data S slot
    T.gauge("move", ["GY", "FA"], R, label="Y15 to FA")
    T.gauge("move", ["FA", "GA"], R, label="Y15 to GA")
    T.take("GY", f"Y{n_y_runs - 1}")
    T.put("GA", f"Y{n_y_runs - 1}")
    # ------------------------------------------------ Reed-Muller 15-to-1
    for _ in range(n_cliff):
        T.gauge("clifford", ["FA", "GA"], R, label="A encoder")
    for c in range(15):
        courier_chain(T, site, alanes[site % 2], "FA")
        p = T.gauge("consume", ["FA", "GA"], R, served=site, label="A teleport")
        p["slot"] = T.take("FA", site)
        T.gauge("readout", ["GA"], R, label="A X readout")
        T.gauge("s_slot", ["GY", "FA"], R, branch=True, label=f"S slot {c}")
        T.take("GY", f"Y{c}")
        site += 1
    for _ in range(n_cliff):
        T.gauge("clifford", ["FA", "GA"], R, label="A decoder")
    # ------------------------------------------------------ data T gate
    T.gauge("move", ["FA", "GA"], R, label="A output to GA")
    T.gauge("consume_data", ["D", "GA"], R, label="T teleport into data")
    T.gauge("readout", ["D"], R, label="data X readout")
    T.gauge("s_slot", ["GA"], R, branch=True, label="data S slot")
    end = max(p["window"][1] for p in T.prims) + R
    for b in ("FY", "GY", "FA", "GA", "D"):
        T.discard(b, after=end)
    return T


def export(T):
    return dict(R=T.R, dcour=T.dcour, cell_side=T.S, margin=T.margin,
                cells=T.cells, owners=T.owners, prims=T.prims)


# ==================================================================== verify
def box(xyz, S, shrink=0):
    return tuple((c * S + shrink, (c + 1) * S - shrink) for c in xyz)


def inside(a, b):
    return all(b[i][0] <= a[i][0] and a[i][1] <= b[i][1] for i in range(3))


def expand(a, r):
    return tuple((lo - r, hi + r) for lo, hi in a)


def bbox(boxes):
    return tuple((min(b[i][0] for b in boxes), max(b[i][1] for b in boxes)) for i in range(3))


def vol(a):
    v = 1
    for lo, hi in a:
        v *= hi - lo
    return v


def alive(o, t):
    return o["birth"] <= t <= o["death"]


def rows_of(tr, branch_choice):
    """Detector rows from the templates.  branch_choice(pid) -> 'apply'/'idle'."""
    rows = []
    own = tr["owners"]
    for name, o in own.items():
        if o["kind"] != "block":
            continue
        for t in range(o["birth"], o["death"]):
            rows.append((("B", name), [(name, t), (name, t + 1)]))      # T0
        rows.append((("B", name), [(name, o["birth"])]))                # T5 birth
        rows.append((("B", name), [(name, o["death"])]))                # T5 discard
    for p in tr["prims"]:
        if not p["grid"]:
            continue
        if p["branch"] and branch_choice(p["id"]) == "idle":
            continue
        s, e = p["window"]
        g = p["grid"]
        tag = ("P", p["id"])
        for b in p["blocks"]:
            rows.append((tag, [(b, s), (b, s + 1), (g, s + 1)]))        # T3 entry
            for t in range(s + 1, e):
                rows.append((tag, [(b, t), (g, t)]))                    # deformed
            rows.append((tag, [(g, e - 1), (b, e)]))                    # T4 exit
        for t in range(s + 1, e - 1):
            rows.append((tag, [(g, t), (g, t + 1)]))                    # grid T0
    return rows + tr.get("extra_rows", [])


def verify(tr, r_star=2, branch_samples=4, seed=1):
    own, prims, S, M, R = tr["owners"], tr["prims"], tr["cell_side"], tr["margin"], tr["R"]
    cells = {k: tuple(v) for k, v in tr["cells"].items()}
    fails = defaultdict(list)
    pid = {p["id"]: p for p in prims}

    # ---------------------------------------------------------- lifecycle
    occ = defaultdict(list)
    for name, o in own.items():
        if o["kind"] == "block":
            occ[o["cell"]].append((o["birth"], o["death"], name))
    for c, iv in occ.items():
        iv.sort()
        for (a0, a1, n1), (b0, b1, n2) in zip(iv, iv[1:]):
            if b0 <= a1:
                fails["lifecycle"].append(f"cell {c}: {n2} born at {b0} while {n1} lives to {a1}")
    for p in prims:
        s, e = p["window"]
        for b in p["blocks"]:
            o = own[b]
            if not (o["birth"] <= s and e <= o["death"]):
                fails["lifecycle"].append(f"{p['id']} uses {b} outside its life")
        if p["grid"]:
            g = own[p["grid"]]
            if not (s < g["birth"] and g["death"] < e):
                fails["lifecycle"].append(f"{p['id']} grid lives outside the window")

    # ----------------------------------------------------------- geometry
    cb = {c: box(xyz, S) for c, xyz in cells.items()}
    for a, b in combinations(cells, 2):
        A, B = cb[a], cb[b]
        if all(A[i][0] < B[i][1] and B[i][0] < A[i][1] for i in range(3)):
            fails["geometry"].append(f"cells {a},{b} overlap")
    for name, o in own.items():
        if o["kind"] == "block":
            interior = box(cells[o["cell"]], S, M)
            if not inside(expand(interior, r_star), cb[o["cell"]]):
                fails["geometry"].append(f"routing ball of {name} leaves its cell")
    for p in prims:
        if not p["grid"]:
            continue
        bcells = [own[b]["cell"] for b in p["blocks"] if own[b]["kind"] == "block"]
        ubox = bbox([cb[c] for c in set(bcells)])
        if vol(ubox) != sum(vol(cb[c]) for c in set(bcells)):
            fails["geometry"].append(f"{p['id']} ({p['label'] or p['kind']}) joins non-adjacent cells {sorted(set(bcells))}")
            continue
        grid = bbox([box(cells[c], S, M) for c in bcells])
        if not inside(expand(grid, r_star), ubox):
            fails["geometry"].append(f"{p['id']} grid routing leaves its cells")
        for b in p["blocks"]:
            if own[b]["kind"] == "bare" and own[b]["cell"] not in bcells:
                fails["geometry"].append(f"{p['id']} bare qubit outside the cells")

    # -------------------------------------------------------------- slots
    for p in prims:
        if p["kind"] in ("terminal_transfer", "raw_injection", "courier_transfer") and p["site"] is None:
            fails["slots"].append(f"{p['id']} serves no site")
        if p["kind"] == "consume" and p["served"] is None:
            fails["slots"].append(f"{p['id']} consumes no site")

    occupy = defaultdict(list)          # (block, slot) -> [(start, end, site)]
    term = {p["site"]: p for p in prims if p["kind"] == "terminal_transfer"}
    for p in prims:
        if p["kind"] == "consume":
            t = term.get(p["served"])
            f = [b for b in p["blocks"] if t and b in t["blocks"]]
            if not t or not f or t.get("slot") != p.get("slot"):
                fails["slots"].append(f"{p['id']} consumes a resource that was not delivered to its slot")
                continue
            occupy[(f[0], p["slot"])].append((t["window"][1], p["window"][0], p["served"]))
    for key, iv in occupy.items():
        iv.sort()
        for (a0, a1, s1), (b0, b1, s2) in zip(iv, iv[1:]):
            if b0 <= a1:
                fails["slots"].append(f"slot {key}: site {s2} delivered while site {s1} still waits")

    # ---------------------------------------------------------------- I3
    def sites_of(p):
        s = {own[b]["site"] for b in p["blocks"] if own[b]["kind"] in ("bare", "block")
             and own[b]["site"] is not None}
        if p.get("served") is not None:
            s.add(p["served"])
        if p.get("site") is not None:
            s.add(p["site"])
        return s
    for p in prims:
        if len(sites_of(p)) > 1:
            fails["I3"].append(f"{p['id']} ({p['kind']}) involves sites {sorted(sites_of(p))}")

    # ---------------------------------------------------------------- I1
    rng = random.Random(seed)
    branch_ids = [p["id"] for p in prims if p["branch"]]
    choices = [("all apply", lambda i: "apply"), ("all idle", lambda i: "idle")]
    for k in range(branch_samples):
        pick = {i: rng.choice(("apply", "idle")) for i in branch_ids}
        choices.append((f"random {k}", lambda i, pick=pick: pick[i]))

    def check_rows(rows, label):
        bad = 0
        for tag, mech in rows:
            for (o, t) in mech:
                if not alive(own[o], t):
                    bad += 1
                    break
            else:
                if tag[0] == "B":
                    ok = all(o == tag[1] for o, _ in mech)
                else:
                    p = pid[tag[1]]
                    s, e = p["window"]
                    allowed = set(p["blocks"]) | {p["grid"]}
                    ok = all(o in allowed and s <= t <= e for o, t in mech)
                if not ok:
                    bad += 1
        if bad:
            fails["I1"].append(f"{label}: {bad} rows outside one block or one primitive region")
    n_rows = {}
    for label, ch in choices:
        rows = rows_of(tr, ch)
        n_rows[label] = len(rows)
        check_rows(rows, label)
    union = rows_of(tr, lambda i: "apply")      # idle adds no rows
    check_rows(union, "union")

    # ------------------------------------------------------- union graph
    adj = defaultdict(set)
    tau = 0
    for _, mech in union:
        for u, v in combinations(mech, 2):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
                tau = max(tau, abs(u[1] - v[1]))

    # ---------------------------------------------------------------- I2
    wins = defaultdict(list)
    for p in prims:
        for b in p["blocks"]:
            if own[b]["kind"] == "block":
                wins[b].append(tuple(p["window"]))
    min_sep_full, min_sep_cour = 10 ** 9, 10 ** 9
    for b, ws in wins.items():
        rp = own[b]["rpad"]
        if not rp > tau:
            fails["I2"].append(f"{b}: R_pad={rp} is not larger than tau={tau}")
        ws.sort()
        for (s0, e0), (s1, e1) in zip(ws, ws[1:]):
            gap = s1 - e0
            if gap < rp:
                fails["I2"].append(f"{b}: windows {s0}-{e0} and {s1}-{e1} only {gap} apart (< {rp})")
            if own[b]["role"] == "courier":
                min_sep_cour = min(min_sep_cour, gap)
            else:
                min_sep_full = min(min_sep_full, gap)

    # ------------------------------------------------------ regions, SV.19
    def region(p):
        s, e = p["window"]
        out = set()
        for o in p["blocks"] + ([p["grid"]] if p["grid"] else []):
            for t in range(s, e + 1):
                if alive(own[o], t):
                    out.add((o, t))
        return out
    reg = {p["id"]: region(p) for p in prims}
    owner_prim = {}
    for i, rg in reg.items():
        for v in rg:
            if v in owner_prim:
                fails["I2"].append(f"regions of {owner_prim[v]} and {i} overlap at {v}")
            owner_prim[v] = i
    sv19_viol, sv19_min_ratio = 0, None
    for p in prims:
        bl = [own[b]["rpad"] for b in p["blocks"] if own[b]["kind"] == "block"]
        need = min(bl) / max(tau, 1)
        depth = int(need) + 1
        seen = {v: 0 for v in reg[p["id"]]}
        dq = deque(reg[p["id"]])
        while dq:
            v = dq.popleft()
            if seen[v] >= depth:
                continue
            for w in adj[v]:
                if w not in seen:
                    seen[w] = seen[v] + 1
                    q = owner_prim.get(w)
                    if q is not None and q != p["id"]:
                        if seen[w] < need:
                            sv19_viol += 1
                        r = seen[w] / need
                        sv19_min_ratio = r if sv19_min_ratio is None else min(sv19_min_ratio, r)
                        continue
                    dq.append(w)
    if sv19_viol:
        fails["SV.19"].append(f"{sv19_viol} primitive pairs closer than the padding bound")

    # ------------------------------------------------------------- SV.25
    anchor = {}
    sites = sorted({o["site"] for o in own.values() if o["site"] is not None})
    for name, o in own.items():
        if o["site"] is not None and o["kind"] in ("bare", "block"):
            for t in range(o["birth"], o["death"] + 1):
                anchor[(name, t)] = o["site"]
    for p in prims:
        if p["kind"] in ("raw_injection", "courier_transfer", "terminal_transfer"):
            for v in reg[p["id"]]:
                if anchor.get(v, p["site"]) != p["site"]:
                    fails["SV.25"].append(f"anchors of two sites share {v}")
                anchor[v] = p["site"]
    dist, lab = {}, {}
    dq = deque()
    for v, s in anchor.items():
        dist[v], lab[v] = 0, s
        dq.append(v)
    best = None
    while dq:
        v = dq.popleft()
        for w in adj[v]:
            if w not in dist:
                dist[w], lab[w] = dist[v] + 1, lab[v]
                dq.append(w)
            elif lab[w] != lab[v]:
                d = dist[v] + dist[w] + 1
                best = d if best is None else min(best, d)
    if best is not None and best < R / max(tau, 1):
        fails["SV.25"].append(f"two site neighborhoods only {best} edges apart (< R/tau = {R / tau})")

    # concurrency and reuse, reported to show the cases are present
    conc = 0
    gp = [p for p in prims if p["grid"]]
    for a, b in combinations(gp, 2):
        if not set(a["blocks"]) & set(b["blocks"]):
            if a["window"][0] <= b["window"][1] and b["window"][0] <= a["window"][1]:
                conc += 1
    reuse = sum(len(v) - 1 for v in occ.values())
    stats = dict(cells=len(cells), blocks=sum(o["kind"] == "block" for o in own.values()),
                 primitives=len(prims), adaptive_slots=len(branch_ids), sites=len(sites),
                 rounds=max(p["window"][1] for p in prims) + 1,
                 vertices=len(adj), rows_union=len(union),
                 rows_per_branch_choice=n_rows, tau=tau,
                 min_window_gap_full_distance=min_sep_full,
                 min_window_gap_courier=min_sep_cour,
                 min_site_separation_edges=best, required=R / max(tau, 1),
                 sv19_min_distance_over_bound=sv19_min_ratio,
                 concurrent_primitive_pairs_on_disjoint_blocks=conc,
                 cell_reuses=reuse)
    return dict(passed=not fails, failures={k: v[:5] for k, v in fails.items()},
                failure_counts={k: len(v) for k, v in fails.items()}, stats=stats)


# ========================================================= negative controls
def controls(T):
    base = export(T)
    out = []

    def run(name, tr, expect, **kw):
        res = verify(tr, **kw)
        hit = [k for k in expect if res["failure_counts"].get(k)]
        out.append(dict(control=name, expected=expect, detected=hit,
                        ok=bool(hit) and len(hit) == len(expect),
                        failures=res["failure_counts"]))

    # N1: two terminal transfers on one factory block without the padding
    tr = copy.deepcopy(base)
    terms = [p for p in tr["prims"] if p["kind"] == "terminal_transfer" and "FY" in p["blocks"]]
    a, b = terms[3], terms[4]
    shift = b["window"][0] - (a["window"][1] + 1)
    b["window"] = [b["window"][0] - shift, b["window"][1] - shift]
    g = tr["owners"][b["grid"]]
    g["birth"] -= shift
    g["death"] -= shift
    run("N1 terminal transfers on FY one round apart", tr, ["I2", "SV.25"])
    # N2: a courier transfer of one site uses another site's courier block
    tr = copy.deepcopy(base)
    p = next(p for p in tr["prims"] if p["kind"] == "courier_transfer" and p["site"] == 10)
    p["blocks"][1] = "B1_9"
    run("N2 courier block shared by two sites", tr, ["I3"])
    # N3: a courier block born in its cell before the previous one is discarded
    tr = copy.deepcopy(base)
    o_prev, o_next = tr["owners"]["B0_20"], tr["owners"]["B0_22"]
    o_next["birth"] = o_prev["death"] - 1
    run("N3 cell reused before the discard", tr, ["lifecycle"])
    # N4: a move between cells that are not adjacent
    tr = copy.deepcopy(base)
    p = next(p for p in tr["prims"] if p["label"] == "Y15 to FA")
    p["blocks"] = ["GY", "D"]
    run("N4 two-block primitive across non-adjacent cells", tr, ["geometry"])
    # N5: routing radius larger than the cell margin
    run("N5 routing radius 3 with margin 2", copy.deepcopy(base), ["geometry"], r_star=3)
    # N6: a detector row joining two blocks outside any primitive
    tr = copy.deepcopy(base)
    t0 = tr["owners"]["D"]["birth"] + 3
    tr["extra_rows"] = [(("B", "FY"), [("FY", t0), ("D", t0)])]
    run("N6 row joining FY and D", tr, ["I1"])
    # N7: courier padding not larger than tau
    run("N7 courier distances (1, 2)", export(build(R=T.R, dcour=(1, 2))), ["I2"])
    # N8: an adaptive slot whose applied branch touches a block outside the slot
    tr = copy.deepcopy(base)
    p = next(p for p in tr["prims"] if p["kind"] == "s_slot")
    tr["extra_rows"] = [(("P", p["id"]), [(p["grid"], p["window"][0] + 1),
                                           ("D", p["window"][0] + 1)])]
    run("N8 applied branch reaching block D", tr, ["I1"])
    # N9: a consumption that reads a slot the resource was never delivered to
    tr = copy.deepcopy(base)
    p = next(p for p in tr["prims"] if p["kind"] == "consume" and p["served"] == 30)
    p["slot"] = p["slot"] + 1
    run("N9 consumption from the wrong factory slot", tr, ["slots"])
    return out


def main():
    T = build()
    tr = export(T)
    (ROOT / "schedule_trace.json").write_text(json.dumps(tr, indent=0), encoding="utf-8")
    res = verify(tr)
    ctl = controls(T)
    report = dict(schedule=res, negative_controls=ctl)
    (ROOT / "schedule_verifier_report.json").write_text(json.dumps(report, indent=1),
                                                        encoding="utf-8")
    print(json.dumps(res, indent=1))
    for c in ctl:
        print(f"{c['control']:52s} expected {c['expected']}, detected {c['detected']}, "
              f"{'OK' if c['ok'] else 'MISSED'}")
    assert res["passed"], res["failures"]
    assert all(c["ok"] for c in ctl), ctl
    print("schedule passes every check; every negative control is detected")


if __name__ == "__main__":
    main()
