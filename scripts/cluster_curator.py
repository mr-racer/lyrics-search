"""Cluster Curator — interactive CLI for labeling Sonic Descriptor clusters.

Workflow:
  1. Trigger cluster discovery on a collection (or load existing assignments).
  2. For each cluster, the curator shows representatives — track titles, artists.
  3. The user enters a label (or skips). Labels persist after each entry.
  4. After all clusters are labeled (or user types ``train``), MLP classifier training runs.

Run::

    python scripts/cluster_curator.py --collection music_explorer --host http://localhost:8000

The script talks to the running FastAPI server via HTTP — make sure ``uvicorn`` is up.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Tuple

import httpx


# ── Pure logic helpers (testable without network) ───────────────────────────


def format_cluster_summary(cluster: dict) -> str:
    """One-screen summary of a cluster for the operator to read."""
    lines = [
        f"Cluster {cluster['cluster_id']} — {cluster['size']} tracks — "
        f"{cluster.get('current_label') or '(unlabeled)'}",
    ]
    for i, t in enumerate(cluster.get("representative_tracks", []), 1):
        lines.append(f"  [{i}] {t.get('title', '?')} — {t.get('artist', '?')}")
    return "\n".join(lines)


def parse_command(line: str) -> Tuple[str, tuple]:
    """Parse a single line of curator input. Returns (cmd, args_tuple)."""
    line = line.strip()
    if not line:
        return "skip", ()
    parts = line.split(maxsplit=1)
    head = parts[0].lower()
    if head in ("q", "quit", "exit"):
        return "quit", ()
    if head in ("s", "skip"):
        return "skip", ()
    if head in ("t", "train"):
        return "train", ()
    if head in ("h", "help", "?"):
        return "help", ()
    if head in ("l", "label"):
        rest = parts[1] if len(parts) > 1 else ""
        rest_parts = rest.split(maxsplit=1)
        if len(rest_parts) >= 2:
            return "label", (rest_parts[0], rest_parts[1])
        return "unknown", ()
    return "unknown", ()


# ── Network operations ─────────────────────────────────────────────────────


def wait_for_job(client: httpx.Client, job_url: str, label: str) -> Optional[dict]:
    """Poll a job status URL until completion. Returns final result dict or None."""
    print(f"[curator] {label} job started; polling...")
    while True:
        r = client.get(job_url)
        if r.status_code == 404:
            print(f"[curator] {label} job lost (404)")
            return None
        data = r.json()
        status = data.get("status")
        if status == "completed":
            print(f"[curator] {label} done: {data.get('result')}")
            return data.get("result")
        if status == "failed":
            print(f"[curator] {label} failed: {data.get('error')}")
            return None
        time.sleep(1.0)


# ── Main interactive loop ───────────────────────────────────────────────────


HELP_TEXT = """
Commands:
  label <cluster_id> <name>   Assign a name to a cluster (e.g. 'label 3 Lo-fi indie')
  skip                        Skip current cluster
  train                       Finish labeling and trigger MLP training
  quit                        Exit without training
  help                        This message
"""


def main(host: str, collection: str) -> int:
    with httpx.Client(base_url=host, timeout=300.0) as client:
        # Step 1: trigger cluster discovery (unless assignments already exist)
        print(f"[curator] checking existing clusters for '{collection}'...")
        reps = client.get(
            "/api/v1/library/clusters/representatives",
            params={"collection": collection},
        ).json()
        if not reps:
            print("[curator] no existing clusters — triggering discovery")
            r = client.post(
                "/api/v1/library/cluster-discovery",
                params={"collection": collection},
            )
            r.raise_for_status()
            job_id = r.json()["job_id"]
            wait_for_job(client, f"/api/v1/library/cluster-jobs/{job_id}", "discovery")
            reps = client.get(
                "/api/v1/library/clusters/representatives",
                params={"collection": collection},
            ).json()

        if not reps:
            print("[curator] no clusters found after discovery — bailing out")
            return 1

        # Step 2: interactive labeling loop
        print(f"[curator] {len(reps)} clusters to label. Type 'help' for commands.")
        labels: dict[int, str] = {int(r["cluster_id"]): r.get("current_label") for r in reps if r.get("current_label")}
        for cluster in reps:
            cid = int(cluster["cluster_id"])
            if cid in labels and labels[cid]:
                print(f"[curator] skipping already-labeled cluster {cid}: '{labels[cid]}'")
                continue
            print()
            print(format_cluster_summary(cluster))
            while True:
                line = input(f"label for cluster {cid}> ")
                cmd, args = parse_command(line)
                if cmd == "label":
                    target_id, name = args
                    labels[int(target_id)] = name
                    client.post(
                        "/api/v1/library/clusters/labels",
                        json={"collection": collection, "labels": {str(k): v for k, v in labels.items()}},
                    )
                    print(f"  → labeled cluster {target_id} as '{name}'")
                    break
                if cmd == "skip":
                    break
                if cmd == "quit":
                    print("[curator] quitting without training")
                    return 0
                if cmd == "train":
                    break
                if cmd == "help":
                    print(HELP_TEXT)
                    continue
                print("  unknown command — type 'help' for options")

        # Step 3: train classifier
        if not labels:
            print("[curator] no labels assigned — nothing to train")
            return 0
        print(f"[curator] triggering classifier training on {len(labels)} labeled clusters")
        r = client.post(
            "/api/v1/library/sonic-classifier/train",
            params={"collection": collection},
        )
        r.raise_for_status()
        job_id = r.json()["job_id"]
        wait_for_job(client, f"/api/v1/library/sonic-classifier/jobs/{job_id}", "training")
        print("[curator] done!")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--collection", required=True)
    args = ap.parse_args()
    sys.exit(main(host=args.host, collection=args.collection))


if __name__ == "__main__":
    cli()
