"""Report repeated openings from a headerless conversation TSV export.

Usage: python scripts/audit_reply_style.py conversation.tsv
Columns match the message export: id, conversation, order, sender, text, ...
This is an offline diagnostic, never a delivery gate or a quality score.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def audit(path):
    conversations = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 5 and row[3] == "backend":
                conversations[row[1]].append((int(row[2]), row[4]))
    output = []
    for conversation, rows in conversations.items():
        openings = [" ".join(re.findall(r"[\w']+", text.lower().replace("’", "'"))[:4])
                    for _, text in sorted(rows)]
        counts = Counter(openings)
        output.append({
            "conversation": conversation,
            "replies": len(rows),
            "repeated_first_four_words": {k: n for k, n in counts.items() if k and n > 1},
            "opening_repeats_within_previous_four_replies": sum(
                bool(value) and value in openings[max(0, i - 4):i]
                for i, value in enumerate(openings)
            ),
        })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    args = parser.parse_args()
    print(json.dumps(audit(args.export), indent=2))
