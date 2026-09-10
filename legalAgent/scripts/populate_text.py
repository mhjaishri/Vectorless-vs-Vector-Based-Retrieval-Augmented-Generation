"""
Recovery script: populate missing text fields in tree JSON files using the PDFs.

The tree files were generated without if_add_node_text: "yes", so all nodes
have start_index/end_index but no text field. This script reads the PDF pages
for each node's range and fills in the text, then saves to the correct filenames.

Run from project root:
    python3 scripts/populate_text.py
"""

import os
import sys
import json

PAGEINDEX_REPO = os.environ.get("PAGEINDEX_REPO", "/tmp/pageindex")
if PAGEINDEX_REPO not in sys.path:
    sys.path.insert(0, PAGEINDEX_REPO)

import fitz  # pymupdf

INDEXES_DIR = "indexes"

# Input files (user-placed) → output files (code expects) → PDF source
DOCS = [
    ("constitution_tree.json", "constitution.tree.json", "constitution.pdf"),
    ("bns_tree.json",          "bns.tree.json",          "bns.pdf"),
    ("bnss_tree.json",         "bnss.tree.json",          "bnss.pdf"),
]


def get_pdf_pages(pdf_path: str) -> list[str]:
    """Extract text from every page of a PDF. Returns list indexed by page (0-based)."""
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]


def populate_text(nodes: list, pages: list[str], doc_name: str) -> tuple[int, int]:
    """Recursively fill text fields. Returns (filled, total) counts."""
    filled = 0
    total = 0
    for node in nodes:
        total += 1
        if not node.get("text", "").strip():
            start = node.get("start_index", 1)
            end = node.get("end_index", start)
            # Handle inverted ranges (data quirk in some generated trees)
            if start > end:
                start, end = end, start
            # start_index/end_index are 1-based page numbers
            text_parts = []
            for page_num in range(start, end + 1):
                idx = page_num - 1  # convert to 0-based
                if 0 <= idx < len(pages):
                    text_parts.append(pages[idx])
            node["text"] = "\n".join(text_parts)
            filled += 1
        if node.get("nodes"):
            child_filled, child_total = populate_text(node["nodes"], pages, doc_name)
            filled += child_filled
            total += child_total
    return filled, total


def main() -> None:
    print("Populating text fields from PDFs...\n")

    for input_file, output_file, pdf_file in DOCS:
        input_path = os.path.join(INDEXES_DIR, input_file)
        output_path = os.path.join(INDEXES_DIR, output_file)

        if not os.path.exists(input_path):
            print(f"  SKIP {input_file} — not found")
            continue

        if not os.path.exists(pdf_file):
            print(f"  SKIP {input_file} — PDF {pdf_file} not found")
            continue

        print(f"  [{input_file}] Loading PDF {pdf_file}...")
        pages = get_pdf_pages(pdf_file)
        print(f"  [{input_file}] {len(pages)} pages loaded")

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        structure = data.get("structure", [])
        filled, total = populate_text(structure, pages, input_file)
        print(f"  [{input_file}] Filled {filled}/{total} nodes with text")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        size_kb = os.path.getsize(output_path) // 1024
        print(f"  [{input_file}] Saved to {output_path} ({size_kb} KB)\n")

    print("Done. Run: npx tsx src/index.ts")


if __name__ == "__main__":
    main()
