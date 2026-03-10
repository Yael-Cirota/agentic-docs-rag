"""
index.py – Standalone script to (re-)index project documents into Pinecone.

Usage
-----
    python index.py path/to/project1 path/to/project2

Example
-------
    python index.py ./sample_project
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index.py <project_root> [<project_root> ...]")
        print("Example: python index.py ./sample_project")
        sys.exit(1)

    roots = sys.argv[1:]
    print(f"Indexing {len(roots)} project root(s): {roots}")

    from pipeline import build_index
    index = build_index(project_roots=roots)
    print("✅ Indexing complete. You can now run: python app.py")
