"""
index.py – CLI to (re-)index project documents into FAISS via IndexingWorkflow.

Usage
-----
    python index.py path/to/project1 path/to/project2
    python index.py ./sample_project --watch    # keep index fresh on file changes

Example
-------
    python index.py ./sample_project
"""
import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index .md files into FAISS")
    parser.add_argument(
        "roots",
        nargs="+",
        metavar="PATH",
        help="Project root directories containing .md files.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="After initial index, watch for .md changes and auto-rebuild.",
    )
    args = parser.parse_args()

    print(f"Indexing {len(args.roots)} project root(s): {args.roots}")

    from index_workflow import run_indexing
    asyncio.run(run_indexing(reindex=True, project_roots=args.roots))
    print("✅ Indexing complete.")

    if args.watch:
        import time
        from watcher import start_watcher
        print("👁  Watching for .md changes … (Ctrl+C to stop)")
        observer = start_watcher(project_roots=args.roots)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping watcher…")
            observer.stop()
            observer.join()
    else:
        print("You can now run: python app.py")
