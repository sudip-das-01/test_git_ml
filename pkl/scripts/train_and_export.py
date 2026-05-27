"""Backward-compatible entry: runs create_model + inference via run_benchmark."""

from run_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
