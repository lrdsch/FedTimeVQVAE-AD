"""Per-dataset loaders.

Each module here exposes ``load_<name>_records(cfg, split)`` returning a
``list[TimeSeriesRecord]``. The dispatcher in ``data.py:load_records`` imports
from this package lazily.
"""
