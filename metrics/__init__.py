"""Vendored third-party metric implementations used by `detect.py`.

Kept out of root and away from pip dependencies (the upstream packages either
have aggressive `==` pins that downgrade the env, or aren't on PyPI at all).

  * pate_local        — PATE / PATE-F1 (Ghorbani et al., NeurIPS 2024)
  * affiliation_local — affiliation precision/recall/F1 (Huet et al., KDD 2022)
  * vus_local         — VUS-ROC / VUS-PR (Paparrizos et al., VLDB 2022)
"""
