"""Cross-cutting infrastructure helpers shared across the codebase.

These are pure plumbing (NOT domain code like data/model/metrics): imported as
`from lib.<mod> import ...` by the pipeline scripts AND the comparison wrappers,
which each put the project root on sys.path. Kept here (rather than the project
root) so the root holds entry points + domain modules only.

  * lib.proctitle  — human-readable process titles (ps/htop/nvidia-smi)
  * lib.profiling  — opt-in torch.profiler wrapper (TVQ_PROFILE=1)
"""
