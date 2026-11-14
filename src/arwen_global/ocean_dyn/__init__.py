"""The dynamic ocean (stage 1): a hydrostatic Boussinesq primitive-equation
ocean on z* levels, Arakawa C grid, split-explicit free surface, KPP mixed
layer, GM eddy transport, Wright (1997) equation of state and two-band
shortwave penetration.

Design: docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md.  Off by default: nothing here is
imported unless ``[ocean] mode = "dynamic"`` (or an ``ocean`` subcommand).

Every module is array arithmetic on the module it is handed (numpy on the
host, cupy on the card), one code path for both, so the CPU tests exercise
the same arithmetic the card runs.  Written from the papers each module
names; no source is transcribed from MOM6, NEMO or POP.
"""
