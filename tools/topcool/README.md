# Top-level drift tools (2026-11-06)

- `../arwen_global_top_budget.py run CONFIG --outdir RUN`: the run with read-only per-process
  accumulators of the top 6 levels (global and zonal), `$TOPBUDGET_OUT` JSON at exit.
- `fetch_gdas_top_tmp.sh`, `fetch_gdas_top_ugrd.sh`: GDAS f000 TMP (0.4-5 hPa) and UGRD (1-10 hPa)
  by idx byte ranges from noaa-gfs-bdp-pds, for each verifying day.
- `anlzon.py DATE...`: analysis zonal-mean theta at the level-0..2 mid pressures, 10-degree bands.
- `drift.py BUDGET ANL_FLDMEAN [steps_per_day]`: global-mean top-level drift table against GDAS.
- `zerr.py ANL_ZONAL DAY0 SPD RUN...`: zonal-mean theta error growth (band RMS) per day.
- `zuerr.py DAY0 SPD RUN...`: zonal-mean u error against GDAS per day (top 5 levels).
- `charts.py OUTDIR label=run ...`: the analysis charts (matplotlib; not weather fields).
