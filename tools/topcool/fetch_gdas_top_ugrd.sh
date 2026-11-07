#!/bin/bash
# zonal wind analysis at 1-10 hPa (GDAS f000), byte ranges from the idx
cd "${1:-.}"  # the directory holding the idx files fetch_gdas_top_tmp.sh wrote
for d in 20261015 20261016 20261017 20261018 20261019 20261020; do
  u=https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.$d/00/atmos/gdas.t00z.pgrb2.0p25.f000
  out=ugrd-$d.grb; : > $out
  for lev in "1 mb" "2 mb" "3 mb" "5 mb" "10 mb"; do
    line=$(grep -n ":UGRD:$lev:anl" $d.idx | head -1 | cut -d: -f1)
    start=$(sed -n "${line}p" $d.idx | cut -d: -f2)
    end=$(( $(sed -n "$((line+1))p" $d.idx | cut -d: -f2) - 1 ))
    curl -s -m 120 -r $start-$end $u >> $out
  done
  echo $d $(stat -c %s $out)
done
