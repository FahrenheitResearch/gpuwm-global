#!/bin/bash
# top-level analysis TMP (0.4-5 hPa) from GDAS f000, byte ranges from the idx
for d in 20261015 20261016 20261017 20261018 20261019 20261020; do
  u=https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.$d/00/atmos/gdas.t00z.pgrb2.0p25.f000
  curl -s -m 60 $u.idx > $d.idx
  out=tmp-$d.grb; : > $out
  awk -F: "{print \$1, \$2, \$4, \$5}" $d.idx > $d.tab
  n=$(wc -l < $d.tab)
  for lev in "0.4 mb" "0.7 mb" "1 mb" "2 mb" "3 mb" "5 mb"; do
    line=$(grep -n ":TMP:$lev:anl" $d.idx | head -1 | cut -d: -f1)
    start=$(sed -n "${line}p" $d.idx | cut -d: -f2)
    end=$(( $(sed -n "$((line+1))p" $d.idx | cut -d: -f2) - 1 ))
    curl -s -m 120 -r $start-$end $u >> $out
  done
  echo $d $(stat -c %s $out)
done
