//! WOOF Global cold start: the analysis's horizontal regrid and vertical
//! remap.
//!
//! Two operations, both moved here from host NumPy so the cold start's data
//! path runs in Rust:
//!
//! * [`regrid`]: periodic bilinear interpolation from a regular global
//!   latitude-longitude grid onto the spectral transform's Gaussian grid.
//! * [`remap`]: linear-in-ln(p) interpolation from ascending isobaric
//!   levels onto model full levels, holding above the top source level and
//!   holding (or, for temperature, continuing the bottom layer's ln(p)
//!   gradient) below the bottom one.
//!
//! PARITY.  Both are byte-identical to the NumPy expressions they replace.
//! Every arithmetic step is the same IEEE-754 double operation in the same
//! order NumPy's element-wise ufuncs evaluate it: the bilinear sum is
//! `(1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)` with
//! every product and sum rounded on its own (Rust never contracts a
//! multiply and an add into one fused instruction unless asked), and the
//! index arithmetic reproduces `numpy.mod`, `numpy.clip`, `astype(int64)`,
//! `numpy.floor` and `numpy.searchsorted` exactly.  Each output element is
//! written by exactly one thread from inputs no thread writes, so the
//! thread count cannot move a bit.

use std::num::NonZeroUsize;

pub mod capi;

/// An input the operation cannot honour, named in the sentence the caller
/// re-raises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn refuse<T>(message: impl Into<String>) -> Result<T, Refusal> {
    Err(Refusal(message.into()))
}

/// `numpy.mod(a, b)` for float64 (numpy's `npy_divmod` remainder): the C
/// `fmod`, moved into the divisor's sign, with a zero carrying the
/// divisor's sign.
#[inline]
pub fn numpy_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if b == 0.0 {
        return m;
    }
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// `numpy.clip(x, lo, hi)` for float64: `min(max(x, lo), hi)` with NumPy's
/// NaN-propagating comparisons.
#[inline]
pub fn numpy_clip(x: f64, lo: f64, hi: f64) -> f64 {
    let m = if x.is_nan() || x > lo { x } else { lo };
    if m.is_nan() || m < hi {
        m
    } else {
        hi
    }
}

/// `numpy.searchsorted(sorted, x, side="left")`: the first index whose
/// value is not less than `x` under NumPy's ordering, which places NaN
/// after every number.
#[inline]
pub fn searchsorted_left(sorted: &[f64], x: f64) -> usize {
    let less = |a: f64, b: f64| a < b || (b.is_nan() && !a.is_nan());
    let (mut lo, mut hi) = (0usize, sorted.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if less(sorted[mid], x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The bilinear weights from a regular global grid onto a target grid.
///
/// Built from the 1-D coordinate vectors exactly as the NumPy regridder
/// built them.  Latitudes may be given north-to-south; they are read
/// ascending and the field's rows are read the same way.
#[derive(Debug, Clone)]
pub struct Regridder {
    nlat: usize,
    nlon: usize,
    flip: bool,
    y0: Vec<usize>,
    wy: Vec<f64>,
    x0: Vec<usize>,
    x1: Vec<usize>,
    wx: Vec<f64>,
}

impl Regridder {
    pub fn new(
        latitude: &[f64],
        longitude: &[f64],
        target_latitude: &[f64],
        target_longitude: &[f64],
    ) -> Result<Self, Refusal> {
        let (nlat, nlon) = (latitude.len(), longitude.len());
        if nlat < 2 || nlon < 2 {
            return refuse("analysis grid must be two-dimensional");
        }
        if target_latitude.is_empty() || target_longitude.is_empty() {
            return refuse("the target grid is empty");
        }
        for (name, values) in [
            ("analysis latitude", latitude),
            ("analysis longitude", longitude),
            ("target latitude", target_latitude),
            ("target longitude", target_longitude),
        ] {
            if values.iter().any(|v| !v.is_finite()) {
                return refuse(format!(
                    "{name} carries a non-finite coordinate, so no interpolation cell can be named for it"
                ));
            }
        }
        let flip = latitude[0] > latitude[nlat - 1];
        let (lat0, dlat) = if flip {
            (latitude[nlat - 1], latitude[nlat - 2] - latitude[nlat - 1])
        } else {
            (latitude[0], latitude[1] - latitude[0])
        };
        let lon0 = longitude[0];
        let dlon = longitude[1] - longitude[0];
        if dlat == 0.0 || dlon == 0.0 {
            return refuse("analysis grid spacing is zero");
        }
        let top = (nlat - 1) as f64;
        let mut y0 = Vec::with_capacity(target_latitude.len());
        let mut wy = Vec::with_capacity(target_latitude.len());
        for &t in target_latitude {
            let fy = numpy_clip((t - lat0) / dlat, 0.0, top);
            // astype(int64) truncates; fy is clipped to [0, nlat - 1].
            let row = (fy as i64).min(nlat as i64 - 2) as usize;
            y0.push(row);
            wy.push(fy - row as f64);
        }
        let mut x0 = Vec::with_capacity(target_longitude.len());
        let mut x1 = Vec::with_capacity(target_longitude.len());
        let mut wx = Vec::with_capacity(target_longitude.len());
        for &t in target_longitude {
            let fx = numpy_mod(t - lon0, 360.0) / dlon;
            let column = (fx as i64).rem_euclid(nlon as i64) as usize;
            x0.push(column);
            x1.push((column + 1) % nlon);
            wx.push(fx - fx.floor());
        }
        Ok(Self { nlat, nlon, flip, y0, wy, x0, x1, wx })
    }

    /// Points in one source plane.
    pub fn source_plane(&self) -> usize {
        self.nlat * self.nlon
    }

    /// Points in one target plane.
    pub fn target_plane(&self) -> usize {
        self.y0.len() * self.x0.len()
    }

    #[inline]
    fn source_row(&self, ascending_row: usize) -> usize {
        if self.flip {
            self.nlat - 1 - ascending_row
        } else {
            ascending_row
        }
    }

    /// One target row of one plane.
    fn row(&self, plane: &[f64], i: usize, out: &mut [f64]) {
        let lower = self.source_row(self.y0[i]) * self.nlon;
        let upper = self.source_row(self.y0[i] + 1) * self.nlon;
        let wy = self.wy[i];
        for (j, slot) in out.iter_mut().enumerate() {
            let (x0, x1, wx) = (self.x0[j], self.x1[j], self.wx[j]);
            let a = plane[lower + x0];
            let b = plane[lower + x1];
            let c = plane[upper + x0];
            let d = plane[upper + x1];
            *slot = (1.0 - wy) * ((1.0 - wx) * a + wx * b) + wy * ((1.0 - wx) * c + wx * d);
        }
    }

    /// Regrid `planes` stacked source planes into `out`.
    pub fn apply(&self, field: &[f64], out: &mut [f64], threads: usize) -> Result<(), Refusal> {
        let (src, tgt) = (self.source_plane(), self.target_plane());
        if field.len() % src != 0 {
            return refuse(format!(
                "the field holds {} values, not a whole number of {}x{} planes",
                field.len(),
                self.nlat,
                self.nlon
            ));
        }
        let planes = field.len() / src;
        if out.len() != planes * tgt {
            return refuse(format!(
                "the output holds {} values where {} planes of {} need {}",
                out.len(),
                planes,
                tgt,
                planes * tgt
            ));
        }
        let ntlon = self.x0.len();
        let ntlat = self.y0.len();
        if ntlon == 0 || planes == 0 {
            return Ok(());
        }
        // Work unit: one target row of one plane.
        let rows = planes * ntlat;
        let work = |first_row: usize, chunk: &mut [f64]| {
            for (offset, out_row) in chunk.chunks_mut(ntlon).enumerate() {
                let r = first_row + offset;
                let (p, i) = (r / ntlat, r % ntlat);
                self.row(&field[p * src..(p + 1) * src], i, out_row);
            }
        };
        run_rows(out, rows, ntlon, threads, work);
        Ok(())
    }
}

/// Linear-in-ln(p) remap of `values` (`nsrc` levels by `npts` points,
/// level-major) from ascending `ln_source` onto `ln_target` (`ntgt` levels
/// by `npts` points), written into `out` (`ntgt` by `npts`).
///
/// Above the top source level the value is held (the weight is floored at
/// zero); below the bottom it is held unless `extrapolate_below`, which
/// lifts the weight's ceiling of one so the bottom layer's ln(p) gradient
/// continues.
pub fn remap(
    values: &[f64],
    ln_source: &[f64],
    ln_target: &[f64],
    npts: usize,
    extrapolate_below: bool,
    out: &mut [f64],
    threads: usize,
) -> Result<(), Refusal> {
    let nsrc = ln_source.len();
    if nsrc < 2 {
        return refuse("the remap needs at least two source levels");
    }
    if values.len() != nsrc * npts {
        return refuse(format!(
            "the source column holds {} values where {} levels of {} points need {}",
            values.len(),
            nsrc,
            npts,
            nsrc * npts
        ));
    }
    if npts == 0 {
        return if ln_target.is_empty() && out.is_empty() {
            Ok(())
        } else {
            refuse("a zero-point remap was handed target levels")
        };
    }
    if ln_target.len() % npts != 0 || out.len() != ln_target.len() {
        return refuse(format!(
            "the target holds {} values and the output {}, not one equal whole number of {}-point levels",
            ln_target.len(),
            out.len(),
            npts
        ));
    }
    // Every output element is one (level, point) pair, computed alone.
    let work = |first: usize, chunk: &mut [f64]| {
        for (offset, slot) in chunk.iter_mut().enumerate() {
            let flat = first + offset;
            let p = flat % npts;
            let x = ln_target[flat];
            let upper = searchsorted_left(ln_source, x).clamp(1, nsrc - 1);
            let lower = upper - 1;
            let mut weight = (x - ln_source[lower]) / (ln_source[upper] - ln_source[lower]);
            // numpy.maximum(weight, 0.0) and numpy.minimum(weight, 1.0),
            // both NaN-propagating.
            if weight < 0.0 {
                weight = 0.0;
            }
            if !extrapolate_below && weight > 1.0 {
                weight = 1.0;
            }
            let low = values[lower * npts + p];
            let high = values[upper * npts + p];
            *slot = low * (1.0 - weight) + high * weight;
        }
    };
    let n = out.len();
    run_rows(out, n, 1, threads, work);
    Ok(())
}

/// The operator's thread cap: the first of `values` (read from
/// `RAYON_NUM_THREADS`, then `OMP_NUM_THREADS`) that is a positive integer.
/// Shared machines cap every lane's threads with these variables, and a
/// library that read only the core count took every core on a 256-core box
/// whatever the cap said.  The thread split moves no bit of the answer, so
/// the cap changes the wall only.
pub fn thread_cap(values: &[Option<String>]) -> Option<usize> {
    values.iter().flatten().find_map(|v| v.trim().parse::<usize>().ok().filter(|n| *n > 0))
}

fn env_thread_cap() -> Option<usize> {
    thread_cap(&[std::env::var("RAYON_NUM_THREADS").ok(), std::env::var("OMP_NUM_THREADS").ok()])
}

/// The worker count a call uses when the caller passes zero: the cores
/// this process may use, under the operator's cap when one is set.
pub fn default_threads() -> usize {
    let cores = std::thread::available_parallelism().map(NonZeroUsize::get).unwrap_or(1);
    env_thread_cap().map_or(cores, |cap| cap.min(cores))
}

/// Split `out` (`rows` rows of `width`) into contiguous blocks of whole rows
/// and hand each to `work(first_row, block)` on its own scoped thread.
fn run_rows<F>(out: &mut [f64], rows: usize, width: usize, threads: usize, work: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    let threads = if threads == 0 { default_threads() } else { threads };
    // Below a few hundred thousand values a thread costs more than it saves.
    let wanted = (out.len() / 262_144).max(1);
    let workers = threads.min(wanted).min(rows).max(1);
    if workers == 1 {
        work(0, out);
        return;
    }
    let per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (block, chunk) in out.chunks_mut(per * width).enumerate() {
            let work = &work;
            scope.spawn(move || work(block * per, chunk));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_thread_cap_reads_the_first_positive_integer() {
        let s = |v: &str| Some(v.to_string());
        assert_eq!(thread_cap(&[s("4"), s("8")]), Some(4));
        assert_eq!(thread_cap(&[None, s("8")]), Some(8));
        assert_eq!(thread_cap(&[s("0"), s(" 3 ")]), Some(3));
        assert_eq!(thread_cap(&[s("lots"), None]), None);
        assert_eq!(thread_cap(&[None, None]), None);
    }

    #[test]
    fn numpy_mod_matches_python_semantics() {
        assert_eq!(numpy_mod(-10.0, 360.0), 350.0);
        assert_eq!(numpy_mod(370.0, 360.0), 10.0);
        assert_eq!(numpy_mod(-360.0, 360.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(numpy_mod(0.0, 360.0).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn searchsorted_matches_numpy_left() {
        let s = [1.0, 2.0, 3.0];
        assert_eq!(searchsorted_left(&s, 0.5), 0);
        assert_eq!(searchsorted_left(&s, 1.0), 0);
        assert_eq!(searchsorted_left(&s, 1.5), 1);
        assert_eq!(searchsorted_left(&s, 3.0), 2);
        assert_eq!(searchsorted_left(&s, 9.0), 3);
        assert_eq!(searchsorted_left(&s, f64::NAN), 3);
    }

    #[test]
    fn regrid_is_identity_on_matching_nodes() {
        let lat = [-90.0, 0.0, 90.0];
        let lon = [0.0, 90.0, 180.0, 270.0];
        let r = Regridder::new(&lat, &lon, &[0.0], &[90.0, 270.0]).unwrap();
        let field: Vec<f64> = (0..12).map(|v| v as f64).collect();
        let mut out = vec![0.0; 2];
        r.apply(&field, &mut out, 1).unwrap();
        assert_eq!(out, vec![5.0, 7.0]);
    }

    #[test]
    fn thread_count_moves_no_bit() {
        let nlat = 181;
        let nlon = 360;
        let lat: Vec<f64> = (0..nlat).map(|j| 90.0 - j as f64).collect();
        let lon: Vec<f64> = (0..nlon).map(|i| i as f64).collect();
        let tlat: Vec<f64> = (0..96).map(|j| -88.5 + j as f64 * 1.86).collect();
        let tlon: Vec<f64> = (0..192).map(|i| i as f64 * 1.875).collect();
        let r = Regridder::new(&lat, &lon, &tlat, &tlon).unwrap();
        let field: Vec<f64> = (0..nlat * nlon * 8).map(|v| (v as f64 * 0.37).sin()).collect();
        let mut one = vec![0.0; 8 * 96 * 192];
        let mut many = vec![0.0; 8 * 96 * 192];
        r.apply(&field, &mut one, 1).unwrap();
        r.apply(&field, &mut many, 7).unwrap();
        assert!(one.iter().zip(&many).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn remap_holds_above_and_extrapolates_below_on_request() {
        let ln_source = [1.0, 2.0];
        let values = [10.0, 20.0];
        let mut out = [0.0; 3];
        remap(&values, &ln_source, &[0.0, 1.5, 3.0], 1, false, &mut out, 1).unwrap();
        assert_eq!(out, [10.0, 15.0, 20.0]);
        remap(&values, &ln_source, &[0.0, 1.5, 3.0], 1, true, &mut out, 1).unwrap();
        assert_eq!(out, [10.0, 15.0, 30.0]);
    }
}
