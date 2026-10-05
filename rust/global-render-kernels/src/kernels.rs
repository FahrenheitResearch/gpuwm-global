//! The kernels, in the IEEE operation order of the NumPy oracle.
//!
//! Every array is C-ordered with the level axis first and the horizontal
//! points flattened behind it, exactly the memory of the `(nlev, ny, nx)`
//! NumPy arrays the Python half hands over.  Every output point depends only
//! on its own inputs, so the thread split never changes a single bit.

use crate::npsem::{clip_unit, floor_zero, interp, remainder};

/// A raw output pointer shared by threads that write disjoint elements.
#[derive(Clone, Copy)]
pub struct Out(pub *mut f64);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// # Safety
    /// `index` is in bounds and no other thread writes it.
    #[inline]
    pub unsafe fn put(self, index: usize, value: f64) {
        unsafe { *self.0.add(index) = value };
    }
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

fn worker_count() -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    env_thread_cap().map_or(cores, |cap| cap.min(cores))
}

/// Run `body(lo, hi)` over `0..n` split into contiguous ranges, one per
/// worker, never fewer than `min_chunk` items to a range.
pub fn for_ranges(n: usize, min_chunk: usize, body: impl Fn(usize, usize) + Sync) {
    if n == 0 {
        return;
    }
    let wanted = n.div_ceil(min_chunk.max(1));
    let workers = worker_count().min(wanted).max(1);
    if workers == 1 {
        body(0, n);
        return;
    }
    let chunk = n.div_ceil(workers);
    std::thread::scope(|scope| {
        for index in 0..workers {
            let lo = index * chunk;
            let hi = ((index + 1) * chunk).min(n);
            if lo >= hi {
                continue;
            }
            let body = &body;
            scope.spawn(move || body(lo, hi));
        }
    });
}

/// What is applied to each regridded value before it is stored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Post {
    /// The bilinear value as it is.
    None,
    /// `np.clip(value, 0.0, None)`: the tape's water species.
    FloorZero,
    /// `np.exp(value)`: surface pressure from its logarithm.
    Exp,
}

impl Post {
    pub fn from_code(code: u32) -> Option<Post> {
        match code {
            0 => Some(Post::None),
            1 => Some(Post::FloorZero),
            2 => Some(Post::Exp),
            _ => None,
        }
    }

    #[inline]
    fn apply(self, value: f64) -> f64 {
        match self {
            Post::None => value,
            Post::FloorZero => floor_zero(value),
            Post::Exp => value.exp(),
        }
    }
}

/// Bilinear weights from a Gaussian grid (latitudes strictly increasing,
/// longitudes uniform from `lon0` over one period) to the target rows and
/// columns of a regular grid.  The oracle is `_gaussian_to_regular`.
pub struct GaussianWeights {
    y0: Vec<usize>,
    wy: Vec<f64>,
    x0: Vec<usize>,
    x1: Vec<usize>,
    wx: Vec<f64>,
}

impl GaussianWeights {
    pub fn new(
        src_lat: &[f64],
        nsx: usize,
        lon0: f64,
        tgt_lat: &[f64],
        tgt_lon: &[f64],
    ) -> Result<Self, String> {
        let nsy = src_lat.len();
        if nsy < 2 || nsx < 1 {
            return Err(format!(
                "a Gaussian source grid needs at least 2 latitudes and 1 longitude, got {nsy}x{nsx}"
            ));
        }
        if src_lat.windows(2).any(|pair| !(pair[1] > pair[0])) {
            return Err("Gaussian latitudes must increase strictly".into());
        }
        if !lon0.is_finite() {
            return Err("the first Gaussian longitude is not finite".into());
        }
        if tgt_lat.iter().chain(tgt_lon).any(|value| !value.is_finite()) {
            return Err("a target latitude or longitude is not finite".into());
        }
        let index: Vec<f64> = (0..nsy).map(|j| j as f64).collect();
        let (mut y0, mut wy) = (Vec::with_capacity(tgt_lat.len()), Vec::with_capacity(tgt_lat.len()));
        for &lat in tgt_lat {
            let fy = interp(lat, src_lat, &index, index[0], index[nsy - 1]);
            let row = (fy.trunc() as i64).min(nsy as i64 - 2);
            y0.push(row as usize);
            wy.push(clip_unit(fy - row as f64));
        }
        let dlon = 360.0 / nsx as f64;
        let (mut x0, mut x1, mut wx) = (
            Vec::with_capacity(tgt_lon.len()),
            Vec::with_capacity(tgt_lon.len()),
            Vec::with_capacity(tgt_lon.len()),
        );
        for &lon in tgt_lon {
            let fx = remainder(lon - lon0, 360.0) / dlon;
            let column = (fx.trunc() as i64).rem_euclid(nsx as i64) as usize;
            x0.push(column);
            x1.push((column + 1) % nsx);
            wx.push(fx - fx.floor());
        }
        Ok(Self { y0, wy, x0, x1, wx })
    }

    /// Regrid `nlev` levels of `field` (`nlev x nsy x nsx`) into `out`
    /// (`nlev x nty x ntx`).
    pub fn apply(&self, field: &[f64], nsy: usize, nsx: usize, nlev: usize, post: Post, out: &mut [f64]) {
        let (nty, ntx) = (self.y0.len(), self.x0.len());
        if nty == 0 || ntx == 0 || nlev == 0 {
            return;
        }
        let plane = nsy * nsx;
        // One disjoint mutable row per (level, target latitude), handed out
        // in contiguous groups, so no thread shares a byte of output.
        let mut rows: Vec<(usize, &mut [f64])> = out[..nlev * nty * ntx].chunks_mut(ntx).enumerate().collect();
        let chunk = rows.len().div_ceil(worker_count()).max(1);
        std::thread::scope(|scope| {
            for group in rows.chunks_mut(chunk) {
                scope.spawn(move || {
                    for (flat, row_out) in group.iter_mut() {
                        let lev = *flat / nty;
                        let i = *flat % nty;
                        let base = lev * plane;
                        let wy = self.wy[i];
                        let lower = base + self.y0[i] * nsx;
                        let upper = lower + nsx;
                        let one_minus_wy = 1.0 - wy;
                        for j in 0..ntx {
                            let (a, b) = (self.x0[j], self.x1[j]);
                            let wx = self.wx[j];
                            let one_minus_wx = 1.0 - wx;
                            let south = one_minus_wx * field[lower + a] + wx * field[lower + b];
                            let north = one_minus_wx * field[upper + a] + wx * field[upper + b];
                            row_out[j] = post.apply(one_minus_wy * south + wy * north);
                        }
                    }
                });
            }
        });
    }
}

/// Bilinear interpolation on a regular, periodic-longitude parent grid at
/// arbitrary target points.  The oracle is `periodic_bilinear`; the
/// Python half has already refused every malformed grid and every target
/// outside the parent's latitude span.
#[allow(clippy::too_many_arguments)]
pub fn periodic_bilinear(
    src_lat: &[f64],
    lon0: f64,
    spacing: f64,
    nlon: usize,
    field: &[f64],
    nlev: usize,
    tgt_lat: &[f64],
    tgt_lon: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    let nlat = src_lat.len();
    if nlat < 2 || nlon < 1 {
        return Err(format!("parent grid {nlat}x{nlon} is too small"));
    }
    let descending = src_lat[0] > src_lat[nlat - 1];
    let lat: Vec<f64> = if descending {
        src_lat.iter().rev().copied().collect()
    } else {
        src_lat.to_vec()
    };
    let row = |j: usize| if descending { nlat - 1 - j } else { j };
    let npts = tgt_lat.len();
    let plane = nlat * nlon;
    let sink = Out(out.as_mut_ptr());
    for_ranges(npts, 4096, |lo, hi| {
        for p in lo..hi {
            let y = tgt_lat[p];
            let j1 = lat.partition_point(|&value| value <= y).clamp(1, nlat - 1);
            let j0 = j1 - 1;
            let wy = (y - lat[j0]) / (lat[j1] - lat[j0]);
            let x = remainder(tgt_lon[p] - lon0, 360.0) / spacing;
            let fl = x.floor();
            let i0 = (fl as i64).rem_euclid(nlon as i64) as usize;
            let i1 = (i0 + 1) % nlon;
            let wx = x - fl;
            let (r0, r1) = (row(j0) * nlon, row(j1) * nlon);
            for lev in 0..nlev {
                let base = lev * plane;
                let f00 = field[base + r0 + i0];
                let f01 = field[base + r0 + i1];
                let f10 = field[base + r1 + i0];
                let f11 = field[base + r1 + i1];
                let lower = f00 * (1.0 - wx) + f01 * wx;
                let upper = f10 * (1.0 - wx) + f11 * wx;
                // SAFETY: (lev, p) is written once, by the range owning p.
                unsafe { sink.put(lev * npts + p, lower * (1.0 - wy) + upper * wy) };
            }
        }
    });
    Ok(())
}

/// Independent-column linear interpolation in ln p, held outside the
/// parent's span, with an optional continuation used wherever a target lies
/// below the parent's bottom level.  The oracle is
/// `log_pressure_interpolate`, whose refusals stay in Python.
pub fn log_pressure_interpolate(
    src_p: &[f64],
    src_v: &[f64],
    nsrc: usize,
    tgt_p: &[f64],
    ntgt: usize,
    ncol: usize,
    bottom: Option<&[f64]>,
    out: &mut [f64],
) {
    let sink = Out(out.as_mut_ptr());
    for_ranges(ncol, 256, |lo, hi| {
        let mut xp = vec![0.0; nsrc];
        let mut fp = vec![0.0; nsrc];
        for c in lo..hi {
            for k in 0..nsrc {
                xp[k] = src_p[k * ncol + c].ln();
                fp[k] = src_v[k * ncol + c];
            }
            let bottom_p = src_p[(nsrc - 1) * ncol + c];
            for t in 0..ntgt {
                let at = t * ncol + c;
                let target = tgt_p[at];
                let mut value = interp(target.ln(), &xp, &fp, fp[0], fp[nsrc - 1]);
                if let Some(continued) = bottom {
                    if target > bottom_p {
                        value = continued[at];
                    }
                }
                // SAFETY: column c is owned by this range.
                unsafe { sink.put(at, value) };
            }
        }
    });
}

/// Potential temperature continued below the parent's bottom full level at
/// the standard lapse rate.  The oracle is `standard_lapse_theta_below`.
#[allow(clippy::too_many_arguments)]
pub fn standard_lapse_theta_below(
    src_p: &[f64],
    src_theta: &[f64],
    nsrc: usize,
    tgt_p: &[f64],
    ntgt: usize,
    ncol: usize,
    reference_pressure: f64,
    kappa: f64,
    gas_constant: f64,
    gravity: f64,
    lapse_rate: f64,
    out: &mut [f64],
) {
    let sink = Out(out.as_mut_ptr());
    for_ranges(ncol, 1024, |lo, hi| {
        for c in lo..hi {
            let bottom_p = src_p[(nsrc - 1) * ncol + c];
            let bottom_theta = src_theta[(nsrc - 1) * ncol + c];
            let bottom_t = bottom_theta * (bottom_p / reference_pressure).powf(kappa);
            let scale = gas_constant * bottom_t / gravity;
            for t in 0..ntgt {
                let at = t * ncol + c;
                let target = tgt_p[at];
                let depth = scale * (target / bottom_p).ln();
                let continued = bottom_t + lapse_rate * depth;
                // SAFETY: column c is owned by this range.
                unsafe { sink.put(at, continued * (reference_pressure / target).powf(kappa)) };
            }
        }
    });
}

/// Hybrid half- and full-level pressure: `p_half = a + b * ps`,
/// `p_full = sqrt(p_half[k] * p_half[k+1])`.
pub fn hybrid_pressure(a: &[f64], b: &[f64], ps: &[f64], p_half: &mut [f64], p_full: &mut [f64]) {
    let nhalf = a.len();
    let npts = ps.len();
    let (half, full) = (Out(p_half.as_mut_ptr()), Out(p_full.as_mut_ptr()));
    for_ranges(npts, 4096, |lo, hi| {
        for k in 0..nhalf {
            for p in lo..hi {
                let value = a[k] + b[k] * ps[p];
                // SAFETY: points lo..hi are owned by this range.
                unsafe { half.put(k * npts + p, value) };
            }
        }
        for k in 0..nhalf - 1 {
            for p in lo..hi {
                // SAFETY: as above; the reads are of values this range wrote.
                let (top, bottom) = unsafe { (*half.0.add(k * npts + p), *half.0.add((k + 1) * npts + p)) };
                unsafe { full.put(k * npts + p, (top * bottom).sqrt()) };
            }
        }
    });
}

/// Temperature and virtual temperature from potential temperature and the
/// six water species: `T = theta (p/p0)^kappa`,
/// `Tv = T (1 + 0.61 qv - (qc + qr + qi + qs + qg))`.
#[allow(clippy::too_many_arguments)]
pub fn virtual_temperature(
    theta: &[f64],
    species: [&[f64]; 6],
    p_full: &[f64],
    reference_pressure: f64,
    kappa: f64,
    temperature: &mut [f64],
    virtual_t: &mut [f64],
) {
    let n = theta.len();
    let (t_out, tv_out) = (Out(temperature.as_mut_ptr()), Out(virtual_t.as_mut_ptr()));
    let [qv, qc, qr, qi, qs, qg] = species;
    for_ranges(n, 4096, |lo, hi| {
        for i in lo..hi {
            let t = theta[i] * (p_full[i] / reference_pressure).powf(kappa);
            // Python's sum() starts from the integer 0, so the first add is
            // 0 + qc and a negative zero does not survive it.
            let condensate = ((((0.0 + qc[i]) + qr[i]) + qi[i]) + qs[i]) + qg[i];
            let factor = (1.0 + 0.61 * qv[i]) - condensate;
            // SAFETY: element i is owned by this range.
            unsafe {
                t_out.put(i, t);
                tv_out.put(i, t * factor);
            }
        }
    });
}

/// Hydrostatic geopotential, integrated upward from the surface:
/// `phi[k] = phi[k+1] + R Tv[k] ln(p[k+1]/p[k])` on half levels and, when
/// asked, `phi_full[k] = phi[k+1] + R Tv[k] ln(p[k+1]/sqrt(p[k] p[k+1]))`.
pub fn hydrostatic(
    virtual_t: &[f64],
    p_half: &[f64],
    phi_surface: &[f64],
    gas_constant: f64,
    phi_half: &mut [f64],
    phi_full: Option<&mut [f64]>,
) {
    let npts = phi_surface.len();
    let nz = virtual_t.len() / npts.max(1);
    let half = Out(phi_half.as_mut_ptr());
    let full = phi_full.map(|slice| Out(slice.as_mut_ptr()));
    for_ranges(npts, 4096, |lo, hi| {
        for p in lo..hi {
            // SAFETY: points lo..hi are owned by this range.
            unsafe { half.put(nz * npts + p, phi_surface[p]) };
        }
        for k in (0..nz).rev() {
            for p in lo..hi {
                let below = unsafe { *half.0.add((k + 1) * npts + p) };
                let rtv = gas_constant * virtual_t[k * npts + p];
                let (top, bottom) = (p_half[k * npts + p], p_half[(k + 1) * npts + p]);
                unsafe { half.put(k * npts + p, below + rtv * (bottom / top).ln()) };
                if let Some(full) = full {
                    let mid = (top * bottom).sqrt();
                    unsafe { full.put(k * npts + p, below + rtv * (bottom / mid).ln()) };
                }
            }
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
}
