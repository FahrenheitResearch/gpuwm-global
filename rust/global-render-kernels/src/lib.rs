//! The C ABI seam `src/arwen_global/render_kernels.py` loads through ctypes.
//!
//! Same discipline as every engine bridge (`netcdf-writer`, `obs-regrid`,
//! `static-fields`): a cdylib behind ctypes, positional C signatures guarded
//! by an ABI version probe, a thread-local last-error string, a panic guard
//! at every entry point, and a contract literal (`CONTRACT`) the Python half
//! and the door table both check in the bytes before anything is called.
//!
//! Arrays cross as C-contiguous little-endian f64, exactly the memory of a
//! `numpy.ascontiguousarray(..., dtype=numpy.float64)`.

pub mod kernels;
pub mod npsem;

use std::cell::RefCell;
use std::os::raw::c_char;

use kernels::{GaussianWeights, Post};

/// The seam's ABI version.  Bumped when a signature changes shape.
pub const ABI_VERSION: u32 = 1;

/// The contract literal.  `arwen_global.doors` pins it as this door's
/// marker, so a stale build is refused statically, before it is loaded.
pub static CONTRACT: &str = "woof-global-render-kernels-v1\0";

const OK: i32 = 0;
const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn fail(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

fn clear() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// A panic below the seam becomes the ABI's refusal instead of aborting the
/// host interpreter.
fn guard(body: impl FnOnce() -> i32) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            LAST_ERROR.with(|slot| {
                if let Ok(mut message) = slot.try_borrow_mut() {
                    *message = format!("panic in the render kernels: {detail}");
                }
            });
            ERR
        }
    }
}

/// # Safety
/// `ptr` addresses `len` readable values, or is null with `len` 0.
unsafe fn view<'a>(ptr: *const f64, len: usize) -> Option<&'a [f64]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr` addresses `len` writable values, or is null with `len` 0.
unsafe fn view_mut<'a>(ptr: *mut f64, len: usize) -> Option<&'a mut [f64]> {
    if len == 0 {
        return Some(&mut []);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

macro_rules! need {
    ($value:expr, $what:expr) => {
        match $value {
            Some(value) => value,
            None => return fail(concat!("null pointer for ", $what)),
        }
    };
}

#[no_mangle]
pub extern "C" fn grk_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn grk_contract() -> *const c_char {
    CONTRACT.as_ptr().cast()
}

/// The source-revision stamp the release cut reads out of the bytes.
#[no_mangle]
pub extern "C" fn grk_source_rev() -> *const c_char {
    static STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
    STAMP.as_ptr().cast()
}

/// Copy the last error into `buf` (UTF-8, no NUL); returns its full length.
///
/// # Safety
/// `buf` addresses `cap` writable bytes, or is null with `cap` 0.
#[no_mangle]
pub unsafe extern "C" fn grk_last_error(buf: *mut u8, cap: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let raw = message.as_bytes();
        if !buf.is_null() && cap > 0 {
            let n = raw.len().min(cap);
            unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n) };
        }
        raw.len()
    })
}

/// Gaussian grid to regular latitude-longitude, bilinear.
///
/// `field` is `nlev x nsy x nsx`, `out` is `nlev x nty x ntx`.  `post` is 0
/// (none), 1 (`clip(., 0, None)`) or 2 (`exp`).
///
/// # Safety
/// Every pointer addresses the number of values its shape implies.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn grk_gaussian_to_regular(
    src_lat: *const f64,
    nsy: usize,
    src_lon0: f64,
    nsx: usize,
    tgt_lat: *const f64,
    nty: usize,
    tgt_lon: *const f64,
    ntx: usize,
    field: *const f64,
    nlev: usize,
    post: u32,
    out: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        let Some(post) = Post::from_code(post) else {
            return fail(format!("unknown post-operation code {post}"));
        };
        let src_lat = need!(unsafe { view(src_lat, nsy) }, "src_lat");
        let tgt_lat = need!(unsafe { view(tgt_lat, nty) }, "tgt_lat");
        let tgt_lon = need!(unsafe { view(tgt_lon, ntx) }, "tgt_lon");
        let field = need!(unsafe { view(field, nlev * nsy * nsx) }, "field");
        let out = need!(unsafe { view_mut(out, nlev * nty * ntx) }, "out");
        match GaussianWeights::new(src_lat, nsx, src_lon0, tgt_lat, tgt_lon) {
            Ok(weights) => {
                weights.apply(field, nsy, nsx, nlev, post, out);
                OK
            }
            Err(message) => fail(message),
        }
    })
}

/// Regular periodic parent grid to arbitrary target points, bilinear.
///
/// `field` is `nlev x nlat x nlon`; `out` is `nlev x npts`.
///
/// # Safety
/// Every pointer addresses the number of values its shape implies.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn grk_periodic_bilinear(
    src_lat: *const f64,
    nlat: usize,
    lon0: f64,
    spacing: f64,
    nlon: usize,
    field: *const f64,
    nlev: usize,
    tgt_lat: *const f64,
    tgt_lon: *const f64,
    npts: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        let src_lat = need!(unsafe { view(src_lat, nlat) }, "src_lat");
        let field = need!(unsafe { view(field, nlev * nlat * nlon) }, "field");
        let tgt_lat = need!(unsafe { view(tgt_lat, npts) }, "tgt_lat");
        let tgt_lon = need!(unsafe { view(tgt_lon, npts) }, "tgt_lon");
        let out = need!(unsafe { view_mut(out, nlev * npts) }, "out");
        match kernels::periodic_bilinear(src_lat, lon0, spacing, nlon, field, nlev, tgt_lat, tgt_lon, out) {
            Ok(()) => OK,
            Err(message) => fail(message),
        }
    })
}

/// Column-wise ln p interpolation; `bottom` may be null.
///
/// `src_p`, `src_v` are `nsrc x ncol`; `tgt_p`, `bottom`, `out` are
/// `ntgt x ncol`.
///
/// # Safety
/// Every non-null pointer addresses the number of values its shape implies.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn grk_log_pressure_interpolate(
    src_p: *const f64,
    src_v: *const f64,
    nsrc: usize,
    tgt_p: *const f64,
    ntgt: usize,
    ncol: usize,
    bottom: *const f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        if nsrc < 2 {
            return fail("log-pressure interpolation needs at least two levels");
        }
        let src_p = need!(unsafe { view(src_p, nsrc * ncol) }, "src_p");
        let src_v = need!(unsafe { view(src_v, nsrc * ncol) }, "src_v");
        let tgt_p = need!(unsafe { view(tgt_p, ntgt * ncol) }, "tgt_p");
        let bottom = if bottom.is_null() { None } else { unsafe { view(bottom, ntgt * ncol) } };
        let out = need!(unsafe { view_mut(out, ntgt * ncol) }, "out");
        kernels::log_pressure_interpolate(src_p, src_v, nsrc, tgt_p, ntgt, ncol, bottom, out);
        OK
    })
}

/// Standard-lapse continuation of theta below the parent's bottom level.
///
/// # Safety
/// Every pointer addresses the number of values its shape implies.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn grk_standard_lapse_theta_below(
    src_p: *const f64,
    src_theta: *const f64,
    nsrc: usize,
    tgt_p: *const f64,
    ntgt: usize,
    ncol: usize,
    reference_pressure: f64,
    kappa: f64,
    gas_constant: f64,
    gravity: f64,
    lapse_rate: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        if nsrc < 1 {
            return fail("the parent column has no levels");
        }
        let src_p = need!(unsafe { view(src_p, nsrc * ncol) }, "src_p");
        let src_theta = need!(unsafe { view(src_theta, nsrc * ncol) }, "src_theta");
        let tgt_p = need!(unsafe { view(tgt_p, ntgt * ncol) }, "tgt_p");
        let out = need!(unsafe { view_mut(out, ntgt * ncol) }, "out");
        kernels::standard_lapse_theta_below(
            src_p, src_theta, nsrc, tgt_p, ntgt, ncol, reference_pressure, kappa, gas_constant,
            gravity, lapse_rate, out,
        );
        OK
    })
}

/// Hybrid half- and full-level pressure from `a`, `b` (`nhalf`) and `ps`.
///
/// # Safety
/// Every pointer addresses the number of values its shape implies.
#[no_mangle]
pub unsafe extern "C" fn grk_hybrid_pressure(
    a: *const f64,
    b: *const f64,
    nhalf: usize,
    ps: *const f64,
    npts: usize,
    p_half: *mut f64,
    p_full: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        if nhalf < 2 {
            return fail("a hybrid coordinate needs at least two half levels");
        }
        let a = need!(unsafe { view(a, nhalf) }, "a");
        let b = need!(unsafe { view(b, nhalf) }, "b");
        let ps = need!(unsafe { view(ps, npts) }, "ps");
        let p_half = need!(unsafe { view_mut(p_half, nhalf * npts) }, "p_half");
        let p_full = need!(unsafe { view_mut(p_full, (nhalf - 1) * npts) }, "p_full");
        kernels::hybrid_pressure(a, b, ps, p_half, p_full);
        OK
    })
}

/// Temperature and virtual temperature; every array is `n` long.
///
/// # Safety
/// Every pointer addresses `n` values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn grk_virtual_temperature(
    theta: *const f64,
    qv: *const f64,
    qc: *const f64,
    qr: *const f64,
    qi: *const f64,
    qs: *const f64,
    qg: *const f64,
    p_full: *const f64,
    n: usize,
    reference_pressure: f64,
    kappa: f64,
    temperature: *mut f64,
    virtual_t: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        let theta = need!(unsafe { view(theta, n) }, "theta");
        let species = [
            need!(unsafe { view(qv, n) }, "qv"),
            need!(unsafe { view(qc, n) }, "qc"),
            need!(unsafe { view(qr, n) }, "qr"),
            need!(unsafe { view(qi, n) }, "qi"),
            need!(unsafe { view(qs, n) }, "qs"),
            need!(unsafe { view(qg, n) }, "qg"),
        ];
        let p_full = need!(unsafe { view(p_full, n) }, "p_full");
        let temperature = need!(unsafe { view_mut(temperature, n) }, "temperature");
        let virtual_t = need!(unsafe { view_mut(virtual_t, n) }, "virtual_t");
        kernels::virtual_temperature(theta, species, p_full, reference_pressure, kappa, temperature, virtual_t);
        OK
    })
}

/// Hydrostatic half-level (and, when `phi_full` is not null, full-level)
/// geopotential from the surface up.  `virtual_t` is `nz x npts`, `p_half`
/// and `phi_half` are `(nz+1) x npts`.
///
/// # Safety
/// Every non-null pointer addresses the number of values its shape implies.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn grk_hydrostatic(
    virtual_t: *const f64,
    p_half: *const f64,
    nz: usize,
    phi_surface: *const f64,
    npts: usize,
    gas_constant: f64,
    phi_half: *mut f64,
    phi_full: *mut f64,
) -> i32 {
    guard(|| {
        clear();
        let virtual_t = need!(unsafe { view(virtual_t, nz * npts) }, "virtual_t");
        let p_half = need!(unsafe { view(p_half, (nz + 1) * npts) }, "p_half");
        let phi_surface = need!(unsafe { view(phi_surface, npts) }, "phi_surface");
        let phi_half = need!(unsafe { view_mut(phi_half, (nz + 1) * npts) }, "phi_half");
        let phi_full = if phi_full.is_null() { None } else { unsafe { view_mut(phi_full, nz * npts) } };
        kernels::hydrostatic(virtual_t, p_half, phi_surface, gas_constant, phi_half, phi_full);
        OK
    })
}
