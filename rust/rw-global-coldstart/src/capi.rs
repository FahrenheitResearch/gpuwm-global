//! The C ABI `src/arwen_global/coldstart_bridge.py` loads through ctypes.
//!
//! The engine's library-door discipline (obs-regrid, static-fields,
//! netcdf-writer): a cdylib behind ctypes, positional C signatures guarded
//! by an ABI version probe, a thread-local last-error string, a
//! source-revision stamp the release cut reads as bytes, and a contract
//! symbol (`arwen_global_coldstart_remap`) the door row names as its marker.
//!
//! Stateless: every call takes the caller's C-contiguous float64 buffers
//! (exactly `numpy.ascontiguousarray(..., dtype=float64)`) and writes into
//! a buffer the caller allocated, so no array crosses by copy.

use std::cell::RefCell;
use std::os::raw::c_char;

use crate::{remap, Regridder};

/// Bump when a signature changes shape, never for a rebuild.
pub const COLDSTART_ABI_VERSION: u32 = 1;

const OK: i32 = 0;
const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Turn a panic below the seam into the ABI's refusal: an unwind reaching
/// an `extern "C"` boundary aborts the host interpreter instead.
fn guard<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            LAST_ERROR.with(|slot| {
                if let Ok(mut message) = slot.try_borrow_mut() {
                    *message = format!("panic in the cold-start seam: {detail}");
                }
            });
            on_panic
        }
    }
}

/// # Safety
/// `ptr` must address `len` readable values, or be null with `len` 0.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr` must address `len` writable values, or be null with `len` 0.
unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]> {
    if len == 0 {
        return Some(&mut []);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

#[no_mangle]
pub extern "C" fn arwen_global_coldstart_abi_version() -> u32 {
    guard(0, || COLDSTART_ABI_VERSION)
}

/// The contract literal the door row (`arwen_global.doors`, marker
/// `arwen-global-coldstart-abi-v1`) finds in these bytes without executing
/// them.  Returned through an export so the linker cannot drop it; bump it
/// with [`COLDSTART_ABI_VERSION`].
#[no_mangle]
pub extern "C" fn arwen_global_coldstart_contract() -> *const c_char {
    guard(std::ptr::null(), || {
        static CONTRACT: &str = "arwen-global-coldstart-abi-v1\0";
        CONTRACT.as_ptr().cast()
    })
}

/// The source-revision stamp, read out of the binary as bytes by the
/// release cut and never executed.
#[no_mangle]
pub extern "C" fn arwen_global_coldstart_source_rev() -> *const c_char {
    guard(std::ptr::null(), || {
        static SOURCE_REV_STAMP: &str =
            concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
        SOURCE_REV_STAMP.as_ptr().cast()
    })
}

/// Copy the thread-local last error into `buf` (UTF-8, no NUL); returns the
/// full length so a short buffer is detectable.
///
/// # Safety
/// `buf` must address `cap` writable bytes, or be null with `cap` 0.
#[no_mangle]
pub unsafe extern "C" fn arwen_global_coldstart_last_error(buf: *mut u8, cap: usize) -> usize {
    guard(0, || {
        LAST_ERROR.with(|slot| {
            let message = slot.borrow();
            let raw = message.as_bytes();
            if !buf.is_null() && cap > 0 {
                let n = raw.len().min(cap);
                unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n) };
            }
            raw.len()
        })
    })
}

/// Bilinear regrid of `planes` stacked `nlat` x `nlon` planes from a regular
/// global grid onto the `ntlat` x `ntlon` target grid; `out` holds
/// `planes * ntlat * ntlon` values.  `threads` 0 means every core.
///
/// # Safety
/// Every pointer must address the number of values its lengths imply.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn arwen_global_coldstart_regrid(
    latitude: *const f64,
    nlat: usize,
    longitude: *const f64,
    nlon: usize,
    target_latitude: *const f64,
    ntlat: usize,
    target_longitude: *const f64,
    ntlon: usize,
    field: *const f64,
    planes: usize,
    out: *mut f64,
    threads: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (Some(latitude), Some(longitude), Some(target_latitude), Some(target_longitude)) = (
            unsafe { slice(latitude, nlat) },
            unsafe { slice(longitude, nlon) },
            unsafe { slice(target_latitude, ntlat) },
            unsafe { slice(target_longitude, ntlon) },
        ) else {
            return set_error("null coordinate pointer");
        };
        let regridder = match Regridder::new(latitude, longitude, target_latitude, target_longitude) {
            Ok(value) => value,
            Err(refusal) => return set_error(refusal.0),
        };
        let (Some(source_len), Some(target_len)) = (
            planes.checked_mul(nlat * nlon),
            planes.checked_mul(ntlat * ntlon),
        ) else {
            return set_error("the field's shape overflows");
        };
        let (Some(field), Some(out)) =
            (unsafe { slice(field, source_len) }, unsafe { slice_mut(out, target_len) })
        else {
            return set_error("null field or output pointer");
        };
        match regridder.apply(field, out, threads) {
            Ok(()) => OK,
            Err(refusal) => set_error(refusal.0),
        }
    })
}

/// Linear-in-ln(p) remap of `nsrc` x `npts` source values onto `ntgt` x
/// `npts` target ln(p) values; `out` holds `ntgt * npts` values.
/// `extrapolate_below` nonzero continues the bottom layer's gradient below
/// the bottom source level.  `threads` 0 means every core.
///
/// # Safety
/// Every pointer must address the number of values its lengths imply.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn arwen_global_coldstart_remap(
    values: *const f64,
    ln_source: *const f64,
    nsrc: usize,
    npts: usize,
    ln_target: *const f64,
    ntgt: usize,
    extrapolate_below: u8,
    out: *mut f64,
    threads: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (Some(source_len), Some(target_len)) = (nsrc.checked_mul(npts), ntgt.checked_mul(npts)) else {
            return set_error("the column's shape overflows");
        };
        let (Some(values), Some(ln_source), Some(ln_target), Some(out)) = (
            unsafe { slice(values, source_len) },
            unsafe { slice(ln_source, nsrc) },
            unsafe { slice(ln_target, target_len) },
            unsafe { slice_mut(out, target_len) },
        ) else {
            return set_error("null remap pointer");
        };
        match remap(values, ln_source, ln_target, npts, extrapolate_below != 0, out, threads) {
            Ok(()) => OK,
            Err(refusal) => set_error(refusal.0),
        }
    })
}
