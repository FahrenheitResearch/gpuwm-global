//! The three NumPy scalar semantics the kernels must reproduce exactly.
//!
//! The NumPy oracle these kernels replace leans on `np.interp`,
//! `np.mod`/`%` on floats and `np.clip(x, 0, None)`.  Each has an edge
//! behaviour a naive Rust spelling gets wrong (an exact node hit, a tiny
//! negative longitude offset that wraps to exactly one period, a negative
//! zero), and a byte-identical port has to get every one of them right.

/// `np.interp`'s bracket for one abscissa on a strictly increasing `xp`:
/// `-1` below `xp[0]`, `len` above `xp[len-1]`, otherwise the largest `j`
/// with `xp[j] <= x` (so `len-1` at an exact hit on the last node).
#[inline]
pub fn interp_bracket(x: f64, xp: &[f64]) -> isize {
    let n = xp.len();
    if x > xp[n - 1] {
        return n as isize;
    }
    if x < xp[0] {
        return -1;
    }
    xp.partition_point(|&value| value <= x) as isize - 1
}

/// `np.interp(x, xp, fp, left, right)` for one abscissa, operation for
/// operation as NumPy's compiled loop evaluates it (including its NaN
/// recovery branch).
#[inline]
pub fn interp(x: f64, xp: &[f64], fp: &[f64], left: f64, right: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let n = xp.len();
    if n == 1 {
        return if x < xp[0] {
            left
        } else if x > xp[0] {
            right
        } else {
            fp[0]
        };
    }
    let j = interp_bracket(x, xp);
    if j == -1 {
        return left;
    }
    if j == n as isize {
        return right;
    }
    let j = j as usize;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut value = slope * (x - xp[j]) + fp[j];
    if value.is_nan() {
        value = slope * (x - xp[j + 1]) + fp[j + 1];
        if value.is_nan() && fp[j] == fp[j + 1] {
            value = fp[j];
        }
    }
    value
}

/// `np.remainder(a, b)` (also `a % b` on float arrays): the C `fmod`, moved
/// into the divisor's sign, with an exact zero carrying the divisor's sign.
/// `np.mod(-1e-20, 360.0)` is exactly `360.0`, and the regrids depend on it.
#[inline]
pub fn remainder(a: f64, b: f64) -> f64 {
    let mut m = a % b;
    if b == 0.0 {
        return m;
    }
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0.0_f64.copysign(b);
    }
    m
}

/// `np.clip(x, 0.0, None)` and `np.maximum(x, 0.0)` as the oracle's NumPy
/// evaluates them: NaN passes through, anything below zero and either zero
/// becomes `+0.0`.
#[inline]
pub fn floor_zero(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// `np.clip(x, 0.0, 1.0)` for the weight clamp of the Gaussian regrid.
#[inline]
pub fn clip_unit(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x > 1.0 {
        1.0
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remainder_wraps_a_tiny_negative_to_one_period() {
        assert_eq!(remainder(-1.0e-20, 360.0), 360.0);
        assert_eq!(remainder(-0.0, 360.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(remainder(370.0, 360.0), 10.0);
        assert!(remainder(f64::NAN, 360.0).is_nan());
    }

    #[test]
    fn interp_hits_nodes_and_holds_outside() {
        let xp = [0.0, 1.0, 3.0];
        let fp = [10.0, 20.0, 40.0];
        assert_eq!(interp(-1.0, &xp, &fp, -5.0, 99.0), -5.0);
        assert_eq!(interp(4.0, &xp, &fp, -5.0, 99.0), 99.0);
        assert_eq!(interp(3.0, &xp, &fp, -5.0, 99.0), 40.0);
        assert_eq!(interp(1.0, &xp, &fp, -5.0, 99.0), 20.0);
        assert_eq!(interp(2.0, &xp, &fp, -5.0, 99.0), 30.0);
    }

    #[test]
    fn floor_zero_canonicalises_negative_zero() {
        assert_eq!(floor_zero(-0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(floor_zero(-3.0), 0.0);
        assert!(floor_zero(f64::NAN).is_nan());
    }
}
