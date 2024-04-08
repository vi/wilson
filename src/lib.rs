#![doc = include_str!("../README.md")]

#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "f64")]
/// Floating-point type used in this crate. Can be configured to f32 or to f64 depending on mutually exclusive Cargo features.
pub type FP = f64;
#[cfg(feature = "f32")]
/// Floating-point type used in this crate. Can be configured to f32 or to f64 depending on mutually exclusive Cargo features.
pub type FP = f32;

/// Result of the [`wilson`] calculation.
/// 
/// Next "trial" is expected to be "success" with probability from `low` to `high` with a confidence that depend on `z` parameter.
pub struct WilsonResult {
    /// Lower bound of a Wilson confidence interval
    pub low: FP,
    /// Higher bound of a Wilson confidence interval
    pub high: FP,
}

/// Calculate upper and lower bounds of the Wilson interval.
/// 
/// `successes` divided by `trials` should be between `low` and `high`.
/// 
/// The higher `trials` and the lower `z` is, the narrowed the resulting interval.
/// 
/// It may panic when invalid valies (such as negatives or when `successes` greater than `trials`),
/// but zero `trials` is handled explicitly and result in a `[0, 1]` interval.
/// 
/// You can use fractional `trials` and/or `successes`. `successes=0` or `successes=trials` should work properly.
/// 
/// `z=3` should appriximately correspond to 99.7% confidence, `z=2` to 95% and `z=1` to about two thirds.
/// 
/// ```
/// # fn ban_user(){}
/// // Imagine we have a forum with statistics about flagged 
/// // and total posts for each user. 
/// // We want to decide whether to ban user based those data.
/// //
/// // Banning is expected to happen if at least third of messages of a user
/// // get flagged, but it should err on the side of not banning in case of 
/// // there is only one message available.
/// 
/// // If there are only two messages and both are flagged, it would ban.
/// // If there is one non-flagged message, it would require three flagged messages
/// // to get banned.
/// 
/// let n_posts = 10.0;
/// let n_flagged = 2.0;
/// 
/// if wilson::wilson(n_flagged, n_posts, 1.5).low > 0.33 {
///     ban_user();
/// }
/// ```
#[must_use]
pub fn wilson(successes: FP, trials: FP, z: FP) -> WilsonResult {
    if trials <= 0.001 {
        return WilsonResult {
            low: 0.0,
            high: 1.0,
        };
    }
    let n = trials;
    let s = successes;
    let p = (s + 0.5 * z * z) / (n + z * z);
    let d = z / (n + z * z) * (s * (n - s) / n + z * z / 4.0).sqrt();
    let high = p + d;
    let low = p - d;
    WilsonResult { low, high }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn it_works() {
        {
            let out = wilson(1.0, 2.0, 2.0);
            assert_abs_diff_eq!(out.low, 0.09175170954, epsilon = 0.000001);
            assert_abs_diff_eq!(out.high, 0.9082482905, epsilon = 0.000001);
        }
        {
            let out = wilson(10.0, 20.0, 2.0);
            assert_abs_diff_eq!(out.low, 0.2958758548, epsilon = 0.000001);
            assert_abs_diff_eq!(out.high, 0.7041241452, epsilon = 0.000001);
        }
        {
            let out = wilson(2.0, 20.0, 2.0);
            assert_abs_diff_eq!(out.low, 0.02722332891, epsilon = 0.000001);
            assert_abs_diff_eq!(out.high, 0.3061100044, epsilon = 0.000001);
        }
        {
            let out = wilson(2.0, 20.0, 3.0);
            assert_abs_diff_eq!(out.low, 0.01595229175, epsilon = 0.000001);
            assert_abs_diff_eq!(out.high, 0.4323235703, epsilon = 0.000001);
        }
        {
            let out = wilson(20.0, 20.0, 2.0);
            assert_abs_diff_eq!(out.low, 0.8333333333, epsilon = 0.000001);
            assert_abs_diff_eq!(out.high, 1.0, epsilon = 0.000001);
        }  {
            let out = wilson(0.0, 20.0, 2.0);
            assert_abs_diff_eq!(out.low, 0.0, epsilon = 0.000001);
            assert_abs_diff_eq!(out.high, 0.1666666667, epsilon = 0.000001);
        }
    }

    #[test]
    fn degenerate() {
        let out = wilson(0.0, 0.0, 2.0);
        assert_abs_diff_eq!(out.low, 0.0, epsilon = 0.000001);
        assert_abs_diff_eq!(out.high, 1.0, epsilon = 0.000001);
    }

    #[test]
    fn degenerate2() {
        let out = wilson(0.005, 0.01, 2.0);
        assert_abs_diff_eq!(out.low, 0.0006238305611, epsilon = 0.000001);
        assert_abs_diff_eq!(out.high, 0.9993761694, epsilon = 0.000001);
    }
}
