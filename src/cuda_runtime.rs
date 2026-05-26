//! Shared CUDA runtime probing helpers.
//!
//! Two call sites use this module with intentionally different policies:
//! - The runtime validation path in `transcribe::parakeet` knows which CUDA
//!   major the bundled ONNX Runtime was built against, so it must prefer the
//!   matching soname first and hard-fail explicit readable mismatches.
//! - The setup path is inventory collection, not validation of one specific
//!   binary. It probes every known soname first, then applies an explicit host
//!   policy to the detected majors.
//!
//! Keeping probe collection separate from decision logic avoids the regression
//! where mixed CUDA 12/13 installs depended on whichever soname happened to be
//! tried first. The helpers also surface "library present but version unreadable"
//! as `Indeterminate` so callers can choose the right trade-off for their path.

use std::collections::BTreeSet;
use std::ffi::CString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaRuntimeCandidate {
    pub soname: &'static str,
    pub major_hint: Option<i32>,
}

const CUDART_12: CudaRuntimeCandidate = CudaRuntimeCandidate {
    soname: "libcudart.so.12",
    major_hint: Some(12),
};
const CUDART_13: CudaRuntimeCandidate = CudaRuntimeCandidate {
    soname: "libcudart.so.13",
    major_hint: Some(13),
};
const CUDART_UNVERSIONED: CudaRuntimeCandidate = CudaRuntimeCandidate {
    soname: "libcudart.so",
    major_hint: None,
};

/// Probe order for validating a bundled CUDA build.
///
/// This path knows the exact CUDA major the binary expects, so mixed-install
/// hosts must try the matching soname first. Simply hardcoding "13 before 12"
/// or the reverse makes one side of the mixed-install matrix fail incorrectly.
#[cfg(any(test, feature = "parakeet-cuda", feature = "parakeet-tensorrt"))]
pub(crate) fn runtime_probe_candidates(expected_major: i32) -> [CudaRuntimeCandidate; 3] {
    match expected_major {
        13 => [CUDART_13, CUDART_12, CUDART_UNVERSIONED],
        _ => [CUDART_12, CUDART_13, CUDART_UNVERSIONED],
    }
}

/// Probe order for setup-time host inventory.
///
/// Setup deliberately keeps probing neutral: collect every versioned runtime we
/// can see, then apply an explicit mixed-install policy later. That prevents
/// setup recommendations from inheriting an accidental preference from dlopen
/// order.
pub(crate) fn setup_probe_candidates() -> [CudaRuntimeCandidate; 3] {
    // Setup is inventory collection, not preference selection. Probe both
    // supported versioned sonames plus the unversioned fallback, then apply
    // an explicit policy to the detected majors.
    [CUDART_12, CUDART_13, CUDART_UNVERSIONED]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaRuntimeProbeResult {
    Missing,
    Detected { version: i32 },
    /// The library opened, but we could not read a runtime version from it.
    /// Callers decide whether that should fail closed or fall back to a more
    /// permissive "let the downstream runtime validate it" behavior.
    Indeterminate,
}

impl CudaRuntimeProbeResult {
    pub(crate) fn detected_major(self) -> Option<i32> {
        match self {
            CudaRuntimeProbeResult::Detected { version } => Some(version / 1000),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProbedCudaRuntime {
    pub candidate: CudaRuntimeCandidate,
    pub result: CudaRuntimeProbeResult,
}

pub(crate) fn probe_cuda_runtime_candidates<I, F>(
    candidates: I,
    mut probe: F,
) -> Vec<ProbedCudaRuntime>
where
    I: IntoIterator<Item = CudaRuntimeCandidate>,
    F: FnMut(&'static str) -> CudaRuntimeProbeResult,
{
    candidates
        .into_iter()
        .map(|candidate| ProbedCudaRuntime {
            candidate,
            result: probe(candidate.soname),
        })
        .collect()
}

pub(crate) fn probe_system_cuda_runtimes<I>(candidates: I) -> Vec<ProbedCudaRuntime>
where
    I: IntoIterator<Item = CudaRuntimeCandidate>,
{
    probe_cuda_runtime_candidates(candidates, probe_system_cuda_runtime)
}

fn probe_system_cuda_runtime(soname: &'static str) -> CudaRuntimeProbeResult {
    let soname = match CString::new(soname) {
        Ok(soname) => soname,
        Err(_) => return CudaRuntimeProbeResult::Missing,
    };

    let handle = unsafe { libc::dlopen(soname.as_ptr(), libc::RTLD_LAZY) };
    if handle.is_null() {
        return CudaRuntimeProbeResult::Missing;
    }

    let sym_name = CString::new("cudaRuntimeGetVersion").expect("literal has no interior NUL");
    let sym = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
    if sym.is_null() {
        unsafe { libc::dlclose(handle) };
        return CudaRuntimeProbeResult::Indeterminate;
    }

    type CudaRuntimeGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
    let get_version: CudaRuntimeGetVersion = unsafe { std::mem::transmute(sym) };

    let mut version: i32 = 0;
    let result = unsafe { get_version(&mut version) };
    unsafe { libc::dlclose(handle) };

    if result != 0 {
        return CudaRuntimeProbeResult::Indeterminate;
    }

    CudaRuntimeProbeResult::Detected { version }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct DetectedCudaRuntimes {
    majors: BTreeSet<i32>,
}

impl DetectedCudaRuntimes {
    pub(crate) fn from_probes(probes: &[ProbedCudaRuntime]) -> Self {
        let majors = probes
            .iter()
            .filter_map(|probe| probe.result.detected_major())
            .collect();
        Self { majors }
    }

    #[cfg(test)]
    pub(crate) fn majors(&self) -> Vec<i32> {
        self.majors.iter().copied().collect()
    }

    /// Explicit host policy for setup-time mixed installs: prefer the highest
    /// detected CUDA major. This keeps setup deterministic and independent of
    /// soname probe order while matching the newest installed runtime by default.
    pub(crate) fn preferred_major(&self) -> Option<i32> {
        self.majors.iter().next_back().copied()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.majors.is_empty()
    }
}

#[cfg(any(test, feature = "parakeet-cuda", feature = "parakeet-tensorrt"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeProbeDecision {
    Compatible {
        candidate: CudaRuntimeCandidate,
        version: i32,
    },
    Mismatch {
        candidate: CudaRuntimeCandidate,
        version: i32,
    },
    Missing,
    /// No compatible readable runtime was found, but at least one candidate did
    /// open. Callers that prefer avoiding false negatives can proceed and let a
    /// downstream runtime validate compatibility at session creation time.
    Indeterminate {
        candidate: CudaRuntimeCandidate,
    },
}

/// Reduce raw probe results to the decision a caller cares about.
///
/// Probe order matters here. We return the first compatible runtime in the
/// caller-supplied order so bundled CUDA 12 and CUDA 13 builds each prefer the
/// matching soname on mixed-install hosts.
///
/// When `enforce_expected_major` is `true`, precedence is:
///   Compatible > Mismatch > Indeterminate > Missing
///
/// A readable wrong-major result is already proof the host CUDA is wrong;
/// returning `Indeterminate` in that case would allow the bundled path to
/// proceed when it should fail closed. `Indeterminate` is only returned when
/// no compatible runtime and no readable mismatch were found but at least one
/// candidate opened without yielding a readable version.
///
/// When `enforce_expected_major` is `false` (load-dynamic / permissive path),
/// any readable runtime is accepted as `Compatible` immediately in the loop, so
/// the post-loop precedence is never reached for readable results.
#[cfg(any(test, feature = "parakeet-cuda", feature = "parakeet-tensorrt"))]
pub(crate) fn evaluate_runtime_probe(
    expected_major: i32,
    probes: &[ProbedCudaRuntime],
    enforce_expected_major: bool,
) -> RuntimeProbeDecision {
    let mut first_detected_mismatch = None;
    let mut first_indeterminate = None;

    for probe in probes {
        match probe.result {
            CudaRuntimeProbeResult::Detected { version } => {
                let major = version / 1000;
                if !enforce_expected_major || major == expected_major {
                    return RuntimeProbeDecision::Compatible {
                        candidate: probe.candidate,
                        version,
                    };
                }
                if first_detected_mismatch.is_none() {
                    first_detected_mismatch = Some((probe.candidate, version));
                }
            }
            CudaRuntimeProbeResult::Indeterminate => {
                if first_indeterminate.is_none() {
                    first_indeterminate = Some(probe.candidate);
                }
            }
            CudaRuntimeProbeResult::Missing => {}
        }
    }

    if let Some((candidate, version)) = first_detected_mismatch {
        return RuntimeProbeDecision::Mismatch { candidate, version };
    }

    if let Some(candidate) = first_indeterminate {
        return RuntimeProbeDecision::Indeterminate { candidate };
    }

    RuntimeProbeDecision::Missing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detected(major: i32) -> CudaRuntimeProbeResult {
        CudaRuntimeProbeResult::Detected {
            version: major * 1000,
        }
    }

    fn probes_for(
        candidates: [CudaRuntimeCandidate; 3],
        results: &[(&'static str, CudaRuntimeProbeResult)],
    ) -> Vec<ProbedCudaRuntime> {
        probe_cuda_runtime_candidates(candidates, |soname| {
            results
                .iter()
                .find_map(|(name, result)| (*name == soname).then_some(*result))
                .unwrap_or(CudaRuntimeProbeResult::Missing)
        })
    }

    #[test]
    fn runtime_probe_accepts_only_cuda_12_when_expected() {
        let probes = probes_for(
            runtime_probe_candidates(12),
            &[("libcudart.so.12", detected(12))],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: runtime_probe_candidates(12)[0],
                version: 12000,
            }
        );
    }

    #[test]
    fn runtime_probe_accepts_only_cuda_13_when_expected() {
        let probes = probes_for(
            runtime_probe_candidates(13),
            &[("libcudart.so.13", detected(13))],
        );

        assert_eq!(
            evaluate_runtime_probe(13, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: runtime_probe_candidates(13)[0],
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_prefers_matching_cuda_12_on_mixed_install() {
        let probes = probes_for(
            runtime_probe_candidates(12),
            &[
                ("libcudart.so.12", detected(12)),
                ("libcudart.so.13", detected(13)),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: runtime_probe_candidates(12)[0],
                version: 12000,
            }
        );
    }

    #[test]
    fn runtime_probe_prefers_matching_cuda_13_on_mixed_install() {
        let probes = probes_for(
            runtime_probe_candidates(13),
            &[
                ("libcudart.so.12", detected(12)),
                ("libcudart.so.13", detected(13)),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(13, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: runtime_probe_candidates(13)[0],
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_falls_back_to_unversioned_when_it_matches() {
        let probes = probes_for(
            runtime_probe_candidates(12),
            &[
                ("libcudart.so.13", detected(13)),
                ("libcudart.so", detected(12)),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: runtime_probe_candidates(12)[2],
                version: 12000,
            }
        );
    }

    #[test]
    fn runtime_probe_rejects_cuda_13_when_cuda_12_is_required() {
        let probes = probes_for(
            runtime_probe_candidates(12),
            &[("libcudart.so.13", detected(13))],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Mismatch {
                candidate: runtime_probe_candidates(12)[1],
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_rejects_cuda_12_when_cuda_13_is_required() {
        let probes = probes_for(
            runtime_probe_candidates(13),
            &[("libcudart.so.12", detected(12))],
        );

        assert_eq!(
            evaluate_runtime_probe(13, &probes, true),
            RuntimeProbeDecision::Mismatch {
                candidate: runtime_probe_candidates(13)[1],
                version: 12000,
            }
        );
    }

    #[test]
    fn runtime_probe_accepts_matching_unversioned_runtime() {
        let probes = probes_for(
            runtime_probe_candidates(13),
            &[("libcudart.so", detected(13))],
        );

        assert_eq!(
            evaluate_runtime_probe(13, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: runtime_probe_candidates(13)[2],
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_reports_missing_when_no_runtime_is_available() {
        let probes = probes_for(runtime_probe_candidates(12), &[]);

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Missing
        );
    }

    #[test]
    fn runtime_probe_mismatch_beats_indeterminate_when_enforcing_major_mismatch_first() {
        // Probe order: mismatch encountered before indeterminate.
        // Expected 12, but libcudart.so.13 is detected (mismatch) and
        // libcudart.so yields Indeterminate. Mismatch must win.
        let candidates = runtime_probe_candidates(12);
        let probes = probes_for(
            candidates,
            &[
                ("libcudart.so.13", detected(13)),
                ("libcudart.so", CudaRuntimeProbeResult::Indeterminate),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Mismatch {
                candidate: candidates[1], // libcudart.so.13
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_mismatch_beats_indeterminate_when_enforcing_major_indeterminate_first() {
        // Probe order: indeterminate encountered before mismatch.
        // Expected 12: libcudart.so.12 missing, libcudart.so is Indeterminate,
        // libcudart.so.13 is detected (mismatch). Mismatch must still win.
        let candidates = runtime_probe_candidates(12);
        // Build a custom probe set: [libcudart.so.12=Missing, libcudart.so=Indeterminate, libcudart.so.13=Detected(13)]
        // We can do this by reversing the unversioned/versioned positions.
        let probes: Vec<ProbedCudaRuntime> = vec![
            ProbedCudaRuntime {
                candidate: candidates[0], // libcudart.so.12
                result: CudaRuntimeProbeResult::Missing,
            },
            ProbedCudaRuntime {
                candidate: candidates[2], // libcudart.so
                result: CudaRuntimeProbeResult::Indeterminate,
            },
            ProbedCudaRuntime {
                candidate: candidates[1], // libcudart.so.13
                result: detected(13),
            },
        ];

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Mismatch {
                candidate: candidates[1], // libcudart.so.13
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_mismatch_beats_indeterminate_when_enforcing_major_cuda13_expected() {
        // Mirror test with expected = 13 and detected major = 12.
        let candidates = runtime_probe_candidates(13);
        let probes = probes_for(
            candidates,
            &[
                ("libcudart.so.12", detected(12)),
                ("libcudart.so", CudaRuntimeProbeResult::Indeterminate),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(13, &probes, true),
            RuntimeProbeDecision::Mismatch {
                candidate: candidates[1], // libcudart.so.12
                version: 12000,
            }
        );
    }

    #[test]
    fn runtime_probe_compatible_beats_mismatch_and_indeterminate() {
        // Compatible match must win even when both mismatch and indeterminate are present.
        let candidates = runtime_probe_candidates(12);
        let probes = probes_for(
            candidates,
            &[
                ("libcudart.so.12", detected(12)),
                ("libcudart.so.13", detected(13)),
                ("libcudart.so", CudaRuntimeProbeResult::Indeterminate),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Compatible {
                candidate: candidates[0], // libcudart.so.12
                version: 12000,
            }
        );
    }

    #[test]
    fn runtime_probe_indeterminate_returned_when_no_mismatch_no_compatible() {
        // Only an indeterminate candidate is present; no readable mismatch.
        let candidates = runtime_probe_candidates(12);
        let probes = probes_for(
            candidates,
            &[("libcudart.so", CudaRuntimeProbeResult::Indeterminate)],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, true),
            RuntimeProbeDecision::Indeterminate {
                candidate: candidates[2], // libcudart.so
            }
        );
    }

    #[test]
    fn runtime_probe_load_dynamic_accepts_wrong_major() {
        // enforce_expected_major = false: any readable runtime is Compatible,
        // even if the major does not match expected. The post-loop Mismatch
        // branch is unreachable in this path.
        let probes = probes_for(
            setup_probe_candidates(),
            &[
                ("libcudart.so.13", detected(13)),
                ("libcudart.so", CudaRuntimeProbeResult::Indeterminate),
            ],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, false),
            RuntimeProbeDecision::Compatible {
                candidate: setup_probe_candidates()[1], // libcudart.so.13
                version: 13000,
            }
        );
    }

    #[test]
    fn runtime_probe_is_permissive_for_load_dynamic_builds() {
        let probes = probes_for(
            setup_probe_candidates(),
            &[("libcudart.so.13", detected(13))],
        );

        assert_eq!(
            evaluate_runtime_probe(12, &probes, false),
            RuntimeProbeDecision::Compatible {
                candidate: setup_probe_candidates()[1],
                version: 13000,
            }
        );
    }

    #[test]
    fn setup_detection_reports_only_cuda_12() {
        let probes = probes_for(
            setup_probe_candidates(),
            &[("libcudart.so.12", detected(12))],
        );
        let detected = DetectedCudaRuntimes::from_probes(&probes);

        assert_eq!(detected.majors(), vec![12]);
        assert_eq!(detected.preferred_major(), Some(12));
    }

    #[test]
    fn setup_detection_reports_only_cuda_13() {
        let probes = probes_for(
            setup_probe_candidates(),
            &[("libcudart.so.13", detected(13))],
        );
        let detected = DetectedCudaRuntimes::from_probes(&probes);

        assert_eq!(detected.majors(), vec![13]);
        assert_eq!(detected.preferred_major(), Some(13));
    }

    #[test]
    fn setup_detection_uses_explicit_highest_major_policy_for_mixed_installs() {
        let probes = probes_for(
            setup_probe_candidates(),
            &[
                ("libcudart.so.12", detected(12)),
                ("libcudart.so.13", detected(13)),
            ],
        );
        let detected = DetectedCudaRuntimes::from_probes(&probes);

        assert_eq!(detected.majors(), vec![12, 13]);
        assert_eq!(detected.preferred_major(), Some(13));
    }

    #[test]
    fn setup_detection_uses_unversioned_runtime_as_fallback() {
        let probes = probes_for(setup_probe_candidates(), &[("libcudart.so", detected(13))]);
        let detected = DetectedCudaRuntimes::from_probes(&probes);

        assert_eq!(detected.majors(), vec![13]);
        assert_eq!(detected.preferred_major(), Some(13));
    }

    #[test]
    fn setup_detection_returns_none_when_no_runtime_is_available() {
        let detected =
            DetectedCudaRuntimes::from_probes(&probes_for(setup_probe_candidates(), &[]));

        assert!(detected.is_empty());
        assert_eq!(detected.preferred_major(), None);
    }
}
