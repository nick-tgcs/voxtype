//! Parakeet/ONNX backend management.
//!
//! User-facing wrapper around [`super::binary`] for the legacy
//! `voxtype setup onnx`/`voxtype setup parakeet` CLI.

use super::binary::{self, EngineFamily, Variant};
use crate::cuda_runtime::{self, DetectedCudaRuntimes};
use std::path::Path;

/// Parakeet backend variants exposed to existing callers (status formatting,
/// CLI dispatch). Each maps to one [`Variant`] in the `Onnx` family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParakeetBackend {
    Avx2,
    Avx512,
    /// CUDA 12.x (NVIDIA, ort built against libcudart.so.12)
    Cuda12,
    /// CUDA 13.x (NVIDIA, ort built against libcudart.so.13, requires driver 580+)
    Cuda13,
    /// Unversioned CUDA binary (source-built or pre-0.7.0).
    Cuda,
    Migraphx,
    /// Custom binary (source-compiled without specific suffix)
    Custom,
}

impl ParakeetBackend {
    fn variant(self) -> Variant {
        match self {
            ParakeetBackend::Avx2 => Variant::OnnxAvx2,
            ParakeetBackend::Avx512 => Variant::OnnxAvx512,
            ParakeetBackend::Cuda12 => Variant::OnnxCuda12,
            ParakeetBackend::Cuda13 => Variant::OnnxCuda13,
            ParakeetBackend::Cuda => Variant::OnnxCuda,
            ParakeetBackend::Migraphx => Variant::OnnxMigraphx,
            ParakeetBackend::Custom => Variant::OnnxNative,
        }
    }

    fn from_variant(v: Variant) -> Option<Self> {
        match v {
            Variant::OnnxAvx2 => Some(ParakeetBackend::Avx2),
            Variant::OnnxAvx512 => Some(ParakeetBackend::Avx512),
            Variant::OnnxCuda12 => Some(ParakeetBackend::Cuda12),
            Variant::OnnxCuda13 => Some(ParakeetBackend::Cuda13),
            Variant::OnnxCuda => Some(ParakeetBackend::Cuda),
            Variant::OnnxMigraphx => Some(ParakeetBackend::Migraphx),
            Variant::OnnxNative => Some(ParakeetBackend::Custom),
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        self.variant().display()
    }

    fn whisper_equivalent(&self) -> Variant {
        match self {
            ParakeetBackend::Avx2 => Variant::WhisperAvx2,
            ParakeetBackend::Avx512 => Variant::WhisperAvx512,
            // GPU users get Vulkan as the closest Whisper equivalent.
            ParakeetBackend::Cuda12
            | ParakeetBackend::Cuda13
            | ParakeetBackend::Cuda
            | ParakeetBackend::Migraphx => Variant::WhisperVulkan,
            ParakeetBackend::Custom => Variant::WhisperNative,
        }
    }
}

/// True if the active variant is in the ONNX family.
pub fn is_parakeet_active() -> bool {
    binary::active_variant()
        .map(|v| v.family() == EngineFamily::Onnx)
        .unwrap_or(false)
}

pub fn detect_current_parakeet_backend() -> Option<ParakeetBackend> {
    ParakeetBackend::from_variant(binary::active_variant()?)
}

pub fn detect_available_backends() -> Vec<ParakeetBackend> {
    binary::enumerate_installed()
        .into_iter()
        .filter_map(ParakeetBackend::from_variant)
        .collect()
}

/// Detect which Whisper backend is currently active (legacy helper retained
/// for `show_status` output).
fn detect_current_whisper_variant() -> Option<Variant> {
    binary::active_variant().filter(|v| v.family() == EngineFamily::Whisper)
}

/// Pick the best ONNX variant for this system.
fn detect_best_parakeet_backend() -> Option<ParakeetBackend> {
    let host_cuda = detect_cuda_runtimes();
    detect_best_parakeet_backend_for_inventory(&binary::inventory(), &host_cuda)
}

// Keep the mixed-install CUDA policy centralized here so `setup onnx --enable`,
// `setup gpu --enable`, and the status path all make the same CUDA 12 vs 13
// choice instead of each depending on local probe order.
fn detect_best_parakeet_backend_for_inventory(
    inv: &binary::Inventory,
    host_cuda: &DetectedCudaRuntimes,
) -> Option<ParakeetBackend> {
    let installed_onnx: Vec<&binary::VariantStatus> = inv
        .variants
        .iter()
        .filter(|s| s.installed && s.variant.family() == EngineFamily::Onnx)
        .collect();

    if installed_onnx.is_empty() {
        return None;
    }

    // Prefer CUDA on NVIDIA hosts. cu12 vs cu13 binaries differ only in which
    // ONNX Runtime prebuilt they bundle (libcudart.so.12 vs .13); pick the one
    // matching one the host can run. Mixed 12/13 installs explicitly prefer
    // the highest detected major; a no-detection fallback still prefers cu13
    // first because that's the rolling-distro default.
    for v in preferred_cuda_variant_order(host_cuda) {
        if let Some(status) = installed_onnx.iter().find(|s| s.variant == v) {
            if status.runs_on_this_cpu && status.gpu_available {
                return ParakeetBackend::from_variant(v);
            }
        }
    }

    // Then MIGraphX, then CPU-only backends.
    let preference = [
        Variant::OnnxMigraphx,
        Variant::OnnxAvx512,
        Variant::OnnxAvx2,
        Variant::OnnxNative,
    ];
    for v in preference {
        if let Some(status) = installed_onnx.iter().find(|s| s.variant == v) {
            if status.runs_on_this_cpu && status.gpu_available {
                return ParakeetBackend::from_variant(v);
            }
        }
    }
    // Fall back to whatever's installed even if the heuristic warns against it.
    installed_onnx
        .first()
        .and_then(|s| ParakeetBackend::from_variant(s.variant))
}

/// Ordered CUDA variant preference derived from the explicit setup policy.
/// Mixed installs prefer the highest detected runtime major; no-detection still
/// falls back to a stable documented order rather than whatever `dlopen` found.
pub(crate) fn preferred_cuda_variant_order(host_cuda: &DetectedCudaRuntimes) -> [Variant; 3] {
    match host_cuda.preferred_major() {
        Some(13) => [Variant::OnnxCuda13, Variant::OnnxCuda, Variant::OnnxCuda12],
        Some(12) => [Variant::OnnxCuda12, Variant::OnnxCuda, Variant::OnnxCuda13],
        // Host CUDA detection failed entirely; prefer cu13 first as the
        // rolling-distro default, then try cu12, then the generic legacy name.
        _ => [Variant::OnnxCuda13, Variant::OnnxCuda12, Variant::OnnxCuda],
    }
}

/// Detect all CUDA runtime majors the host can report via libcudart sonames.
/// Mixed installs are returned as a set of majors; callers that need a single
/// answer should use `preferred_major()`, which explicitly prefers the highest
/// detected major rather than whichever soname happened to open first.
pub(crate) fn detect_cuda_runtimes() -> DetectedCudaRuntimes {
    let probes = cuda_runtime::probe_system_cuda_runtimes(cuda_runtime::setup_probe_candidates());
    DetectedCudaRuntimes::from_probes(&probes)
}

/// Detect the host's preferred CUDA runtime major.
///
/// When both CUDA 12 and CUDA 13 are installed, setup explicitly prefers the
/// highest detected major. When only the unversioned `libcudart.so` exists, the
/// reported major from that library becomes the preference.
pub fn detect_cuda_runtime_major() -> Option<i32> {
    detect_cuda_runtimes().preferred_major()
}

pub fn show_status() {
    println!("=== Voxtype ONNX Engine Status ===\n");

    if is_parakeet_active() {
        if let Some(backend) = detect_current_parakeet_backend() {
            println!("Active engine: Parakeet");
            println!("  Backend: {}", backend.display_name());
            println!(
                "  Binary: {}",
                Path::new(binary::LIB_DIR)
                    .join(backend.variant().binary_name())
                    .display()
            );
        }
    } else {
        println!("Active engine: Whisper");
        if let Some(variant) = detect_current_whisper_variant() {
            println!(
                "  Binary: {}",
                Path::new(binary::LIB_DIR)
                    .join(variant.binary_name())
                    .display()
            );
        }
    }

    println!("\nAvailable ONNX backends:");
    let available = detect_available_backends();
    let current = detect_current_parakeet_backend();

    if available.is_empty() {
        println!("  No ONNX binaries installed.");
        println!("\n  Install an ONNX-enabled voxtype package to use this feature.");
    } else {
        for backend in [
            ParakeetBackend::Avx2,
            ParakeetBackend::Avx512,
            ParakeetBackend::Cuda12,
            ParakeetBackend::Cuda13,
            ParakeetBackend::Cuda,
            ParakeetBackend::Migraphx,
            ParakeetBackend::Custom,
        ] {
            let installed = available.contains(&backend);
            let active = current == Some(backend);
            let status = if active {
                "active"
            } else if installed {
                "installed"
            } else {
                "not installed"
            };
            println!("  {} - {}", backend.display_name(), status);
        }
    }

    // GPU detection for CUDA/MIGraphX
    println!();
    let gpus = binary::detect_gpus();
    let cpu = binary::detect_cpu();
    if gpus.nvidia {
        println!("NVIDIA GPU: detected");
    }
    if gpus.amd {
        println!("AMD GPU: detected");
    }
    if !gpus.nvidia && !gpus.amd {
        println!("GPU: not detected");
    }
    if (gpus.nvidia || gpus.amd) && !cpu.avx512 {
        println!("\nNote: ONNX GPU binaries (CUDA/MIGraphX) require AVX-512 CPU support.");
        println!("  Your CPU supports AVX2 only. Use ONNX (AVX2) for CPU-based inference,");
        println!("  or use the Whisper engine with Vulkan for GPU acceleration.");
    }

    println!();
    if !is_parakeet_active() && !available.is_empty() {
        println!("To enable ONNX engines:");
        println!("  sudo voxtype setup onnx --enable");
    } else if is_parakeet_active() {
        println!("To switch back to Whisper:");
        println!("  sudo voxtype setup onnx --disable");
    }
}

pub fn enable() -> anyhow::Result<()> {
    let available = detect_available_backends();
    if available.is_empty() {
        anyhow::bail!(
            "No ONNX binaries installed.\n\
             Install an ONNX-enabled voxtype package first."
        );
    }

    if is_parakeet_active() {
        println!("ONNX engine is already enabled.");
        if let Some(backend) = detect_current_parakeet_backend() {
            println!("  Current backend: {}", backend.display_name());
        }
        return Ok(());
    }

    let backend = detect_best_parakeet_backend()
        .ok_or_else(|| anyhow::anyhow!("No suitable ONNX backend found"))?;

    binary::switch_to(backend.variant())?;

    if super::systemd::regenerate_service_file()? {
        println!("Updated systemd service to use ONNX backend.");
    }

    println!("Switched to {} backend.", backend.display_name());
    println!();
    println!("Restart voxtype to use ONNX engines:");
    println!("  systemctl --user restart voxtype");

    Ok(())
}

pub fn disable() -> anyhow::Result<()> {
    if !is_parakeet_active() {
        println!("ONNX engine is not currently enabled (already using Whisper).");
        return Ok(());
    }

    let preferred = detect_current_parakeet_backend()
        .map(|b| b.whisper_equivalent())
        .unwrap_or(Variant::WhisperAvx2);

    let installed = binary::enumerate_installed();
    let target = if installed.contains(&preferred) {
        preferred
    } else {
        // Fall back to any installed Whisper variant in this preference order.
        let order = [
            Variant::WhisperAvx512,
            Variant::WhisperAvx2,
            Variant::WhisperVulkan,
            Variant::WhisperNative,
        ];
        let chosen = order
            .iter()
            .find(|v| installed.contains(v))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("No Whisper backend found to switch to"))?;
        if chosen != preferred {
            eprintln!(
                "Note: {} not found, using {} instead",
                preferred.binary_name(),
                chosen.binary_name()
            );
        }
        chosen
    };

    binary::switch_to(target)?;

    if super::systemd::regenerate_service_file()? {
        println!("Updated systemd service to use Whisper backend.");
    }

    println!(
        "Switched to Whisper ({}) backend.",
        target.binary_name().trim_start_matches("voxtype-")
    );
    println!();
    println!("Restart voxtype to use Whisper:");
    println!("  systemctl --user restart voxtype");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::binary::{Cpu, Gpus, InstallKind, Recommendation, VariantStatus};
    use std::path::PathBuf;

    fn detected_runtimes(majors: &[i32]) -> DetectedCudaRuntimes {
        let probes = majors
            .iter()
            .map(|major| cuda_runtime::ProbedCudaRuntime {
                candidate: cuda_runtime::runtime_probe_candidates(*major)[0],
                result: cuda_runtime::CudaRuntimeProbeResult::Detected {
                    version: major * 1000,
                },
            })
            .collect::<Vec<_>>();
        DetectedCudaRuntimes::from_probes(&probes)
    }

    fn inventory_with_installed(installed: &[Variant]) -> binary::Inventory {
        let cpu = Cpu {
            avx2: true,
            avx512: true,
        };
        let gpus = Gpus {
            nvidia: true,
            amd: false,
        };
        binary::Inventory {
            install_kind: InstallKind::Package,
            binary_path: PathBuf::from("/usr/bin/voxtype"),
            package_lib_dir: Some(PathBuf::from(binary::LIB_DIR)),
            active_variant: None,
            variants: Variant::ALL
                .iter()
                .map(|variant| VariantStatus {
                    variant: *variant,
                    binary_name: variant.binary_name().to_string(),
                    installed: installed.contains(variant),
                    runs_on_this_cpu: true,
                    gpu_available: true,
                    active: false,
                })
                .collect(),
            cpu: cpu.clone(),
            gpus: gpus.clone(),
            compiled_features: vec![],
            recommendation: Recommendation {
                whisper: Variant::WhisperVulkan,
                whisper_reason: "test",
                onnx: Variant::OnnxCuda13,
                onnx_reason: "test",
                primary: Variant::WhisperVulkan,
            },
        }
    }

    #[test]
    fn parakeet_backend_round_trip() {
        for b in [
            ParakeetBackend::Avx2,
            ParakeetBackend::Avx512,
            ParakeetBackend::Cuda12,
            ParakeetBackend::Cuda13,
            ParakeetBackend::Cuda,
            ParakeetBackend::Migraphx,
            ParakeetBackend::Custom,
        ] {
            assert_eq!(ParakeetBackend::from_variant(b.variant()), Some(b));
        }
    }

    #[test]
    fn parakeet_backend_binary_names() {
        assert_eq!(
            ParakeetBackend::Cuda12.variant().binary_name(),
            "voxtype-onnx-cuda-12"
        );
        assert_eq!(
            ParakeetBackend::Cuda13.variant().binary_name(),
            "voxtype-onnx-cuda-13"
        );
        assert_eq!(
            ParakeetBackend::Migraphx.variant().binary_name(),
            "voxtype-onnx-migraphx"
        );
    }

    #[test]
    fn whisper_variants_dont_resolve_to_parakeet() {
        for v in [
            Variant::WhisperAvx2,
            Variant::WhisperAvx512,
            Variant::WhisperVulkan,
            Variant::WhisperNative,
        ] {
            assert_eq!(ParakeetBackend::from_variant(v), None);
        }
    }

    #[test]
    fn whisper_equivalents_are_whisper() {
        for b in [
            ParakeetBackend::Avx2,
            ParakeetBackend::Avx512,
            ParakeetBackend::Cuda12,
            ParakeetBackend::Cuda13,
            ParakeetBackend::Cuda,
            ParakeetBackend::Migraphx,
            ParakeetBackend::Custom,
        ] {
            assert_eq!(b.whisper_equivalent().family(), EngineFamily::Whisper);
        }
    }

    #[test]
    fn is_parakeet_active_does_not_panic() {
        let _ = is_parakeet_active();
    }

    #[test]
    fn detect_available_backends_returns_vec() {
        let backends = detect_available_backends();
        assert!(backends.len() <= 5);
    }

    #[test]
    fn test_backend_enum_equality() {
        assert_eq!(ParakeetBackend::Avx2, ParakeetBackend::Avx2);
        assert_ne!(ParakeetBackend::Avx2, ParakeetBackend::Avx512);
        assert_ne!(ParakeetBackend::Avx512, ParakeetBackend::Cuda12);
        assert_ne!(ParakeetBackend::Cuda12, ParakeetBackend::Cuda13);
    }

    #[test]
    fn test_backend_clone() {
        let backend = ParakeetBackend::Cuda12;
        let cloned = backend;
        assert_eq!(backend, cloned);
    }

    #[test]
    fn mixed_cuda_hosts_explicitly_prefer_cuda_13() {
        let detected = detected_runtimes(&[12, 13]);

        assert_eq!(
            preferred_cuda_variant_order(&detected)[0],
            Variant::OnnxCuda13
        );
    }

    #[test]
    fn backend_selection_uses_explicit_mixed_install_policy() {
        let inventory = inventory_with_installed(&[Variant::OnnxCuda12, Variant::OnnxCuda13]);
        let detected = detected_runtimes(&[12, 13]);

        assert_eq!(
            detect_best_parakeet_backend_for_inventory(&inventory, &detected),
            Some(ParakeetBackend::Cuda13)
        );
    }

    #[test]
    fn backend_selection_falls_back_when_preferred_cuda_variant_is_missing() {
        let inventory = inventory_with_installed(&[Variant::OnnxCuda12]);
        let detected = detected_runtimes(&[12, 13]);

        assert_eq!(
            detect_best_parakeet_backend_for_inventory(&inventory, &detected),
            Some(ParakeetBackend::Cuda12)
        );
    }
}
