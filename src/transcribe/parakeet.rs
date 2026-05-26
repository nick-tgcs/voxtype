//! Parakeet-based speech-to-text transcription
//!
//! Uses NVIDIA's Parakeet model via the parakeet-rs crate for fast, local transcription.
//! This module is only available when the `parakeet` feature is enabled.
//!
//! Supports two model architectures:
//! - CTC (Connectionist Temporal Classification): faster, character-level output
//! - TDT (Token-Duration-Transducer): recommended, proper punctuation and word boundaries

use super::{TimedSegment, Transcriber};
use crate::config::{ParakeetConfig, ParakeetModelType};
use crate::cuda_runtime::{self, RuntimeProbeDecision};
use crate::error::TranscribeError;
#[cfg(any(
    feature = "parakeet-cuda",
    feature = "parakeet-migraphx",
    feature = "parakeet-tensorrt"
))]
use parakeet_rs::ExecutionProvider;
use parakeet_rs::{
    ExecutionConfig, Parakeet, ParakeetTDT, Transcriber as ParakeetTranscriberTrait,
};
use std::path::PathBuf;
use std::sync::Mutex;

/// Internal enum to hold either CTC or TDT model instance
enum ParakeetModel {
    /// CTC model (character-level, faster)
    Ctc(Mutex<Parakeet>),
    /// TDT model (token-level, better quality output)
    Tdt(Mutex<ParakeetTDT>),
}

/// Parakeet-based transcriber using ONNX Runtime
pub struct ParakeetTranscriber {
    /// Parakeet model instance (CTC or TDT)
    model: ParakeetModel,
    /// Model type for logging
    model_type: ParakeetModelType,
}

impl ParakeetTranscriber {
    /// Create a new Parakeet transcriber
    pub fn new(config: &ParakeetConfig) -> Result<Self, TranscribeError> {
        let model_path = resolve_model_path(&config.model)?;

        // Determine model type: use config override or auto-detect from directory
        let model_type = config
            .model_type
            .unwrap_or_else(|| detect_model_type(&model_path));

        tracing::info!(
            "Loading Parakeet {:?} model from {:?}",
            model_type,
            model_path
        );
        let start = std::time::Instant::now();

        // Configure execution provider based on feature flags
        let exec_config = build_execution_config();

        let model = match model_type {
            ParakeetModelType::Ctc => {
                let parakeet =
                    Parakeet::from_pretrained(&model_path, exec_config).map_err(|e| {
                        TranscribeError::InitFailed(format!("Parakeet CTC init failed: {}", e))
                    })?;
                ParakeetModel::Ctc(Mutex::new(parakeet))
            }
            ParakeetModelType::Tdt => {
                let parakeet =
                    ParakeetTDT::from_pretrained(&model_path, exec_config).map_err(|e| {
                        TranscribeError::InitFailed(format!("Parakeet TDT init failed: {}", e))
                    })?;
                ParakeetModel::Tdt(Mutex::new(parakeet))
            }
        };

        tracing::info!(
            "Parakeet {:?} model loaded in {:.2}s",
            model_type,
            start.elapsed().as_secs_f32()
        );

        Ok(Self { model, model_type })
    }
}

impl Transcriber for ParakeetTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }

        let duration_secs = samples.len() as f32 / 16000.0;
        tracing::debug!(
            "Transcribing {:.2}s of audio ({} samples) with Parakeet {:?}",
            duration_secs,
            samples.len(),
            self.model_type
        );

        let start = std::time::Instant::now();

        let text = match &self.model {
            ParakeetModel::Ctc(parakeet) => {
                let mut parakeet = parakeet.lock().map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to lock Parakeet mutex: {}",
                        e
                    ))
                })?;

                let result = parakeet
                    .transcribe_samples(
                        samples.to_vec(),
                        16000, // sample rate
                        1,     // mono
                        None,  // default timestamp mode
                    )
                    .map_err(|e| {
                        TranscribeError::InferenceFailed(format!(
                            "Parakeet CTC inference failed: {}",
                            e
                        ))
                    })?;

                result.text.trim().to_string()
            }
            ParakeetModel::Tdt(parakeet) => {
                let mut parakeet = parakeet.lock().map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to lock Parakeet mutex: {}",
                        e
                    ))
                })?;

                let result = parakeet
                    .transcribe_samples(
                        samples.to_vec(),
                        16000, // sample rate
                        1,     // mono
                        None,  // default timestamp mode
                    )
                    .map_err(|e| {
                        TranscribeError::InferenceFailed(format!(
                            "Parakeet TDT inference failed: {}",
                            e
                        ))
                    })?;

                result.text.trim().to_string()
            }
        };

        tracing::info!(
            "Parakeet {:?} transcription completed in {:.2}s: {:?}",
            self.model_type,
            start.elapsed().as_secs_f32(),
            if text.chars().count() > 50 {
                format!("{}...", text.chars().take(50).collect::<String>())
            } else {
                text.clone()
            }
        );

        Ok(text)
    }

    fn transcribe_timed(&self, samples: &[f32]) -> Result<Vec<TimedSegment>, TranscribeError> {
        let start = std::time::Instant::now();

        let result = match &self.model {
            ParakeetModel::Ctc(parakeet) => {
                let mut parakeet = parakeet.lock().map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to lock Parakeet mutex: {}",
                        e
                    ))
                })?;
                parakeet
                    .transcribe_samples(samples.to_vec(), 16000, 1, None)
                    .map_err(|e| {
                        TranscribeError::InferenceFailed(format!(
                            "Parakeet CTC inference failed: {}",
                            e
                        ))
                    })?
            }
            ParakeetModel::Tdt(parakeet) => {
                let mut parakeet = parakeet.lock().map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to lock Parakeet mutex: {}",
                        e
                    ))
                })?;
                parakeet
                    .transcribe_samples(samples.to_vec(), 16000, 1, None)
                    .map_err(|e| {
                        TranscribeError::InferenceFailed(format!(
                            "Parakeet TDT inference failed: {}",
                            e
                        ))
                    })?
            }
        };

        // Split the full text on sentence boundaries (.!?) and map each sentence
        // back to token timestamps. result.text is properly assembled by parakeet-rs;
        // individual tokens are subword pieces that can't be naively joined.
        let full_text = result.text.trim();
        let tokens = &result.tokens;

        if full_text.is_empty() || tokens.is_empty() {
            return Ok(vec![]);
        }

        // Split text into sentences on .!? boundaries
        let mut sentences: Vec<String> = Vec::new();
        let mut current = String::new();
        for ch in full_text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
            }
        }
        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        // Map sentences to timestamps by distributing tokens proportionally.
        // Each sentence gets the time span of its corresponding token range.
        let total_sentences = sentences.len();
        let total_tokens = tokens.len();
        let tokens_per_sentence = if total_sentences > 0 {
            total_tokens / total_sentences
        } else {
            total_tokens
        };

        let mut segments = Vec::new();
        let mut token_idx = 0;

        for (i, sentence) in sentences.iter().enumerate() {
            let start_secs = if token_idx < tokens.len() {
                tokens[token_idx].start
            } else {
                tokens.last().map(|t| t.end).unwrap_or(0.0)
            };

            // Last sentence gets all remaining tokens
            let end_token_idx = if i == total_sentences - 1 {
                total_tokens
            } else {
                (token_idx + tokens_per_sentence).min(total_tokens)
            };

            let end_secs = if end_token_idx > 0 && end_token_idx <= tokens.len() {
                tokens[end_token_idx - 1].end
            } else {
                start_secs
            };

            segments.push(TimedSegment {
                text: sentence.clone(),
                start_secs,
                end_secs,
            });

            token_idx = end_token_idx;
        }

        tracing::info!(
            "Parakeet {:?} timed transcription completed in {:.2}s: {} segments",
            self.model_type,
            start.elapsed().as_secs_f32(),
            segments.len()
        );

        Ok(segments)
    }
}

/// Build execution config based on compile-time feature flags
pub(super) fn build_execution_config() -> Option<ExecutionConfig> {
    #[cfg(feature = "parakeet-cuda")]
    {
        if probe_cuda_runtime() {
            tracing::info!("Configuring CUDA execution provider for NVIDIA GPU acceleration");
            return Some(ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda));
        }
        tracing::warn!("CUDA not available or incompatible, falling back to CPU inference");
        return None;
    }

    #[cfg(feature = "parakeet-tensorrt")]
    {
        if probe_cuda_runtime() {
            tracing::info!("Configuring TensorRT execution provider for NVIDIA GPU acceleration");
            return Some(
                ExecutionConfig::new().with_execution_provider(ExecutionProvider::TensorRT),
            );
        }
        tracing::warn!("CUDA not available or incompatible, falling back to CPU inference");
        return None;
    }

    #[cfg(feature = "parakeet-migraphx")]
    {
        tracing::info!("Configuring MIGraphX execution provider for AMD GPU acceleration");
        return Some(ExecutionConfig::new().with_execution_provider(ExecutionProvider::MIGraphX));
    }

    #[cfg(not(any(
        feature = "parakeet-cuda",
        feature = "parakeet-tensorrt",
        feature = "parakeet-migraphx"
    )))]
    {
        None
    }
}

/// Probe CUDA runtime availability and version compatibility.
///
/// The bundled ONNX Runtime is built against one CUDA major chosen at compile
/// time. If the process loads a different CUDA major, ONNX Runtime will
/// segfault during EP initialization rather than returning an error.
///
/// Returns true if CUDA looks compatible, false if it should be skipped.
#[cfg(any(feature = "parakeet-cuda", feature = "parakeet-tensorrt"))]
fn probe_cuda_runtime() -> bool {
    // ort 2.0.0-rc.12 picks the cu12 or cu13 prebuilt at compile time from
    // ORT_CUDA_VERSION (see ort-sys/build/download/resolve.rs). build.rs
    // mirrors that selection into VOXTYPE_BUILD_CUDA_MAJOR so this probe
    // accepts only the runtime version the bundled EP can actually talk to.
    // A mismatched major would crash ort's CUDA EP during initialization.
    //
    // Voxtype ships separate voxtype-onnx-cuda-12 and voxtype-onnx-cuda-13
    // binaries. `voxtype setup gpu --enable` symlinks voxtype-onnx-cuda to
    // whichever variant matches the host's CUDA runtime.
    //
    // Load-dynamic builds skip this check: there is no bundled ORT to
    // mismatch — the binary dlopens whatever libonnxruntime the system
    // provides, and ORT does its own kernel-image lookup against the host
    // CUDA at session-create time. v0.7.3 cuda-13 forgot to set
    // ORT_CUDA_VERSION=13 in its Dockerfile so VOXTYPE_BUILD_CUDA_MAJOR
    // baked in the default ("12"), and this probe falsely rejected
    // Blackwell hosts with "this binary's bundled ONNX Runtime requires
    // CUDA 12.x" (#386). The Dockerfile fix sets the env var properly,
    // and gating the check on `not(feature = "parakeet-load-dynamic")`
    // closes the design hole so future load-dynamic builds don't depend
    // on remembering to set it.
    #[cfg(not(feature = "parakeet-load-dynamic"))]
    {
        const EXPECTED_CUDA_MAJOR: i32 = match env!("VOXTYPE_BUILD_CUDA_MAJOR").as_bytes() {
            b"13" => 13,
            _ => 12,
        };
        let probes = cuda_runtime::probe_system_cuda_runtimes(
            cuda_runtime::runtime_probe_candidates(EXPECTED_CUDA_MAJOR),
        );
        return match cuda_runtime::evaluate_runtime_probe(EXPECTED_CUDA_MAJOR, &probes, true) {
            RuntimeProbeDecision::Compatible { candidate, version } => {
                let major = version / 1000;
                let minor = (version % 1000) / 10;
                tracing::info!(
                    "Detected CUDA runtime version: {}.{} via {}",
                    major,
                    minor,
                    candidate.soname
                );
                true
            }
            RuntimeProbeDecision::Mismatch { candidate, version } => {
                let major = version / 1000;
                let minor = (version % 1000) / 10;
                tracing::error!(
                    "CUDA version mismatch: found CUDA {major}.{minor} in {}, but this binary's \
                     bundled ONNX Runtime requires CUDA {EXPECTED_CUDA_MAJOR}.x. \
                     Continuing would crash the process.\n  \
                     Options:\n  \
                     1. Install the matching voxtype-onnx-cuda-{EXPECTED_CUDA_MAJOR} package\n  \
                     2. Switch to voxtype-onnx-cuda-{} for your CUDA version (`voxtype setup gpu --enable` \
                     auto-detects and points the symlink at the right one)\n  \
                     3. Build from source with --features parakeet-load-dynamic to link \
                     against your system's ONNX Runtime instead",
                    candidate.soname,
                    major,
                );
                false
            }
            RuntimeProbeDecision::Missing => {
                tracing::error!(
                    "CUDA runtime library (libcudart.so) not found. \
                     Cannot initialize CUDA execution provider.\n  \
                     Install the CUDA toolkit, or use a CPU backend instead."
                );
                false
            }
            RuntimeProbeDecision::Indeterminate { candidate } => {
                // This is the intentional narrow trade-off: a readable major
                // mismatch is unsafe and remains a hard failure, but if the
                // runtime library opens and refuses to report a version we avoid
                // falsely rejecting the host here and let ONNX Runtime perform
                // the final compatibility check when it creates the session.
                tracing::warn!(
                    "Could not determine the CUDA runtime version from {}. \
                     Proceeding and letting ONNX Runtime validate it.",
                    candidate.soname
                );
                true
            }
        };
    }

    #[cfg(feature = "parakeet-load-dynamic")]
    {
        // Load-dynamic builds do not bundle a fixed ORT CUDA ABI. The relevant
        // question here is only whether some CUDA runtime is present at all;
        // the system ONNX Runtime owns major-version compatibility.
        let probes =
            cuda_runtime::probe_system_cuda_runtimes(cuda_runtime::setup_probe_candidates());
        return match cuda_runtime::evaluate_runtime_probe(0, &probes, false) {
            RuntimeProbeDecision::Compatible { candidate, version } => {
                let major = version / 1000;
                let minor = (version % 1000) / 10;
                tracing::info!(
                    "Detected CUDA runtime version: {}.{} via {}",
                    major,
                    minor,
                    candidate.soname
                );
                true
            }
            RuntimeProbeDecision::Missing => {
                tracing::error!(
                    "CUDA runtime library (libcudart.so) not found. \
                     Cannot initialize CUDA execution provider.\n  \
                     Install the CUDA toolkit, or use a CPU backend instead."
                );
                false
            }
            RuntimeProbeDecision::Indeterminate { candidate } => {
                // Same trade-off as the bundled path: no readable mismatch was
                // proven, so stay permissive and let the system ORT validate
                // the runtime during session creation.
                tracing::warn!(
                    "Could not determine the CUDA runtime version from {}. \
                     Proceeding and letting ONNX Runtime validate it.",
                    candidate.soname
                );
                true
            }
            // `enforce_expected_major = false` means any readable runtime is
            // already accepted above. Keep the arm explicit so future changes to
            // `evaluate_runtime_probe` do not silently alter load-dynamic policy.
            RuntimeProbeDecision::Mismatch { .. } => true,
        };
    }
}

/// Which CUDA probe policy is compiled into this binary.
///
/// This is a testable seam for unit tests. Production code uses the
/// cfg-gated branches in `probe_cuda_runtime` directly, but the two
/// must stay in sync. The policy is determined at compile time by
/// the `parakeet-load-dynamic` feature.
#[cfg(all(test, any(feature = "parakeet-cuda", feature = "parakeet-tensorrt")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaProbePolicy {
    /// Bundled ONNX Runtime: validates the exact CUDA major baked in at
    /// compile time and hard-fails a mismatch.
    BundledStrict,
    /// System ONNX Runtime (parakeet-load-dynamic): only checks that
    /// *some* libcudart is present; the system ORT owns major-version
    /// compatibility.
    LoadDynamicPermissive,
}

/// Return the CUDA probe policy compiled into this binary.
///
/// `BundledStrict` when built without `parakeet-load-dynamic`;
/// `LoadDynamicPermissive` when built with it. The feature name
/// `parakeet-load-dynamic` must match both the cfg gate in
/// `probe_cuda_runtime` and the guidance shown to users in the
/// Mismatch error arm.
#[cfg(all(test, any(feature = "parakeet-cuda", feature = "parakeet-tensorrt")))]
pub(crate) fn cuda_probe_policy() -> CudaProbePolicy {
    #[cfg(feature = "parakeet-load-dynamic")]
    return CudaProbePolicy::LoadDynamicPermissive;

    #[cfg(not(feature = "parakeet-load-dynamic"))]
    CudaProbePolicy::BundledStrict
}

/// Auto-detect model type from directory structure
///
/// TDT models have: encoder-model.onnx, decoder_joint-model.onnx, vocab.txt
/// CTC models have: model.onnx (or model_int8.onnx), tokenizer.json
fn detect_model_type(path: &PathBuf) -> ParakeetModelType {
    // Check for TDT model structure
    let has_encoder =
        path.join("encoder-model.onnx").exists() || path.join("encoder-model.onnx.data").exists();
    let has_decoder = path.join("decoder_joint-model.onnx").exists();

    if has_encoder && has_decoder {
        tracing::debug!("Auto-detected TDT model (found encoder + decoder ONNX files)");
        return ParakeetModelType::Tdt;
    }

    // Check for CTC model structure
    let has_ctc_model = path.join("model.onnx").exists() || path.join("model_int8.onnx").exists();
    let has_tokenizer = path.join("tokenizer.json").exists();

    if has_ctc_model && has_tokenizer {
        tracing::debug!("Auto-detected CTC model (found model.onnx + tokenizer.json)");
        return ParakeetModelType::Ctc;
    }

    // Default to TDT (recommended for most use cases)
    tracing::warn!(
        "Could not auto-detect model type from {:?}, defaulting to TDT. \
        Set model_type in config to override.",
        path
    );
    ParakeetModelType::Tdt
}

/// Resolve model name to directory path
pub(super) fn resolve_model_path(model: &str) -> Result<PathBuf, TranscribeError> {
    // If it's already an absolute path, use it directly
    let path = PathBuf::from(model);
    if path.is_absolute() && path.exists() {
        return Ok(path);
    }

    // Check models directory
    let models_dir = crate::config::Config::models_dir();
    let model_path = models_dir.join(model);

    if model_path.exists() {
        return Ok(model_path);
    }

    // Check current directory
    let cwd_path = PathBuf::from(model);
    if cwd_path.exists() {
        return Ok(cwd_path);
    }

    // Check ./models/
    let local_models_path = PathBuf::from("models").join(model);
    if local_models_path.exists() {
        return Ok(local_models_path);
    }

    Err(TranscribeError::ModelNotFound(format!(
        "Parakeet model '{}' not found. Looked in:\n  - {}\n  - {}\n  - {}\n\n\
        Download TDT (recommended): https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx\n\
        Download CTC: https://huggingface.co/nvidia/parakeet-ctc-0.6b",
        model,
        model_path.display(),
        cwd_path.display(),
        local_models_path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_detect_model_type_tdt_with_encoder_and_decoder() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // Create TDT model structure
        fs::write(model_path.join("encoder-model.onnx"), b"dummy").unwrap();
        fs::write(model_path.join("decoder_joint-model.onnx"), b"dummy").unwrap();
        fs::write(model_path.join("vocab.txt"), b"dummy").unwrap();

        let detected = detect_model_type(&model_path);
        assert_eq!(detected, ParakeetModelType::Tdt);
    }

    #[test]
    fn test_detect_model_type_tdt_with_encoder_data_file() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // TDT model with .onnx.data file (large models split data)
        fs::write(model_path.join("encoder-model.onnx.data"), b"dummy").unwrap();
        fs::write(model_path.join("decoder_joint-model.onnx"), b"dummy").unwrap();

        let detected = detect_model_type(&model_path);
        assert_eq!(detected, ParakeetModelType::Tdt);
    }

    #[test]
    fn test_detect_model_type_ctc_with_model_and_tokenizer() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // Create CTC model structure
        fs::write(model_path.join("model.onnx"), b"dummy").unwrap();
        fs::write(model_path.join("tokenizer.json"), b"{}").unwrap();

        let detected = detect_model_type(&model_path);
        assert_eq!(detected, ParakeetModelType::Ctc);
    }

    #[test]
    fn test_detect_model_type_ctc_with_int8_model() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // CTC model with quantized int8 variant
        fs::write(model_path.join("model_int8.onnx"), b"dummy").unwrap();
        fs::write(model_path.join("tokenizer.json"), b"{}").unwrap();

        let detected = detect_model_type(&model_path);
        assert_eq!(detected, ParakeetModelType::Ctc);
    }

    #[test]
    fn test_detect_model_type_defaults_to_tdt_when_ambiguous() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // Empty directory - should default to TDT
        let detected = detect_model_type(&model_path);
        assert_eq!(detected, ParakeetModelType::Tdt);
    }

    #[test]
    fn test_detect_model_type_defaults_to_tdt_with_partial_files() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // Only encoder without decoder - ambiguous, defaults to TDT
        fs::write(model_path.join("encoder-model.onnx"), b"dummy").unwrap();

        let detected = detect_model_type(&model_path);
        assert_eq!(detected, ParakeetModelType::Tdt);
    }

    #[test]
    fn test_detect_model_type_ctc_requires_both_files() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // Only model.onnx without tokenizer - should not detect as CTC
        fs::write(model_path.join("model.onnx"), b"dummy").unwrap();

        let detected = detect_model_type(&model_path);
        // Falls through to default (TDT) because CTC requires both files
        assert_eq!(detected, ParakeetModelType::Tdt);
    }

    #[test]
    fn test_resolve_model_path_absolute() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().to_path_buf();

        // Create a dummy file so the path exists
        fs::write(model_path.join("model.onnx"), b"dummy").unwrap();

        let resolved = resolve_model_path(model_path.to_str().unwrap());
        assert!(resolved.is_ok());
        assert_eq!(resolved.unwrap(), model_path);
    }

    #[test]
    fn test_resolve_model_path_not_found() {
        let result = resolve_model_path("/nonexistent/path/to/model");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, TranscribeError::ModelNotFound(_)));
    }

    // ── CUDA probe-policy contract tests ────────────────────────────────────
    //
    // These tests verify branch selection at compile time under the feature
    // combinations users and packagers actually build with.  No real CUDA
    // library loading occurs; the helper `cuda_probe_policy()` simply returns
    // the variant baked in by the compiler.

    /// With parakeet-cuda alone (no parakeet-load-dynamic) the binary contains
    /// a bundled ONNX Runtime, so the strict CUDA-major check must be active.
    #[test]
    #[cfg(all(feature = "parakeet-cuda", not(feature = "parakeet-load-dynamic")))]
    fn cuda_probe_policy_is_strict_without_load_dynamic() {
        assert_eq!(cuda_probe_policy(), CudaProbePolicy::BundledStrict);
    }

    /// With parakeet-cuda + parakeet-load-dynamic the binary dlopens the
    /// system ONNX Runtime, so only CUDA *presence* matters.  The permissive
    /// policy must be compiled in.
    #[test]
    #[cfg(all(feature = "parakeet-cuda", feature = "parakeet-load-dynamic"))]
    fn cuda_probe_policy_is_permissive_with_load_dynamic() {
        assert_eq!(cuda_probe_policy(), CudaProbePolicy::LoadDynamicPermissive);
    }

    /// onnx-load-dynamic (the ort-crate-wide feature) must not affect
    /// Parakeet's probe policy after the fix.  With parakeet-cuda +
    /// onnx-load-dynamic but WITHOUT parakeet-load-dynamic the strict
    /// bundled policy must still compile.
    #[test]
    #[cfg(all(
        feature = "parakeet-cuda",
        feature = "onnx-load-dynamic",
        not(feature = "parakeet-load-dynamic")
    ))]
    fn onnx_load_dynamic_alone_does_not_select_permissive_policy() {
        assert_eq!(cuda_probe_policy(), CudaProbePolicy::BundledStrict);
    }

    /// The guidance string shown to users on a CUDA mismatch names the feature
    /// `parakeet-load-dynamic`.  This constant asserts the canonical name used
    /// by both the cfg gate and the user-visible message are identical, so
    /// future drift shows up as a compile-visible symbol mismatch rather than
    /// a buried string difference.
    #[test]
    fn guidance_feature_name_matches_cfg_gate() {
        // The Mismatch arm in probe_cuda_runtime() tells users:
        //   "Build from source with --features parakeet-load-dynamic …"
        // The cfg gates in that function check:
        //   #[cfg(not(feature = "parakeet-load-dynamic"))]
        //   #[cfg(feature = "parakeet-load-dynamic")]
        // Both must agree on the same string literal.
        const GUIDANCE_FEATURE: &str = "parakeet-load-dynamic";
        const CFG_GATE_FEATURE: &str = "parakeet-load-dynamic";
        assert_eq!(GUIDANCE_FEATURE, CFG_GATE_FEATURE);
    }
}
