//! Interactive model selection and download

use super::{print_failure, print_info, print_success, print_warning};
use crate::config::{Config, TranscriptionEngine};
use crate::setup::verify::{
    ensure_file_integrity, ensure_file_integrity_with_prompt, fetch_hf_lfs_sha256,
    prompt_hash_mismatch_continue, CanonicalHashStatus, IntegrityContext, IntegrityOutcome,
    IntegrityPromptCache,
};
use crate::transcribe::whisper::{get_model_filename, get_model_url};
use anyhow::Context;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

/// Section-header tag rendered next to the engines whose ONNX graphs the
/// MIGraphX 7.2 EP can't compile (Moonshine/SenseVoice/Paraformer/Dolphin/
/// Omnilingual). Only shown on the AMD-targeted binary so users picking a
/// model see at a glance which engines stay on CPU even with their GPU
/// installed. NVIDIA/CPU binaries don't print this.
#[cfg(feature = "onnx-migraphx-enabled")]
const AMD_CPU_ONLY_TAG: &str = " \x1b[33m[CPU on AMD GPU]\x1b[0m";
#[cfg(not(feature = "onnx-migraphx-enabled"))]
const AMD_CPU_ONLY_TAG: &str = "";

/// Model information for display
struct ModelInfo {
    name: &'static str,
    size_mb: u32,
    description: &'static str,
    english_only: bool,
}

const MODELS: &[ModelInfo] = &[
    // Tiny models
    ModelInfo {
        name: "tiny",
        size_mb: 75,
        description: "Fastest, lowest accuracy",
        english_only: false,
    },
    ModelInfo {
        name: "tiny.en",
        size_mb: 39,
        description: "Fastest, lowest accuracy",
        english_only: true,
    },
    // Base models
    ModelInfo {
        name: "base",
        size_mb: 142,
        description: "Good balance (default)",
        english_only: false,
    },
    ModelInfo {
        name: "base.en",
        size_mb: 142,
        description: "Good balance (default)",
        english_only: true,
    },
    // Small models
    ModelInfo {
        name: "small",
        size_mb: 466,
        description: "Better accuracy",
        english_only: false,
    },
    ModelInfo {
        name: "small.en",
        size_mb: 466,
        description: "Better accuracy",
        english_only: true,
    },
    // Medium models
    ModelInfo {
        name: "medium",
        size_mb: 1500,
        description: "High accuracy",
        english_only: false,
    },
    ModelInfo {
        name: "medium.en",
        size_mb: 1500,
        description: "High accuracy",
        english_only: true,
    },
    // Large models
    ModelInfo {
        name: "large-v3",
        size_mb: 3100,
        description: "Best accuracy",
        english_only: false,
    },
    ModelInfo {
        name: "large-v3-turbo",
        size_mb: 1600,
        description: "Fast + accurate (recommended for GPU)",
        english_only: false,
    },
];

// =============================================================================
// Parakeet Model Definitions
// =============================================================================

/// Parakeet model information for display and download
struct ParakeetModelInfo {
    name: &'static str,
    size_mb: u32,
    description: &'static str,
    files: &'static [(&'static str, u64)], // (filename, expected_size_bytes)
    huggingface_repo: &'static str,
}

const PARAKEET_MODELS: &[ParakeetModelInfo] = &[
    ParakeetModelInfo {
        name: "parakeet-tdt-0.6b-v2",
        size_mb: 2400,
        description: "TDT English-only, best English accuracy",
        files: &[
            ("encoder-model.onnx", 41_770_866),
            ("encoder-model.onnx.data", 2_435_420_160),
            ("decoder_joint-model.onnx", 35_792_059),
            ("vocab.txt", 9_384),
            ("config.json", 97),
        ],
        huggingface_repo: "istupakov/parakeet-tdt-0.6b-v2-onnx",
    },
    ParakeetModelInfo {
        name: "parakeet-tdt-0.6b-v2-int8",
        size_mb: 640,
        description: "TDT English-only quantized, smaller/faster",
        files: &[
            ("encoder-model.int8.onnx", 652_184_014),
            ("decoder_joint-model.int8.onnx", 8_998_286),
            ("vocab.txt", 9_384),
            ("config.json", 97),
        ],
        huggingface_repo: "istupakov/parakeet-tdt-0.6b-v2-onnx",
    },
    ParakeetModelInfo {
        name: "parakeet-tdt-0.6b-v3",
        size_mb: 2600,
        description: "TDT model with punctuation (recommended)",
        files: &[
            ("encoder-model.onnx", 43_825_971),
            ("encoder-model.onnx.data", 2_620_260_352),
            ("decoder_joint-model.onnx", 76_023_939),
            ("vocab.txt", 96_179),
            ("config.json", 97),
        ],
        huggingface_repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
    },
    ParakeetModelInfo {
        name: "parakeet-tdt-0.6b-v3-int8",
        size_mb: 670,
        description: "TDT quantized, smaller/faster",
        files: &[
            ("encoder-model.int8.onnx", 683_671_552),
            ("decoder_joint-model.int8.onnx", 19_087_667),
            ("vocab.txt", 96_179),
            ("config.json", 97),
        ],
        huggingface_repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
    },
];

// =============================================================================
// Moonshine Model Definitions
// =============================================================================

/// Moonshine model information for display and download
struct MoonshineModelInfo {
    /// Short config name (e.g., "base", "tiny", "base-ja")
    name: &'static str,
    /// Directory name under models/ (e.g., "moonshine-base")
    dir_name: &'static str,
    size_mb: u32,
    description: &'static str,
    /// Language display string (e.g., "en", "ja")
    language: &'static str,
    /// "MIT" for English models, "Community" for non-English (non-commercial only)
    license: &'static str,
    /// (repo_path, local_filename) - repo_path is the path within the HuggingFace repo
    files: &'static [(&'static str, &'static str)],
    huggingface_repo: &'static str,
}

const MOONSHINE_MODELS: &[MoonshineModelInfo] = &[
    // English models (MIT license)
    MoonshineModelInfo {
        name: "base",
        dir_name: "moonshine-base",
        size_mb: 237,
        description: "Fast, good accuracy (recommended)",
        language: "en",
        license: "MIT",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-base-ONNX",
    },
    MoonshineModelInfo {
        name: "tiny",
        dir_name: "moonshine-tiny",
        size_mb: 100,
        description: "Fastest, lower accuracy",
        language: "en",
        license: "MIT",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-tiny-ONNX",
    },
    // Multilingual models (Moonshine Community License - non-commercial only)
    MoonshineModelInfo {
        name: "base-ja",
        dir_name: "moonshine-base-ja",
        size_mb: 237,
        description: "Japanese",
        language: "ja",
        license: "Community",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-base-ja-ONNX",
    },
    MoonshineModelInfo {
        name: "base-zh",
        dir_name: "moonshine-base-zh",
        size_mb: 237,
        description: "Mandarin Chinese",
        language: "zh",
        license: "Community",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-base-zh-ONNX",
    },
    MoonshineModelInfo {
        name: "tiny-ja",
        dir_name: "moonshine-tiny-ja",
        size_mb: 100,
        description: "Japanese (tiny)",
        language: "ja",
        license: "Community",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-tiny-ja-ONNX",
    },
    MoonshineModelInfo {
        name: "tiny-zh",
        dir_name: "moonshine-tiny-zh",
        size_mb: 100,
        description: "Mandarin Chinese (tiny)",
        language: "zh",
        license: "Community",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-tiny-zh-ONNX",
    },
    MoonshineModelInfo {
        name: "tiny-ko",
        dir_name: "moonshine-tiny-ko",
        size_mb: 100,
        description: "Korean (tiny)",
        language: "ko",
        license: "Community",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-tiny-ko-ONNX",
    },
    MoonshineModelInfo {
        name: "tiny-ar",
        dir_name: "moonshine-tiny-ar",
        size_mb: 100,
        description: "Arabic (tiny)",
        language: "ar",
        license: "Community",
        files: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        huggingface_repo: "onnx-community/moonshine-tiny-ar-ONNX",
    },
];

// =============================================================================
// SenseVoice Model Definitions
// =============================================================================

/// SenseVoice model information for display and download
struct SenseVoiceModelInfo {
    name: &'static str,
    dir_name: &'static str,
    size_mb: u32,
    description: &'static str,
    languages: &'static str,
    files: &'static [(&'static str, &'static str)], // (repo_path, local_filename)
    huggingface_repo: &'static str,
}

const SENSEVOICE_MODELS: &[SenseVoiceModelInfo] = &[
    SenseVoiceModelInfo {
        name: "small",
        dir_name: "sensevoice-small",
        size_mb: 239,
        description: "Quantized int8 (recommended)",
        languages: "zh/en/ja/ko/yue",
        files: &[
            ("model.int8.onnx", "model.int8.onnx"),
            ("tokens.txt", "tokens.txt"),
        ],
        huggingface_repo: "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    },
    SenseVoiceModelInfo {
        name: "small-fp32",
        dir_name: "sensevoice-small-fp32",
        size_mb: 938,
        description: "Full precision (larger, slightly better accuracy)",
        languages: "zh/en/ja/ko/yue",
        files: &[("model.onnx", "model.onnx"), ("tokens.txt", "tokens.txt")],
        huggingface_repo: "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    },
];

// =============================================================================
// Paraformer Model Definitions
// =============================================================================

/// Paraformer model info (same structure as SenseVoice: model.onnx + tokens.txt)
struct ParaformerModelInfo {
    name: &'static str,
    dir_name: &'static str,
    size_mb: u32,
    description: &'static str,
    languages: &'static str,
    files: &'static [(&'static str, &'static str)],
    huggingface_repo: &'static str,
}

const PARAFORMER_MODELS: &[ParaformerModelInfo] = &[
    ParaformerModelInfo {
        name: "zh",
        dir_name: "paraformer-zh",
        size_mb: 487,
        description: "Chinese + English offline (recommended)",
        languages: "zh/en",
        files: &[
            ("model.int8.onnx", "model.int8.onnx"),
            ("tokens.txt", "tokens.txt"),
        ],
        huggingface_repo: "csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14",
    },
    ParaformerModelInfo {
        name: "en",
        dir_name: "paraformer-en",
        size_mb: 220,
        description: "English offline",
        languages: "en",
        files: &[
            ("model.int8.onnx", "model.int8.onnx"),
            ("tokens.txt", "tokens.txt"),
        ],
        huggingface_repo: "csukuangfj/sherpa-onnx-paraformer-en-2024-03-09",
    },
];

// =============================================================================
// Dolphin Model Definitions
// =============================================================================

struct DolphinModelInfo {
    name: &'static str,
    dir_name: &'static str,
    size_mb: u32,
    description: &'static str,
    languages: &'static str,
    files: &'static [(&'static str, &'static str)],
    huggingface_repo: &'static str,
}

const DOLPHIN_MODELS: &[DolphinModelInfo] = &[DolphinModelInfo {
    name: "base",
    dir_name: "dolphin-base",
    size_mb: 198,
    description: "Dictation-optimized (recommended)",
    languages: "en/zh",
    files: &[
        ("model.int8.onnx", "model.int8.onnx"),
        ("tokens.txt", "tokens.txt"),
    ],
    huggingface_repo: "csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
}];

// =============================================================================
// Omnilingual Model Definitions
// =============================================================================

struct OmnilingualModelInfo {
    name: &'static str,
    dir_name: &'static str,
    size_mb: u32,
    description: &'static str,
    languages: &'static str,
    files: &'static [(&'static str, &'static str)],
    huggingface_repo: &'static str,
}

const OMNILINGUAL_MODELS: &[OmnilingualModelInfo] = &[OmnilingualModelInfo {
    name: "300m",
    dir_name: "omnilingual-300m",
    size_mb: 3900,
    description: "1600+ languages, 300M params",
    languages: "1600+ langs",
    files: &[("model.onnx", "model.onnx"), ("tokens.txt", "tokens.txt")],
    huggingface_repo: "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12",
}];

// =============================================================================
// Cohere Transcribe Model Definitions
// =============================================================================
// Encoder-decoder ASR via ONNX Runtime, Whisper-style task tokens. Currently
// #1 on the Open ASR Leaderboard. The original CohereLabs weights are gated
// on HuggingFace; we use the community ONNX export which is Apache 2.0 and
// does not require an HF token. Each model is 5 files (encoder + decoder
// .onnx structural files, their .data weight sidecars, and tokens.txt).

struct CohereModelInfo {
    name: &'static str,
    dir_name: &'static str,
    size_mb: u32,
    description: &'static str,
    languages: &'static str,
    files: &'static [(&'static str, &'static str)],
    huggingface_repo: &'static str,
}

/// Cohere Transcribe variants from the upstream HF Optimum export at
/// `onnx-community/cohere-transcribe-03-2026-ONNX`. All four share the
/// same I/O signature (HF-standard merged decoder with per-layer
/// `past_key_values.{decoder,encoder}.{key,value}`); the only differences
/// are weight precision and total file size.
///
/// Each variant ships:
/// - `encoder_model{,_<suffix>}.onnx` + `.onnx_data*` shards
/// - `decoder_model_merged{,_<suffix>}.onnx` + `.onnx_data`
/// - `tokenizer.json` (HF tokenizer format, replaces the cstr `tokens.txt`)
/// - `config.json`, `generation_config.json`, `processor_config.json`
///
/// We rename the encoder/decoder ONNX files to canonical names locally
/// (`encoder_model.onnx` / `decoder_model_merged.onnx`) so `cohere.rs`
/// doesn't need to know about the suffix; the `.onnx_data*` shards keep
/// their upstream names because the ONNX graph references them by name.
const COHERE_MODELS: &[CohereModelInfo] = &[
    CohereModelInfo {
        name: "q4f16",
        dir_name: "cohere-transcribe-q4f16",
        size_mb: 1500,
        description: "Encoder-decoder ASR, q4 weights + fp16 activations (smallest, GPU-friendly)",
        languages: "ar,de,el,en,es,fr,it,ja,ko,nl,pl,pt,vi,zh",
        files: &[
            ("encoder_model_q4f16.onnx", "encoder_model.onnx"),
            (
                "encoder_model_q4f16.onnx_data",
                "encoder_model_q4f16.onnx_data",
            ),
            (
                "decoder_model_merged_q4f16.onnx",
                "decoder_model_merged.onnx",
            ),
            (
                "decoder_model_merged_q4f16.onnx_data",
                "decoder_model_merged_q4f16.onnx_data",
            ),
            ("tokenizer.json", "tokenizer.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
            ("config.json", "config.json"),
            ("generation_config.json", "generation_config.json"),
            ("processor_config.json", "processor_config.json"),
        ],
        huggingface_repo: "onnx-community/cohere-transcribe-03-2026-ONNX",
    },
    CohereModelInfo {
        name: "q4",
        dir_name: "cohere-transcribe-q4",
        size_mb: 2000,
        description: "Encoder-decoder ASR, 4-bit weights (MIGraphX-compatible on AMD GPU)",
        languages: "ar,de,el,en,es,fr,it,ja,ko,nl,pl,pt,vi,zh",
        files: &[
            ("encoder_model_q4.onnx", "encoder_model.onnx"),
            ("encoder_model_q4.onnx_data", "encoder_model_q4.onnx_data"),
            ("decoder_model_merged_q4.onnx", "decoder_model_merged.onnx"),
            (
                "decoder_model_merged_q4.onnx_data",
                "decoder_model_merged_q4.onnx_data",
            ),
            ("tokenizer.json", "tokenizer.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
            ("config.json", "config.json"),
            ("generation_config.json", "generation_config.json"),
            ("processor_config.json", "processor_config.json"),
        ],
        huggingface_repo: "onnx-community/cohere-transcribe-03-2026-ONNX",
    },
    CohereModelInfo {
        name: "int8",
        dir_name: "cohere-transcribe-int8",
        size_mb: 2900,
        description: "Encoder-decoder ASR, 8-bit weights",
        languages: "ar,de,el,en,es,fr,it,ja,ko,nl,pl,pt,vi,zh",
        files: &[
            ("encoder_model_quantized.onnx", "encoder_model.onnx"),
            (
                "encoder_model_quantized.onnx_data",
                "encoder_model_quantized.onnx_data",
            ),
            (
                "encoder_model_quantized.onnx_data_1",
                "encoder_model_quantized.onnx_data_1",
            ),
            (
                "decoder_model_merged_quantized.onnx",
                "decoder_model_merged.onnx",
            ),
            (
                "decoder_model_merged_quantized.onnx_data",
                "decoder_model_merged_quantized.onnx_data",
            ),
            ("tokenizer.json", "tokenizer.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
            ("config.json", "config.json"),
            ("generation_config.json", "generation_config.json"),
            ("processor_config.json", "processor_config.json"),
        ],
        huggingface_repo: "onnx-community/cohere-transcribe-03-2026-ONNX",
    },
    CohereModelInfo {
        name: "fp16",
        dir_name: "cohere-transcribe-fp16",
        size_mb: 3900,
        description: "Encoder-decoder ASR, FP16 weights (highest accuracy, GPU-friendly)",
        languages: "ar,de,el,en,es,fr,it,ja,ko,nl,pl,pt,vi,zh",
        files: &[
            ("encoder_model_fp16.onnx", "encoder_model.onnx"),
            (
                "encoder_model_fp16.onnx_data",
                "encoder_model_fp16.onnx_data",
            ),
            (
                "encoder_model_fp16.onnx_data_1",
                "encoder_model_fp16.onnx_data_1",
            ),
            (
                "decoder_model_merged_fp16.onnx",
                "decoder_model_merged.onnx",
            ),
            (
                "decoder_model_merged_fp16.onnx_data",
                "decoder_model_merged_fp16.onnx_data",
            ),
            ("tokenizer.json", "tokenizer.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
            ("config.json", "config.json"),
            ("generation_config.json", "generation_config.json"),
            ("processor_config.json", "processor_config.json"),
        ],
        huggingface_repo: "onnx-community/cohere-transcribe-03-2026-ONNX",
    },
];

// =============================================================================
// Whisper Model Functions
// =============================================================================

/// Check if a model name is valid (Whisper models)
pub fn is_valid_model(name: &str) -> bool {
    MODELS.iter().any(|m| m.name == name)
}

/// Get list of valid model names (for error messages)
pub fn valid_model_names() -> Vec<&'static str> {
    MODELS.iter().map(|m| m.name).collect()
}

/// Run interactive model selection (single menu with all models)
pub async fn interactive_select() -> anyhow::Result<()> {
    println!("Voxtype Model Selection\n");
    println!("=======================\n");

    let models_dir = Config::models_dir();

    println!("Models directory: {:?}\n", models_dir);

    // Load current config to determine active model
    let config = crate::config::load_config(Config::default_path().as_deref()).unwrap_or_default();
    let is_whisper_engine = matches!(config.engine, TranscriptionEngine::Whisper);
    let is_parakeet_engine = matches!(config.engine, TranscriptionEngine::Parakeet);
    let is_moonshine_engine = matches!(config.engine, TranscriptionEngine::Moonshine);
    let is_sensevoice_engine = matches!(config.engine, TranscriptionEngine::SenseVoice);
    let is_paraformer_engine = matches!(config.engine, TranscriptionEngine::Paraformer);
    let is_dolphin_engine = matches!(config.engine, TranscriptionEngine::Dolphin);
    let is_omnilingual_engine = matches!(config.engine, TranscriptionEngine::Omnilingual);
    let is_cohere_engine = matches!(config.engine, TranscriptionEngine::Cohere);
    let current_whisper_model = &config.whisper.model;
    let current_parakeet_model = config.parakeet.as_ref().map(|p| p.model.as_str());
    let current_moonshine_model = config.moonshine.as_ref().map(|m| m.model.as_str());
    let current_sensevoice_model = config.sensevoice.as_ref().map(|s| s.model.as_str());
    let current_paraformer_model = config.paraformer.as_ref().map(|p| p.model.as_str());
    let current_dolphin_model = config.dolphin.as_ref().map(|d| d.model.as_str());
    let current_omnilingual_model = config.omnilingual.as_ref().map(|o| o.model.as_str());
    let current_cohere_model = config.cohere.as_ref().map(|c| c.model.as_str());
    let parakeet_available = cfg!(feature = "parakeet");
    let moonshine_available = cfg!(feature = "moonshine");
    let sensevoice_available = cfg!(feature = "sensevoice");
    let paraformer_available = cfg!(feature = "paraformer");
    let dolphin_available = cfg!(feature = "dolphin");
    let omnilingual_available = cfg!(feature = "omnilingual");
    let cohere_available = cfg!(feature = "cohere");
    let whisper_count = MODELS.len();
    let parakeet_count = PARAKEET_MODELS.len();
    let moonshine_count = MOONSHINE_MODELS.len();
    let sensevoice_count = SENSEVOICE_MODELS.len();
    let paraformer_count = PARAFORMER_MODELS.len();
    let dolphin_count = DOLPHIN_MODELS.len();
    let omnilingual_count = OMNILINGUAL_MODELS.len();
    let cohere_count = COHERE_MODELS.len();

    let available_count = |available: bool, count: usize| if available { count } else { 0 };
    let total_count = whisper_count
        + available_count(parakeet_available, parakeet_count)
        + available_count(moonshine_available, moonshine_count)
        + available_count(sensevoice_available, sensevoice_count)
        + available_count(paraformer_available, paraformer_count)
        + available_count(dolphin_available, dolphin_count)
        + available_count(omnilingual_available, omnilingual_count)
        + available_count(cohere_available, cohere_count);

    // --- Whisper Section ---
    println!("--- Whisper (OpenAI, 99+ languages) ---\n");

    for (i, model) in MODELS.iter().enumerate() {
        let filename = get_model_filename(model.name);
        let model_path = models_dir.join(&filename);
        let installed = model_path.exists();

        let is_current = is_whisper_engine && model.name == current_whisper_model;
        let star = if is_current { "*" } else { " " };

        let status = if installed {
            "\x1b[32m[installed]\x1b[0m"
        } else {
            ""
        };

        let lang = if model.english_only { "en" } else { "multi" };

        println!(
            " {}[{:>2}] {:<16} ({:>4} MB) {} - {} {}",
            star,
            i + 1,
            model.name,
            model.size_mb,
            lang,
            model.description,
            status
        );
    }

    // --- Parakeet Section ---
    println!("\n--- Parakeet (NVIDIA FastConformer, English) ---\n");

    if parakeet_available {
        for (i, model) in PARAKEET_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.name);
            let installed = model_path.exists() && validate_parakeet_model(&model_path).is_ok();

            let is_current = is_parakeet_engine && current_parakeet_model == Some(model.name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<28} ({:>4} MB) - {} {}",
                star,
                whisper_count + i + 1,
                model.name,
                model.size_mb,
                model.description,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features parakeet)\x1b[0m");
    }

    // --- Moonshine Section ---
    let moonshine_offset = whisper_count
        + if parakeet_available {
            parakeet_count
        } else {
            0
        };
    println!(
        "\n--- Moonshine (Moonshine AI, encoder-decoder ASR){} ---\n",
        AMD_CPU_ONLY_TAG
    );

    if moonshine_available {
        for (i, model) in MOONSHINE_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.dir_name);
            let installed = model_path.exists() && validate_moonshine_model(&model_path).is_ok();

            let is_current = is_moonshine_engine && current_moonshine_model == Some(model.name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            let license_tag = if model.license == "Community" {
                " \x1b[33m[non-commercial]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<20} ({:>4} MB) {} - {}{} {}",
                star,
                moonshine_offset + i + 1,
                model.dir_name,
                model.size_mb,
                model.language,
                model.description,
                license_tag,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features moonshine)\x1b[0m");
    }

    // --- SenseVoice Section ---
    let sensevoice_offset = moonshine_offset
        + if moonshine_available {
            moonshine_count
        } else {
            0
        };
    println!(
        "\n--- SenseVoice (Alibaba FunAudioLLM, CJK + English){} ---\n",
        AMD_CPU_ONLY_TAG
    );

    if sensevoice_available {
        for (i, model) in SENSEVOICE_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.dir_name);
            let installed = model_path.exists() && validate_sensevoice_model(&model_path).is_ok();

            let is_current = is_sensevoice_engine && current_sensevoice_model == Some(model.name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<20} ({:>4} MB) {} - {} {}",
                star,
                sensevoice_offset + i + 1,
                model.dir_name,
                model.size_mb,
                model.languages,
                model.description,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features sensevoice)\x1b[0m");
    }

    // --- Paraformer Section ---
    let paraformer_offset =
        sensevoice_offset + available_count(sensevoice_available, sensevoice_count);
    println!(
        "\n--- Paraformer (FunASR, Chinese + English){} ---\n",
        AMD_CPU_ONLY_TAG
    );

    if paraformer_available {
        for (i, model) in PARAFORMER_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.dir_name);
            let installed = model_path.exists() && validate_onnx_ctc_model(&model_path).is_ok();

            let is_current = is_paraformer_engine && current_paraformer_model == Some(model.name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<20} ({:>4} MB) {} - {} {}",
                star,
                paraformer_offset + i + 1,
                model.dir_name,
                model.size_mb,
                model.languages,
                model.description,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features paraformer)\x1b[0m");
    }

    // --- Dolphin Section ---
    let dolphin_offset =
        paraformer_offset + available_count(paraformer_available, paraformer_count);
    println!(
        "\n--- Dolphin (dictation-optimized CTC){} ---\n",
        AMD_CPU_ONLY_TAG
    );

    if dolphin_available {
        for (i, model) in DOLPHIN_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.dir_name);
            let installed = model_path.exists() && validate_onnx_ctc_model(&model_path).is_ok();

            let is_current = is_dolphin_engine && current_dolphin_model == Some(model.name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<20} ({:>4} MB) {} - {} {}",
                star,
                dolphin_offset + i + 1,
                model.dir_name,
                model.size_mb,
                model.languages,
                model.description,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features dolphin)\x1b[0m");
    }

    // --- Omnilingual Section ---
    let omnilingual_offset = dolphin_offset + available_count(dolphin_available, dolphin_count);
    println!(
        "\n--- Omnilingual (FunASR, 50+ languages){} ---\n",
        AMD_CPU_ONLY_TAG
    );

    if omnilingual_available {
        for (i, model) in OMNILINGUAL_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.dir_name);
            let installed = model_path.exists() && validate_onnx_ctc_model(&model_path).is_ok();

            let is_current = is_omnilingual_engine && current_omnilingual_model == Some(model.name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<20} ({:>4} MB) {} - {} {}",
                star,
                omnilingual_offset + i + 1,
                model.dir_name,
                model.size_mb,
                model.languages,
                model.description,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features omnilingual)\x1b[0m");
    }

    // --- Cohere Section ---
    let cohere_offset =
        omnilingual_offset + available_count(omnilingual_available, omnilingual_count);
    println!(
        "\n--- Cohere Transcribe (Cohere Labs, #1 Open ASR Leaderboard){} ---\n",
        AMD_CPU_ONLY_TAG
    );

    if cohere_available {
        for (i, model) in COHERE_MODELS.iter().enumerate() {
            let model_path = models_dir.join(model.dir_name);
            let installed = model_path.exists() && validate_cohere_model(&model_path).is_ok();

            let is_current = is_cohere_engine && current_cohere_model == Some(model.dir_name);
            let star = if is_current { "*" } else { " " };

            let status = if installed {
                "\x1b[32m[installed]\x1b[0m"
            } else {
                ""
            };

            println!(
                " {}[{:>2}] {:<28} ({:>4} MB) {} - {} {}",
                star,
                cohere_offset + i + 1,
                model.dir_name,
                model.size_mb,
                model.languages,
                model.description,
                status
            );
        }
    } else {
        println!("  \x1b[90m(not available - rebuild with --features cohere)\x1b[0m");
    }

    println!("\n  [ 0] Cancel\n");

    // Get user selection
    print!("Select model [0-{}]: ", total_count);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let selection: usize = input.trim().parse().unwrap_or(0);

    if selection == 0 {
        println!("\nCancelled.");
        return Ok(());
    }

    // Route to appropriate handler based on selection
    if selection <= whisper_count {
        handle_whisper_selection(selection).await
    } else if parakeet_available && selection <= whisper_count + parakeet_count {
        let parakeet_index = selection - whisper_count;
        handle_parakeet_selection(parakeet_index).await
    } else if moonshine_available && selection <= moonshine_offset + moonshine_count {
        let moonshine_index = selection - moonshine_offset;
        handle_moonshine_selection(moonshine_index).await
    } else if sensevoice_available && selection <= sensevoice_offset + sensevoice_count {
        let sensevoice_index = selection - sensevoice_offset;
        handle_sensevoice_selection(sensevoice_index).await
    } else if paraformer_available && selection <= paraformer_offset + paraformer_count {
        let idx = selection - paraformer_offset;
        handle_onnx_engine_selection(
            "paraformer",
            PARAFORMER_MODELS
                .iter()
                .map(|m| (m.name, m.dir_name, m.size_mb, m.files, m.huggingface_repo))
                .collect(),
            idx,
            validate_onnx_ctc_model,
        )
        .await
    } else if dolphin_available && selection <= dolphin_offset + dolphin_count {
        let idx = selection - dolphin_offset;
        handle_onnx_engine_selection(
            "dolphin",
            DOLPHIN_MODELS
                .iter()
                .map(|m| (m.name, m.dir_name, m.size_mb, m.files, m.huggingface_repo))
                .collect(),
            idx,
            validate_onnx_ctc_model,
        )
        .await
    } else if omnilingual_available && selection <= omnilingual_offset + omnilingual_count {
        let idx = selection - omnilingual_offset;
        handle_onnx_engine_selection(
            "omnilingual",
            OMNILINGUAL_MODELS
                .iter()
                .map(|m| (m.name, m.dir_name, m.size_mb, m.files, m.huggingface_repo))
                .collect(),
            idx,
            validate_onnx_ctc_model,
        )
        .await
    } else if cohere_available && selection <= cohere_offset + cohere_count {
        let idx = selection - cohere_offset;
        handle_cohere_selection(idx).await
    } else {
        println!("\nInvalid selection.");
        Ok(())
    }
}

/// Handle Whisper model selection (download/config)
async fn handle_whisper_selection(selection: usize) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();

    if selection == 0 || selection > MODELS.len() {
        println!("\nCancelled.");
        return Ok(());
    }

    let model = &MODELS[selection - 1];
    let filename = get_model_filename(model.name);
    let model_path = models_dir.join(&filename);

    // Check if already installed
    if model_path.exists() {
        println!("\nModel '{}' is already installed.\n", model.name);
        println!("  [1] Set as default model (update config)");
        println!("  [2] Re-download");
        println!("  [0] Cancel\n");

        print!("Select option [1]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "" | "1" => {
                // Set as default without re-downloading
                update_config_model(model.name)?;
                restart_daemon_if_running().await;
                return Ok(());
            }
            "2" => {
                // Continue to download below
            }
            _ => {
                println!("Cancelled.");
                return Ok(());
            }
        }
    }

    // Download the model
    download_model(model.name)?;

    // Update config and restart daemon
    update_config_model(model.name)?;
    restart_daemon_if_running().await;

    Ok(())
}

/// Handle Parakeet model selection (download/config)
async fn handle_parakeet_selection(selection: usize) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();

    if selection == 0 || selection > PARAKEET_MODELS.len() {
        println!("\nCancelled.");
        return Ok(());
    }

    let model = &PARAKEET_MODELS[selection - 1];
    let model_path = models_dir.join(model.name);

    // Check if already installed
    if model_path.exists() && validate_parakeet_model(&model_path).is_ok() {
        println!("\nModel '{}' is already installed.\n", model.name);
        println!("  [1] Set as default model (update config)");
        println!("  [2] Re-download");
        println!("  [0] Cancel\n");

        print!("Select option [1]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "" | "1" => {
                // Set as default without re-downloading
                update_config_parakeet(model.name)?;
                restart_daemon_if_running().await;
                return Ok(());
            }
            "2" => {
                // Continue to download below
            }
            _ => {
                println!("Cancelled.");
                return Ok(());
            }
        }
    }

    // Download the model
    download_parakeet_model_by_info(model)?;

    // Update config and restart daemon
    update_config_parakeet(model.name)?;
    restart_daemon_if_running().await;

    Ok(())
}

/// Handle Moonshine model selection (download/config)
async fn handle_moonshine_selection(selection: usize) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();

    if selection == 0 || selection > MOONSHINE_MODELS.len() {
        println!("\nCancelled.");
        return Ok(());
    }

    let model = &MOONSHINE_MODELS[selection - 1];
    let model_path = models_dir.join(model.dir_name);

    // Check if already installed
    if model_path.exists() && validate_moonshine_model(&model_path).is_ok() {
        println!("\nModel '{}' is already installed.\n", model.dir_name);
        println!("  [1] Set as default model (update config)");
        println!("  [2] Re-download");
        println!("  [0] Cancel\n");

        print!("Select option [1]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "" | "1" => {
                // Set as default without re-downloading
                update_config_moonshine(model.name)?;
                restart_daemon_if_running().await;
                return Ok(());
            }
            "2" => {
                // Continue to download below
            }
            _ => {
                println!("Cancelled.");
                return Ok(());
            }
        }
    }

    // Show license warning for non-commercial models
    if model.license == "Community" {
        println!();
        print_warning("This model uses the Moonshine Community License (non-commercial use only).");
        print_info("Commercial use requires a separate license from Moonshine AI.");
        println!();
        print!("Continue? [Y/n]: ");
        io::stdout().flush()?;

        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm)?;
        let confirm = confirm.trim().to_lowercase();
        if confirm == "n" || confirm == "no" {
            println!("Cancelled.");
            return Ok(());
        }
    }

    // Download the model
    download_moonshine_model_by_info(model)?;

    // Update config and restart daemon
    update_config_moonshine(model.name)?;
    restart_daemon_if_running().await;

    Ok(())
}

/// Restart the voxtype daemon if it's running
async fn restart_daemon_if_running() {
    // Check if daemon is running via systemd
    let status = tokio::process::Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", "voxtype"])
        .status()
        .await;

    if status.map(|s| s.success()).unwrap_or(false) {
        // Daemon is running, restart it
        println!("\nRestarting voxtype daemon...");
        let restart = tokio::process::Command::new("systemctl")
            .args(["--user", "restart", "voxtype"])
            .status()
            .await;

        match restart {
            Ok(s) if s.success() => {
                print_success("Daemon restarted with new model");
            }
            _ => {
                print_warning("Could not restart daemon");
                print_info("Restart manually: systemctl --user restart voxtype");
            }
        }
    } else {
        println!("\n---");
        println!("Model setup complete!");
    }
}

// =============================================================================
// Whisper Download Functions
// =============================================================================

struct RuntimeHelperAssetSpec {
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
    download_message: &'static str,
    success_message: &'static str,
    download_failure_message: &'static str,
    curl_missing_message: &'static str,
}

fn curl_download(path: &Path, url: &str, show_progress: bool) -> anyhow::Result<()> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("File path contains non-UTF-8 bytes: {:?}", path))?;

    let mut args = vec!["-L"];
    if show_progress {
        args.push("--progress-bar");
    } else {
        args.push("-sS");
    }
    args.extend(["-o", path_str, url]);

    match Command::new("curl").args(&args).status() {
        Ok(exit_status) if exit_status.success() => Ok(()),
        Ok(exit_status) => {
            let _ = std::fs::remove_file(path);
            anyhow::bail!(
                "Download failed: curl exited with code {}",
                exit_status.code().unwrap_or(-1)
            )
        }
        Err(e) => {
            let _ = std::fs::remove_file(path);
            anyhow::bail!(
                "curl not available: {}. Please ensure curl is installed (e.g., 'sudo pacman -S curl')",
                e
            )
        }
    }
}

/// Fetch the canonical hash status for a model file from HuggingFace metadata.
///
/// Returns `Ok(CanonicalHashStatus::KnownHash(hex))` when the file is found
/// and has an LFS SHA-256 that callers should verify against.
///
/// Returns `Ok(CanonicalHashStatus::NoCanonicalHash)` when the file is found
/// in metadata but is stored inline with no LFS hash (e.g. small config files).
///
/// Returns `Err` on network failure, parse failure, or when the file is not
/// listed in the repository metadata at all. **Callers must propagate this
/// error** — silently treating it as `None` would downgrade a strict explicit
/// setup flow into an unverified success path.
fn fetch_hf_expected_hash(
    repo: &str,
    remote_filename: &str,
    display_name: &str,
) -> anyhow::Result<CanonicalHashStatus> {
    fetch_hf_lfs_sha256(repo, remote_filename).with_context(|| {
        format!(
            "Failed to look up canonical hash for '{}' in HuggingFace repo '{}'",
            display_name, repo
        )
    })
}

/// Convert a `CanonicalHashStatus` to the `Option<&str>` form expected by
/// the integrity engine.
///
/// - `KnownHash` → `Some(hex)` — the engine will verify the file against
///   this hash and repair/prompt on mismatch.
/// - `NoCanonicalHash` → `None` — the engine accepts the file as-is; this
///   is intentional for inline HF files that genuinely have no LFS hash.
///
/// This must only be called after `fetch_hf_expected_hash` has already
/// succeeded. The absence of a hash here means "metadata fetch succeeded
/// and the file truly has no canonical hash", not "lookup failed".
fn resolve_hash_status(status: &CanonicalHashStatus) -> Option<&str> {
    match status {
        CanonicalHashStatus::KnownHash(h) => Some(h.as_str()),
        CanonicalHashStatus::NoCanonicalHash => None,
    }
}

/// Fetch the canonical hash for a file and run integrity-gated download.
///
/// This is the per-file building block used by all model downloaders.
/// The injected `hash_fetcher(repo, repo_path)` returns a `CanonicalHashStatus`
/// or an error; in production code it wraps `fetch_hf_lfs_sha256`, and in
/// tests a stub can be injected to avoid network calls.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn ensure_model_file_with_fetcher<H, D>(
    file_path: &std::path::Path,
    repo: &str,
    repo_path: &str,
    local_filename: &str,
    prompt_cache: &mut IntegrityPromptCache,
    hash_fetcher: H,
    download_file: D,
) -> anyhow::Result<IntegrityOutcome>
where
    H: FnOnce(&str, &str) -> anyhow::Result<CanonicalHashStatus>,
    D: FnMut() -> anyhow::Result<()>,
{
    use anyhow::Context;
    let hash_status = hash_fetcher(repo, repo_path).with_context(|| {
        format!(
            "Failed to look up canonical hash for '{}' from HuggingFace",
            local_filename
        )
    })?;
    let expected_hash = resolve_hash_status(&hash_status);
    ensure_setup_file(file_path, expected_hash, prompt_cache, download_file)
}

fn ensure_setup_file<F>(
    file_path: &Path,
    expected_hash: Option<&str>,
    prompt_cache: &mut IntegrityPromptCache,
    download_file: F,
) -> anyhow::Result<IntegrityOutcome>
where
    F: FnMut() -> anyhow::Result<()>,
{
    ensure_setup_file_with_prompt(
        file_path,
        expected_hash,
        prompt_cache,
        download_file,
        prompt_hash_mismatch_continue,
    )
}

fn ensure_setup_file_with_prompt<F, P>(
    file_path: &Path,
    expected_hash: Option<&str>,
    prompt_cache: &mut IntegrityPromptCache,
    download_file: F,
    prompt: P,
) -> anyhow::Result<IntegrityOutcome>
where
    F: FnMut() -> anyhow::Result<()>,
    P: FnMut(&Path, &anyhow::Error) -> bool,
{
    ensure_file_integrity_with_prompt(
        file_path,
        expected_hash,
        IntegrityContext::ExplicitSetup,
        prompt_cache,
        download_file,
        prompt,
    )
}

/// Classification of a file's setup integrity outcome, used by
/// [`report_setup_outcome`] and by unit tests to assert reporting behaviour
/// without capturing stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SetupOutcomeKind {
    /// File already exists and was verified against its canonical hash.
    AlreadyVerified,
    /// File already exists; no canonical hash was available to verify it.
    AlreadyCachedNoHash,
    /// File had a hash mismatch, was re-downloaded, and is now verified.
    Repaired,
    /// Integrity verification persistently failed; the user chose to keep the
    /// file. The file is retained but must **not** be treated as verified.
    AcceptedByUserUnverified,
    /// File was freshly downloaded and verified (normal success path).
    Downloaded,
    /// Any other outcome that requires no per-file message.
    Other,
}

/// Map an [`IntegrityOutcome`] to a [`SetupOutcomeKind`] for reporting.
///
/// This is a pure function with no side effects, making it directly testable
/// without stdout capture.
pub(crate) fn classify_outcome(outcome: IntegrityOutcome) -> SetupOutcomeKind {
    match outcome {
        IntegrityOutcome::ReusedVerified => SetupOutcomeKind::AlreadyVerified,
        IntegrityOutcome::ReusedWithoutHash => SetupOutcomeKind::AlreadyCachedNoHash,
        IntegrityOutcome::Repaired => SetupOutcomeKind::Repaired,
        IntegrityOutcome::AcceptedByUser => SetupOutcomeKind::AcceptedByUserUnverified,
        IntegrityOutcome::DownloadedVerified => SetupOutcomeKind::Downloaded,
        _ => SetupOutcomeKind::Other,
    }
}

fn report_setup_outcome(label: &str, outcome: IntegrityOutcome) {
    match classify_outcome(outcome) {
        SetupOutcomeKind::AlreadyVerified => {
            println!("  {} already exists, verified", label);
        }
        SetupOutcomeKind::AlreadyCachedNoHash => {
            println!(
                "  {} already exists, reusing cached copy (no canonical hash metadata available)",
                label
            );
        }
        SetupOutcomeKind::Repaired => {
            println!(
                "  {} failed integrity verification and was re-downloaded",
                label
            );
        }
        SetupOutcomeKind::AcceptedByUserUnverified => {
            print_warning(&format!(
                "{}: integrity verification failed; file kept only because you chose to \
                 continue - treat as unverified",
                label
            ));
        }
        SetupOutcomeKind::Downloaded | SetupOutcomeKind::Other => {}
    }
}

fn ensure_runtime_helper_asset_with<F>(
    models_dir: &Path,
    spec: &RuntimeHelperAssetSpec,
    mut download_file: F,
) -> Option<std::path::PathBuf>
where
    F: FnMut(&Path, &str, bool) -> anyhow::Result<()>,
{
    let model_path = models_dir.join(spec.filename);
    let had_cached_file = model_path.exists();

    if let Err(e) = std::fs::create_dir_all(models_dir) {
        eprintln!("Warning: Could not create models directory: {}", e);
        return None;
    }

    if !had_cached_file {
        println!("{}", spec.download_message);
    }

    let mut prompt_cache = IntegrityPromptCache::default();
    let mut first_download = true;

    match ensure_file_integrity(
        &model_path,
        Some(spec.sha256),
        IntegrityContext::RuntimeHelperAsset,
        &mut prompt_cache,
        || {
            let show_progress = !had_cached_file && first_download;
            first_download = false;
            download_file(&model_path, spec.url, show_progress)
        },
    ) {
        Ok(IntegrityOutcome::Fallback) => None,
        Ok(_) => {
            if !had_cached_file {
                println!("{}", spec.success_message);
            }
            Some(model_path)
        }
        Err(e) => {
            if !had_cached_file {
                if e.to_string().contains("curl not available") {
                    eprintln!("{}", spec.curl_missing_message);
                } else {
                    eprintln!("{}", spec.download_failure_message);
                }
            }
            let _ = std::fs::remove_file(&model_path);
            None
        }
    }
}

/// Download a specific Whisper model using curl
pub fn download_model(model_name: &str) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let filename = get_model_filename(model_name);
    let model_path = models_dir.join(&filename);

    // Ensure directory exists
    std::fs::create_dir_all(&models_dir)?;

    let url = get_model_url(model_name);
    let hash_status = fetch_hf_expected_hash("ggerganov/whisper.cpp", &filename, &filename)?;
    let expected_hash = resolve_hash_status(&hash_status);
    let mut prompt_cache = IntegrityPromptCache::default();
    let mut announced_download = false;

    let outcome = ensure_setup_file(&model_path, expected_hash, &mut prompt_cache, || {
        if !announced_download {
            println!("\nDownloading {}...", model_name);
            println!("URL: {}", url);
            announced_download = true;
        }
        curl_download(&model_path, &url, true)
    })?;

    match outcome {
        IntegrityOutcome::ReusedVerified => {
            print_success(&format!(
                "Model already installed and verified: {:?}",
                model_path
            ));
        }
        IntegrityOutcome::ReusedWithoutHash => {
            print_success(&format!("Model already installed: {:?}", model_path));
            print_warning("No canonical SHA-256 metadata was available for this cached file.");
        }
        IntegrityOutcome::AcceptedByUser => {
            print_warning(&format!(
                "Saved to {:?} — WARNING: integrity verification failed. \
                 File kept only because you chose to continue. Treat as unverified.",
                model_path
            ));
        }
        _ => {
            print_success(&format!("Saved to {:?}", model_path));
        }
    }

    Ok(())
}

/// GTCRN speech enhancement model URL and filename
const GTCRN_MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx";
const GTCRN_MODEL_FILENAME: &str = "gtcrn_simple.onnx";
/// SHA-256 of gtcrn_simple.onnx — pinned at the version shipped with voxtype.
/// Computed from the canonical release at k2-fsa/sherpa-onnx.
const GTCRN_MODEL_SHA256: &str = "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534";
const GTCRN_HELPER_ASSET: RuntimeHelperAssetSpec = RuntimeHelperAssetSpec {
    filename: GTCRN_MODEL_FILENAME,
    url: GTCRN_MODEL_URL,
    sha256: GTCRN_MODEL_SHA256,
    download_message: "Downloading GTCRN speech enhancement model (523 KB)...",
    success_message: "Speech enhancement model downloaded.",
    download_failure_message:
        "Warning: Failed to download speech enhancement model. Meetings will work without echo cancellation.",
    curl_missing_message: "Warning: curl not available. Speech enhancement model not downloaded.",
};

/// ECAPA-TDNN speaker embedding model URL and filename
const ECAPA_MODEL_URL: &str =
    "https://huggingface.co/pranjal-pravesh/ecapa_tdnn_onnx/resolve/main/ecapa_tdnn.onnx";
const ECAPA_MODEL_FILENAME: &str = "ecapa_tdnn.onnx";
/// SHA-256 of ecapa_tdnn.onnx from pranjal-pravesh/ecapa_tdnn_onnx.
const ECAPA_MODEL_SHA256: &str = "245eb5995cfffd74494862dee33da2b00c1c2579eb0c6703847784e9901ed458";
const ECAPA_HELPER_ASSET: RuntimeHelperAssetSpec = RuntimeHelperAssetSpec {
    filename: ECAPA_MODEL_FILENAME,
    url: ECAPA_MODEL_URL,
    sha256: ECAPA_MODEL_SHA256,
    download_message: "Downloading ECAPA-TDNN speaker embedding model (~26 MB)...",
    success_message: "Speaker embedding model downloaded.",
    download_failure_message:
        "Warning: Failed to download speaker embedding model. ML diarization will fall back to simple speaker attribution.",
    curl_missing_message: "Warning: curl not available. Speaker embedding model not downloaded.",
};

/// Ensure the GTCRN speech enhancement model is downloaded.
/// Returns the path to the model file if available, or None if download fails.
pub fn ensure_gtcrn_model() -> Option<std::path::PathBuf> {
    ensure_runtime_helper_asset_with(
        &Config::models_dir(),
        &GTCRN_HELPER_ASSET,
        curl_download,
    )
}

/// Ensure the ECAPA-TDNN speaker embedding model is downloaded.
/// Returns the path to the model file if available, or None if download fails.
/// Used by ML-based speaker diarization in meeting mode.
pub fn ensure_ecapa_model() -> Option<std::path::PathBuf> {
    ensure_runtime_helper_asset_with(
        &Config::models_dir(),
        &ECAPA_HELPER_ASSET,
        curl_download,
    )
}

/// Set a specific model as the default (must already be downloaded)
pub async fn set_model(model_name: &str, restart: bool) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let filename = get_model_filename(model_name);
    let model_path = models_dir.join(&filename);

    // Verify the model exists
    if !model_path.exists() {
        print_failure(&format!("Model '{}' is not installed", model_name));
        println!("\n  Run 'voxtype setup model' to download it first.");
        println!("  Or 'voxtype setup model --list' to see installed models.");
        anyhow::bail!("Model not installed: {}", model_name);
    }

    // Update the config
    update_config_model(model_name)?;

    if restart {
        println!("  Restarting daemon...");
        let status = tokio::process::Command::new("systemctl")
            .args(["--user", "restart", "voxtype"])
            .status()
            .await;

        match status {
            Ok(s) if s.success() => {
                print_success("Daemon restarted with new model");
            }
            _ => {
                print_warning("Could not restart daemon (not running as systemd service?)");
                print_info("Restart manually: systemctl --user restart voxtype");
            }
        }
    } else {
        print_info("Restart daemon to use new model: systemctl --user restart voxtype");
        println!(
            "       Or use: voxtype setup model --set {} --restart",
            model_name
        );
    }

    Ok(())
}

/// List installed models
pub fn list_installed() {
    println!("Installed Whisper Models\n");
    println!("========================\n");

    let models_dir = Config::models_dir();

    if !models_dir.exists() {
        println!("No models directory found: {:?}", models_dir);
        return;
    }

    let mut found = false;

    for model in MODELS {
        let filename = get_model_filename(model.name);
        let model_path = models_dir.join(&filename);

        if model_path.exists() {
            let size = std::fs::metadata(&model_path)
                .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);

            println!("  {} ({:.0} MB) - {}", model.name, size, model.description);
            found = true;
        }
    }

    if !found {
        println!("  No models installed.");
        println!("\n  Run 'voxtype setup model' to download a model.");
    }
}

/// Update the config file to use a specific model (with status messages)
fn update_config_model(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_model_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
            print_success(&format!("Config updated to use '{}' model", model_name));
            Ok(())
        } else {
            print_info("No config file found. Run 'voxtype setup' first.");
            Ok(())
        }
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update the config file to use a specific model (quiet, no output)
pub fn set_model_config(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_model_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
        }
        // Silently succeed if config doesn't exist yet - setup will create it
        Ok(())
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update the model setting in a config string (also sets engine to whisper)
fn update_model_in_config(config: &str, model_name: &str) -> String {
    // Simple regex-free replacement for the model line
    let mut result = String::new();
    let mut in_whisper_section = false;
    let mut engine_updated = false;

    for line in config.lines() {
        let trimmed = line.trim();

        // Track if we're in a section
        if trimmed.starts_with('[') {
            in_whisper_section = trimmed == "[whisper]";
        }

        // Update engine line to whisper (at top level, before any section)
        if trimmed.starts_with("engine") && !trimmed.starts_with('[') {
            result.push_str("engine = \"whisper\"\n");
            engine_updated = true;
        }
        // Replace model line in whisper section
        else if in_whisper_section && trimmed.starts_with("model") {
            result.push_str(&format!("model = \"{}\"\n", model_name));
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // If no engine line existed, we don't need to add one (whisper is the default)
    // But if engine was set to something else, we've already updated it above
    let _ = engine_updated; // suppress unused warning

    // Remove trailing newline if original didn't have one
    if !config.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

// =============================================================================
// Parakeet Model Functions
// =============================================================================

/// Check if a model name is a Parakeet model
pub fn is_parakeet_model(name: &str) -> bool {
    PARAKEET_MODELS.iter().any(|m| m.name == name)
}

/// Get list of valid Parakeet model names
pub fn valid_parakeet_model_names() -> Vec<&'static str> {
    PARAKEET_MODELS.iter().map(|m| m.name).collect()
}

/// Validate that a Parakeet model directory has the required files
pub fn validate_parakeet_model(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("Model directory does not exist: {:?}", path);
    }

    // Check for TDT structure: encoder + decoder + vocab
    let has_encoder = path.join("encoder-model.onnx").exists()
        || path.join("encoder-model.onnx.data").exists()
        || path.join("encoder-model.int8.onnx").exists();
    let has_decoder = path.join("decoder_joint-model.onnx").exists()
        || path.join("decoder_joint-model.int8.onnx").exists();
    let has_vocab = path.join("vocab.txt").exists();

    if has_encoder && has_decoder && has_vocab {
        Ok(())
    } else {
        let mut missing = Vec::new();
        if !has_encoder {
            missing.push("encoder model");
        }
        if !has_decoder {
            missing.push("decoder model");
        }
        if !has_vocab {
            missing.push("vocab.txt");
        }
        anyhow::bail!("Incomplete Parakeet model, missing: {}", missing.join(", "))
    }
}

/// Download a Parakeet model by name (public API for run_setup)
pub fn download_parakeet_model(model_name: &str) -> anyhow::Result<()> {
    let model = PARAKEET_MODELS
        .iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown Parakeet model: {}", model_name))?;

    download_parakeet_model_by_info(model)
}

/// Download a Parakeet model using its info struct
fn download_parakeet_model_by_info(model: &ParakeetModelInfo) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let model_path = models_dir.join(model.name);
    let mut prompt_cache = IntegrityPromptCache::default();

    // Create model directory
    std::fs::create_dir_all(&model_path)?;

    println!("\nDownloading {} ({} MB)...\n", model.name, model.size_mb);

    for (filename, _expected_size) in model.files {
        let file_path = model_path.join(filename);

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.huggingface_repo, filename
        );
        let hash_status = fetch_hf_expected_hash(model.huggingface_repo, filename, filename)?;
        let expected_hash = resolve_hash_status(&hash_status);
        let mut announced_download = false;

        let outcome = ensure_setup_file(&file_path, expected_hash, &mut prompt_cache, || {
            if !announced_download {
                println!("Downloading {}...", filename);
                announced_download = true;
            }
            curl_download(&file_path, &url, true)
        })?;

        report_setup_outcome(filename, outcome);
    }

    // Validate all files are present
    validate_parakeet_model(&model_path)?;
    print_success(&format!(
        "Model '{}' downloaded to {:?}",
        model.name, model_path
    ));

    Ok(())
}

/// Update config to use Parakeet engine and a specific model (with status messages)
fn update_config_parakeet(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_parakeet_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
            print_success(&format!(
                "Config updated: engine = \"parakeet\", model = \"{}\"",
                model_name
            ));
            Ok(())
        } else {
            print_info("No config file found. Run 'voxtype setup' first.");
            Ok(())
        }
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update config to use Parakeet engine and a specific model (quiet, no output)
pub fn set_parakeet_config(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_parakeet_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
        }
        Ok(())
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update the config to use Parakeet engine with a specific model
fn update_parakeet_in_config(config: &str, model_name: &str) -> String {
    let mut result = String::new();
    let mut has_engine_line = false;
    let mut has_parakeet_section = false;
    let mut in_parakeet_section = false;
    let mut parakeet_model_updated = false;

    for line in config.lines() {
        let trimmed = line.trim();

        // Track sections
        if trimmed.starts_with('[') {
            // If we were in parakeet section and didn't update model, add it
            if in_parakeet_section && !parakeet_model_updated {
                result.push_str(&format!("model = \"{}\"\n", model_name));
                parakeet_model_updated = true;
            }
            in_parakeet_section = trimmed == "[parakeet]";
            if in_parakeet_section {
                has_parakeet_section = true;
            }
        }

        // Update or add engine line at the top level
        if trimmed.starts_with("engine") && !trimmed.starts_with('[') {
            result.push_str("engine = \"parakeet\"\n");
            has_engine_line = true;
        }
        // Update model line in parakeet section
        else if in_parakeet_section && trimmed.starts_with("model") {
            result.push_str(&format!("model = \"{}\"\n", model_name));
            parakeet_model_updated = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // If we were in parakeet section at EOF and didn't update model, add it
    if in_parakeet_section && !parakeet_model_updated {
        result.push_str(&format!("model = \"{}\"\n", model_name));
    }

    // Add engine line if not present (at the very beginning after any comments)
    if !has_engine_line {
        // Find first non-comment, non-empty line or section
        let mut new_result = String::new();
        let mut engine_added = false;
        for line in result.lines() {
            let trimmed = line.trim();
            if !engine_added
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("engine")
            {
                new_result.push_str("engine = \"parakeet\"\n\n");
                engine_added = true;
            }
            new_result.push_str(line);
            new_result.push('\n');
        }
        result = new_result;
    }

    // Add [parakeet] section if not present
    if !has_parakeet_section {
        result.push_str(&format!("\n[parakeet]\nmodel = \"{}\"\n", model_name));
    }

    // Remove trailing newline if original didn't have one
    if !config.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// List installed Parakeet models
pub fn list_installed_parakeet() {
    println!("\nInstalled Parakeet Models\n");
    println!("=========================\n");

    let models_dir = Config::models_dir();

    if !models_dir.exists() {
        println!("No models directory found: {:?}", models_dir);
        return;
    }

    let mut found = false;

    for model in PARAKEET_MODELS {
        let model_path = models_dir.join(model.name);

        if model_path.exists() && validate_parakeet_model(&model_path).is_ok() {
            let size = std::fs::read_dir(&model_path)
                .map(|entries| {
                    entries
                        .flatten()
                        .filter_map(|e| e.metadata().ok())
                        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                        .sum::<f64>()
                })
                .unwrap_or(0.0);

            println!("  {} ({:.0} MB) - {}", model.name, size, model.description);
            found = true;
        }
    }

    if !found {
        println!("  No Parakeet models installed.");
        println!("\n  Run 'voxtype setup model' and select Parakeet to download.");
    }
}

// =============================================================================
// Moonshine Model Functions
// =============================================================================

/// Check if a model name is a Moonshine model
pub fn is_moonshine_model(name: &str) -> bool {
    MOONSHINE_MODELS.iter().any(|m| m.name == name)
}

/// Get list of valid Moonshine model names
pub fn valid_moonshine_model_names() -> Vec<&'static str> {
    MOONSHINE_MODELS.iter().map(|m| m.name).collect()
}

/// Validate that a Moonshine model directory has the required files
pub fn validate_moonshine_model(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("Model directory does not exist: {:?}", path);
    }

    let has_encoder = path.join("encoder_model.onnx").exists();
    let has_decoder = path.join("decoder_model_merged.onnx").exists();
    let has_tokenizer = path.join("tokenizer.json").exists();

    if has_encoder && has_decoder && has_tokenizer {
        Ok(())
    } else {
        let mut missing = Vec::new();
        if !has_encoder {
            missing.push("encoder_model.onnx");
        }
        if !has_decoder {
            missing.push("decoder_model_merged.onnx");
        }
        if !has_tokenizer {
            missing.push("tokenizer.json");
        }
        anyhow::bail!(
            "Incomplete Moonshine model, missing: {}",
            missing.join(", ")
        )
    }
}

/// Download a Moonshine model by name (public API for run_setup)
pub fn download_moonshine_model(model_name: &str) -> anyhow::Result<()> {
    let model = MOONSHINE_MODELS
        .iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown Moonshine model: {}", model_name))?;

    download_moonshine_model_by_info(model)
}

/// Download a Moonshine model using its info struct
fn download_moonshine_model_by_info(model: &MoonshineModelInfo) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let model_path = models_dir.join(model.dir_name);
    let mut prompt_cache = IntegrityPromptCache::default();

    // Create model directory
    std::fs::create_dir_all(&model_path)?;

    println!(
        "\nDownloading {} ({} MB)...\n",
        model.dir_name, model.size_mb
    );

    for (repo_path, local_filename) in model.files {
        let file_path = model_path.join(local_filename);

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.huggingface_repo, repo_path
        );
        let hash_status =
            fetch_hf_expected_hash(model.huggingface_repo, repo_path, local_filename)?;
        let expected_hash = resolve_hash_status(&hash_status);
        let mut announced_download = false;

        let outcome = ensure_setup_file(&file_path, expected_hash, &mut prompt_cache, || {
            if !announced_download {
                println!("Downloading {}...", local_filename);
                announced_download = true;
            }
            curl_download(&file_path, &url, true)
        })?;

        report_setup_outcome(local_filename, outcome);
    }

    // Validate all files are present
    validate_moonshine_model(&model_path)?;
    print_success(&format!(
        "Model '{}' downloaded to {:?}",
        model.dir_name, model_path
    ));

    Ok(())
}

// =============================================================================
// Cohere Transcribe Functions
// =============================================================================

/// Validate that a Cohere model directory has the required files.
///
/// The downloader renames variant-specific ONNX files to canonical names
/// (`encoder_model.onnx`, `decoder_model_merged.onnx`) and keeps the
/// `.onnx_data*` shards under their upstream variant-specific names because
/// the ONNX graph references them by name. We look up the variant from the
/// directory name to know which shard files to expect; if the directory
/// doesn't match a known variant, we fall back to checking the canonical
/// files shared across all variants.
pub fn validate_cohere_model(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("Model directory does not exist: {:?}", path);
    }
    let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let variant = COHERE_MODELS.iter().find(|m| m.dir_name == dir_name);

    let required: Vec<&str> = match variant {
        Some(model) => model.files.iter().map(|(_, local)| *local).collect(),
        None => vec![
            "encoder_model.onnx",
            "decoder_model_merged.onnx",
            "tokenizer.json",
            "config.json",
            "generation_config.json",
        ],
    };
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|f| !path.join(f).exists())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("Incomplete Cohere model, missing: {}", missing.join(", "))
    }
}

/// Download a Cohere model by name (public API for run_setup).
pub fn download_cohere_model(model_name: &str) -> anyhow::Result<()> {
    let model = COHERE_MODELS
        .iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown Cohere model: {}", model_name))?;
    download_cohere_model_by_info(model)
}

/// Download a Cohere model using its info struct.
fn download_cohere_model_by_info(model: &CohereModelInfo) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let model_path = models_dir.join(model.dir_name);
    let mut prompt_cache = IntegrityPromptCache::default();
    std::fs::create_dir_all(&model_path)?;

    // Cohere is a multi-GB download. Even with a fast connection it's a
    // visible commitment, and on slow links it can mean 30+ minutes. Print
    // the size up front so users don't wonder why their disk is filling.
    println!(
        "\nDownloading {} ({} MB across {} files)...",
        model.dir_name,
        model.size_mb,
        model.files.len()
    );
    println!(
        "This is the largest model voxtype ships. Ensure you have at least \
         {} MB of free space in {}.\n",
        // Add 10% headroom for filesystem overhead.
        model.size_mb + (model.size_mb / 10),
        model_path.display(),
    );

    for (repo_path, local_filename) in model.files {
        let file_path = model_path.join(local_filename);

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.huggingface_repo, repo_path
        );
        let hash_status =
            fetch_hf_expected_hash(model.huggingface_repo, repo_path, local_filename)?;
        let expected_hash = resolve_hash_status(&hash_status);
        let mut announced_download = false;

        let outcome = ensure_setup_file(&file_path, expected_hash, &mut prompt_cache, || {
            if !announced_download {
                println!("Downloading {}...", local_filename);
                announced_download = true;
            }
            curl_download(&file_path, &url, true)
        })?;

        report_setup_outcome(local_filename, outcome);
    }

    validate_cohere_model(&model_path)?;
    print_success(&format!(
        "Model '{}' downloaded to {:?}",
        model.dir_name, model_path
    ));

    Ok(())
}

/// Handle Cohere model selection (download + config update).
async fn handle_cohere_selection(selection: usize) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();

    if selection == 0 || selection > COHERE_MODELS.len() {
        println!("\nCancelled.");
        return Ok(());
    }

    let model = &COHERE_MODELS[selection - 1];
    let model_path = models_dir.join(model.dir_name);

    if model_path.exists() && validate_cohere_model(&model_path).is_ok() {
        println!("\nModel '{}' is already installed.\n", model.dir_name);
        println!("  [1] Set as default model (update config)");
        println!("  [2] Re-download");
        println!("  [0] Cancel\n");

        print!("Select option [1]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "" | "1" => {
                update_config_cohere(model.dir_name)?;
                restart_daemon_if_running().await;
                return Ok(());
            }
            "2" => {}
            _ => {
                println!("Cancelled.");
                return Ok(());
            }
        }
    }

    // Size-confirm before kicking off a multi-GB download.
    println!();
    print_warning(&format!(
        "Cohere is a {} MB download — the largest model voxtype offers.",
        model.size_mb,
    ));
    print_info("It runs entirely on-device with no cloud calls. Apache 2.0 licensed.");
    println!();
    print!("Continue? [Y/n]: ");
    io::stdout().flush()?;

    let mut confirm = String::new();
    io::stdin().read_line(&mut confirm)?;
    let confirm = confirm.trim().to_lowercase();
    if confirm == "n" || confirm == "no" {
        println!("Cancelled.");
        return Ok(());
    }

    download_cohere_model_by_info(model)?;
    update_config_cohere(model.dir_name)?;
    restart_daemon_if_running().await;
    Ok(())
}

/// Update config to use Cohere engine with a specific model (status messages).
fn update_config_cohere(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_cohere_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
            print_success(&format!(
                "Config updated: engine = \"cohere\", model = \"{}\"",
                model_name
            ));
            Ok(())
        } else {
            print_info("No config file found. Run 'voxtype setup' first.");
            Ok(())
        }
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update the config to use Cohere engine with a specific model. Mirrors
/// `update_moonshine_in_config` exactly — the only difference is the engine
/// name and section name. If the section doesn't exist, append a stub at EOF.
fn update_cohere_in_config(config: &str, model_name: &str) -> String {
    let mut result = String::new();
    let mut has_engine_line = false;
    let mut has_cohere_section = false;
    let mut in_cohere_section = false;
    let mut cohere_model_updated = false;

    for line in config.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('[') {
            if in_cohere_section && !cohere_model_updated {
                result.push_str(&format!("model = \"{}\"\n", model_name));
                cohere_model_updated = true;
            }
            in_cohere_section = trimmed == "[cohere]";
            if in_cohere_section {
                has_cohere_section = true;
            }
        }

        if trimmed.starts_with("engine") && !trimmed.starts_with('[') {
            result.push_str("engine = \"cohere\"\n");
            has_engine_line = true;
        } else if in_cohere_section && trimmed.starts_with("model") {
            result.push_str(&format!("model = \"{}\"\n", model_name));
            cohere_model_updated = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    if in_cohere_section && !cohere_model_updated {
        result.push_str(&format!("model = \"{}\"\n", model_name));
    }

    if !has_engine_line {
        let mut new_result = String::new();
        let mut engine_added = false;
        for line in result.lines() {
            let trimmed = line.trim();
            if !engine_added
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("engine")
            {
                new_result.push_str("engine = \"cohere\"\n\n");
                engine_added = true;
            }
            new_result.push_str(line);
            new_result.push('\n');
        }
        if !engine_added {
            new_result.push_str("engine = \"cohere\"\n");
        }
        result = new_result;
    }

    if !has_cohere_section {
        if !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!("\n[cohere]\nmodel = \"{}\"\n", model_name));
    }

    result
}

/// Update config to use Moonshine engine and a specific model (with status messages)
fn update_config_moonshine(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_moonshine_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
            print_success(&format!(
                "Config updated: engine = \"moonshine\", model = \"{}\"",
                model_name
            ));
            Ok(())
        } else {
            print_info("No config file found. Run 'voxtype setup' first.");
            Ok(())
        }
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update the config to use Moonshine engine with a specific model
fn update_moonshine_in_config(config: &str, model_name: &str) -> String {
    let mut result = String::new();
    let mut has_engine_line = false;
    let mut has_moonshine_section = false;
    let mut in_moonshine_section = false;
    let mut moonshine_model_updated = false;

    for line in config.lines() {
        let trimmed = line.trim();

        // Track sections
        if trimmed.starts_with('[') {
            // If we were in moonshine section and didn't update model, add it
            if in_moonshine_section && !moonshine_model_updated {
                result.push_str(&format!("model = \"{}\"\n", model_name));
                moonshine_model_updated = true;
            }
            in_moonshine_section = trimmed == "[moonshine]";
            if in_moonshine_section {
                has_moonshine_section = true;
            }
        }

        // Update or add engine line at the top level
        if trimmed.starts_with("engine") && !trimmed.starts_with('[') {
            result.push_str("engine = \"moonshine\"\n");
            has_engine_line = true;
        }
        // Update model line in moonshine section
        else if in_moonshine_section && trimmed.starts_with("model") {
            result.push_str(&format!("model = \"{}\"\n", model_name));
            moonshine_model_updated = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // If we were in moonshine section at EOF and didn't update model, add it
    if in_moonshine_section && !moonshine_model_updated {
        result.push_str(&format!("model = \"{}\"\n", model_name));
    }

    // Add engine line if not present
    if !has_engine_line {
        let mut new_result = String::new();
        let mut engine_added = false;
        for line in result.lines() {
            let trimmed = line.trim();
            if !engine_added
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("engine")
            {
                new_result.push_str("engine = \"moonshine\"\n\n");
                engine_added = true;
            }
            new_result.push_str(line);
            new_result.push('\n');
        }
        result = new_result;
    }

    // Add [moonshine] section if not present
    if !has_moonshine_section {
        result.push_str(&format!("\n[moonshine]\nmodel = \"{}\"\n", model_name));
    }

    // Remove trailing newline if original didn't have one
    if !config.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Handle SenseVoice model selection (download/config)
async fn handle_sensevoice_selection(selection: usize) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();

    if selection == 0 || selection > SENSEVOICE_MODELS.len() {
        println!("\nCancelled.");
        return Ok(());
    }

    let model = &SENSEVOICE_MODELS[selection - 1];
    let model_path = models_dir.join(model.dir_name);

    // Check if already installed
    if model_path.exists() && validate_sensevoice_model(&model_path).is_ok() {
        println!("\nModel '{}' is already installed.\n", model.dir_name);
        println!("  [1] Set as default model (update config)");
        println!("  [2] Re-download");
        println!("  [0] Cancel\n");

        print!("Select option [1]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "" | "1" => {
                update_config_sensevoice(model.name)?;
                restart_daemon_if_running().await;
                return Ok(());
            }
            "2" => {
                // Continue to download below
            }
            _ => {
                println!("Cancelled.");
                return Ok(());
            }
        }
    }

    // Download the model
    download_sensevoice_model_by_info(model)?;

    // Update config and restart daemon
    update_config_sensevoice(model.name)?;
    restart_daemon_if_running().await;

    Ok(())
}

/// List installed Moonshine models
pub fn list_installed_moonshine() {
    println!("\nInstalled Moonshine Models\n");
    println!("==========================\n");

    let models_dir = Config::models_dir();

    if !models_dir.exists() {
        println!("No models directory found: {:?}", models_dir);
        return;
    }

    let mut found = false;

    for model in MOONSHINE_MODELS {
        let model_path = models_dir.join(model.dir_name);

        if model_path.exists() && validate_moonshine_model(&model_path).is_ok() {
            let size = std::fs::read_dir(&model_path)
                .map(|entries| {
                    entries
                        .flatten()
                        .filter_map(|e| e.metadata().ok())
                        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                        .sum::<f64>()
                })
                .unwrap_or(0.0);

            let license_note = if model.license == "Community" {
                " [non-commercial]"
            } else {
                ""
            };

            println!(
                "  {} ({:.0} MB) - {} ({}){}",
                model.dir_name, size, model.description, model.language, license_note
            );
            found = true;
        }
    }

    if !found {
        println!("  No Moonshine models installed.");
        println!("\n  Run 'voxtype setup model' and select Moonshine to download.");
    }
}

// =============================================================================
// SenseVoice Model Functions
// =============================================================================

/// Check if a model name is a SenseVoice model
pub fn is_sensevoice_model(name: &str) -> bool {
    SENSEVOICE_MODELS.iter().any(|m| m.name == name)
}

/// Get the directory name for a SenseVoice model
pub fn sensevoice_dir_name(name: &str) -> Option<&'static str> {
    SENSEVOICE_MODELS
        .iter()
        .find(|m| m.name == name)
        .map(|m| m.dir_name)
}

/// Get list of valid SenseVoice model names
pub fn valid_sensevoice_model_names() -> Vec<&'static str> {
    SENSEVOICE_MODELS.iter().map(|m| m.name).collect()
}

/// Validate that a SenseVoice model directory has the required files
pub fn validate_sensevoice_model(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("Model directory does not exist: {:?}", path);
    }

    let has_model = path.join("model.int8.onnx").exists() || path.join("model.onnx").exists();
    let has_tokens = path.join("tokens.txt").exists();

    if has_model && has_tokens {
        Ok(())
    } else {
        let mut missing = Vec::new();
        if !has_model {
            missing.push("model.int8.onnx or model.onnx");
        }
        if !has_tokens {
            missing.push("tokens.txt");
        }
        anyhow::bail!(
            "Incomplete SenseVoice model, missing: {}",
            missing.join(", ")
        )
    }
}

/// Download a SenseVoice model by name (public API for run_setup)
pub fn download_sensevoice_model(model_name: &str) -> anyhow::Result<()> {
    let model = SENSEVOICE_MODELS
        .iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown SenseVoice model: {}", model_name))?;

    download_sensevoice_model_by_info(model)
}

/// Download a SenseVoice model using its info struct
fn download_sensevoice_model_by_info(model: &SenseVoiceModelInfo) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let model_path = models_dir.join(model.dir_name);
    let mut prompt_cache = IntegrityPromptCache::default();

    // Create model directory
    std::fs::create_dir_all(&model_path)?;

    println!(
        "\nDownloading {} ({} MB)...\n",
        model.dir_name, model.size_mb
    );

    for (repo_path, local_filename) in model.files {
        let file_path = model_path.join(local_filename);

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.huggingface_repo, repo_path
        );
        let hash_status =
            fetch_hf_expected_hash(model.huggingface_repo, repo_path, local_filename)?;
        let expected_hash = resolve_hash_status(&hash_status);
        let mut announced_download = false;

        let outcome = ensure_setup_file(&file_path, expected_hash, &mut prompt_cache, || {
            if !announced_download {
                println!("Downloading {}...", local_filename);
                announced_download = true;
            }
            curl_download(&file_path, &url, true)
        })?;

        report_setup_outcome(local_filename, outcome);
    }

    // Validate all files are present
    validate_sensevoice_model(&model_path)?;
    print_success(&format!(
        "Model '{}' downloaded to {:?}",
        model.dir_name, model_path
    ));

    Ok(())
}

/// Update config to use SenseVoice engine and a specific model (with status messages)
fn update_config_sensevoice(model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_sensevoice_in_config(&content, model_name);
            std::fs::write(&config_path, updated)?;
            print_success(&format!(
                "Config updated: engine = \"sensevoice\", model = \"{}\"",
                model_name
            ));
            Ok(())
        } else {
            print_info("No config file found. Run 'voxtype setup' first.");
            Ok(())
        }
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update the config to use SenseVoice engine with a specific model
fn update_sensevoice_in_config(config: &str, model_name: &str) -> String {
    let mut result = String::new();
    let mut has_engine_line = false;
    let mut has_sensevoice_section = false;
    let mut in_sensevoice_section = false;
    let mut sensevoice_model_updated = false;

    for line in config.lines() {
        let trimmed = line.trim();

        // Track sections
        if trimmed.starts_with('[') {
            if in_sensevoice_section && !sensevoice_model_updated {
                result.push_str(&format!("model = \"{}\"\n", model_name));
                sensevoice_model_updated = true;
            }
            in_sensevoice_section = trimmed == "[sensevoice]";
            if in_sensevoice_section {
                has_sensevoice_section = true;
            }
        }

        // Update or add engine line at the top level
        if trimmed.starts_with("engine") && !trimmed.starts_with('[') {
            result.push_str("engine = \"sensevoice\"\n");
            has_engine_line = true;
        }
        // Update model line in sensevoice section
        else if in_sensevoice_section && trimmed.starts_with("model") {
            result.push_str(&format!("model = \"{}\"\n", model_name));
            sensevoice_model_updated = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // If we were in sensevoice section at EOF and didn't update model, add it
    if in_sensevoice_section && !sensevoice_model_updated {
        result.push_str(&format!("model = \"{}\"\n", model_name));
    }

    // Add engine line if not present
    if !has_engine_line {
        let mut new_result = String::new();
        let mut engine_added = false;
        for line in result.lines() {
            let trimmed = line.trim();
            if !engine_added
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("engine")
            {
                new_result.push_str("engine = \"sensevoice\"\n\n");
                engine_added = true;
            }
            new_result.push_str(line);
            new_result.push('\n');
        }
        result = new_result;
    }

    // Add [sensevoice] section if not present
    if !has_sensevoice_section {
        result.push_str(&format!("\n[sensevoice]\nmodel = \"{}\"\n", model_name));
    }

    // Remove trailing newline if original didn't have one
    if !config.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// List installed SenseVoice models
pub fn list_installed_sensevoice() {
    println!("\nInstalled SenseVoice Models\n");
    println!("===========================\n");

    let models_dir = Config::models_dir();

    if !models_dir.exists() {
        println!("No models directory found: {:?}", models_dir);
        return;
    }

    let mut found = false;

    for model in SENSEVOICE_MODELS {
        let model_path = models_dir.join(model.dir_name);

        if model_path.exists() && validate_sensevoice_model(&model_path).is_ok() {
            let size = std::fs::read_dir(&model_path)
                .map(|entries| {
                    entries
                        .flatten()
                        .filter_map(|e| e.metadata().ok())
                        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                        .sum::<f64>()
                })
                .unwrap_or(0.0);

            println!(
                "  {} ({:.0} MB) - {} ({})",
                model.dir_name, size, model.description, model.languages
            );
            found = true;
        }
    }

    if !found {
        println!("  No SenseVoice models installed.");
        println!("\n  Run 'voxtype setup model' and select SenseVoice to download.");
    }
}

// =============================================================================
// Generic ONNX Engine Functions (Paraformer, Dolphin, Omnilingual)
// =============================================================================

/// Validate a CTC-based ONNX model directory (model.int8.onnx or model.onnx + tokens.txt)
fn validate_onnx_ctc_model(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("Model directory does not exist: {:?}", path);
    }

    let has_model = path.join("model.int8.onnx").exists() || path.join("model.onnx").exists();
    let has_tokens = path.join("tokens.txt").exists();

    if has_model && has_tokens {
        Ok(())
    } else {
        let mut missing = Vec::new();
        if !has_model {
            missing.push("model.int8.onnx or model.onnx");
        }
        if !has_tokens {
            missing.push("tokens.txt");
        }
        anyhow::bail!("Incomplete model, missing: {}", missing.join(", "))
    }
}

/// Generic handler for ONNX engine model selection (download/config/restart)
#[allow(clippy::type_complexity)]
async fn handle_onnx_engine_selection(
    engine_name: &str,
    models: Vec<(&str, &str, u32, &[(&str, &str)], &str)>,
    selection: usize,
    validate_fn: fn(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();

    if selection == 0 || selection > models.len() {
        println!("\nCancelled.");
        return Ok(());
    }

    let (name, dir_name, size_mb, files, repo) = &models[selection - 1];
    let model_path = models_dir.join(dir_name);

    // Check if already installed
    if model_path.exists() && validate_fn(&model_path).is_ok() {
        println!("\nModel '{}' is already installed.\n", dir_name);
        println!("  [1] Set as default model (update config)");
        println!("  [2] Re-download");
        println!("  [0] Cancel\n");

        print!("Select option [1]: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        match choice {
            "" | "1" => {
                update_config_engine(engine_name, name)?;
                restart_daemon_if_running().await;
                return Ok(());
            }
            "2" => {
                // Continue to download below
            }
            _ => {
                println!("Cancelled.");
                return Ok(());
            }
        }
    }

    // Download the model
    download_onnx_model(dir_name, *size_mb, files, repo)?;

    // Validate
    validate_fn(&model_path)?;
    print_success(&format!(
        "Model '{}' downloaded to {:?}",
        dir_name, model_path
    ));

    // Update config and restart daemon
    update_config_engine(engine_name, name)?;
    restart_daemon_if_running().await;

    Ok(())
}

/// Download an ONNX model from HuggingFace
fn download_onnx_model(
    dir_name: &str,
    size_mb: u32,
    files: &[(&str, &str)],
    repo: &str,
) -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let model_path = models_dir.join(dir_name);
    let mut prompt_cache = IntegrityPromptCache::default();

    std::fs::create_dir_all(&model_path)?;

    println!("\nDownloading {} ({} MB)...\n", dir_name, size_mb);

    for (repo_path, local_filename) in files {
        let file_path = model_path.join(local_filename);

        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, repo_path);
        let hash_status = fetch_hf_expected_hash(repo, repo_path, local_filename)?;
        let expected_hash = resolve_hash_status(&hash_status);
        let mut announced_download = false;

        let outcome = ensure_setup_file(&file_path, expected_hash, &mut prompt_cache, || {
            if !announced_download {
                println!("Downloading {}...", local_filename);
                announced_download = true;
            }
            curl_download(&file_path, &url, true)
        })?;

        report_setup_outcome(local_filename, outcome);
    }

    Ok(())
}

/// Update config to use a specific engine and model
fn update_config_engine(engine_name: &str, model_name: &str) -> anyhow::Result<()> {
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_engine_in_config(&content, engine_name, model_name);
            std::fs::write(&config_path, updated)?;
            print_success(&format!(
                "Config updated: engine = \"{}\", model = \"{}\"",
                engine_name, model_name
            ));
            Ok(())
        } else {
            print_info("No config file found. Run 'voxtype setup' first.");
            Ok(())
        }
    } else {
        anyhow::bail!("Could not determine config path")
    }
}

/// Update a config string to use a specific engine and model
fn update_engine_in_config(config: &str, engine_name: &str, model_name: &str) -> String {
    let section_name = format!("[{}]", engine_name);
    let mut result = String::new();
    let mut has_engine_line = false;
    let mut has_section = false;
    let mut in_section = false;
    let mut model_updated = false;

    for line in config.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('[') {
            if in_section && !model_updated {
                result.push_str(&format!("model = \"{}\"\n", model_name));
                model_updated = true;
            }
            in_section = trimmed == section_name;
            if in_section {
                has_section = true;
            }
        }

        if trimmed.starts_with("engine") && !trimmed.starts_with('[') {
            result.push_str(&format!("engine = \"{}\"\n", engine_name));
            has_engine_line = true;
        } else if in_section && trimmed.starts_with("model") {
            result.push_str(&format!("model = \"{}\"\n", model_name));
            model_updated = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    if in_section && !model_updated {
        result.push_str(&format!("model = \"{}\"\n", model_name));
    }

    if !has_engine_line {
        let mut new_result = String::new();
        let mut engine_added = false;
        for line in result.lines() {
            let trimmed = line.trim();
            if !engine_added
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("engine")
            {
                new_result.push_str(&format!("engine = \"{}\"\n\n", engine_name));
                engine_added = true;
            }
            new_result.push_str(line);
            new_result.push('\n');
        }
        result = new_result;
    }

    if !has_section {
        result.push_str(&format!(
            "\n[{}]\nmodel = \"{}\"\n",
            engine_name, model_name
        ));
    }

    if !config.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::cell::Cell;

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    fn leaked_hash(bytes: &[u8]) -> &'static str {
        Box::leak(sha256_hex(bytes).into_boxed_str())
    }

    fn test_helper_asset(filename: &'static str, sha256: &'static str) -> RuntimeHelperAssetSpec {
        RuntimeHelperAssetSpec {
            filename,
            url: "https://example.com/model.onnx",
            sha256,
            download_message: "Downloading helper asset...",
            success_message: "Helper asset downloaded.",
            download_failure_message: "Warning: helper asset download failed.",
            curl_missing_message: "Warning: curl not available.",
        }
    }

    #[test]
    fn test_update_model_in_config_basic() {
        let config = r#"[whisper]
model = "base.en"
language = "en"
"#;
        let result = update_model_in_config(config, "large-v3");
        assert!(result.contains(r#"model = "large-v3""#));
        assert!(!result.contains("base.en"));
    }

    #[test]
    fn test_update_model_in_config_switches_engine_to_whisper() {
        // When switching to a Whisper model, engine should be set to whisper
        let config = r#"engine = "parakeet"

[whisper]
model = "small"

[parakeet]
model = "parakeet-tdt-0.6b-v3"
"#;
        let result = update_model_in_config(config, "base.en");
        // Engine should now be whisper
        assert!(result.contains(r#"engine = "whisper""#));
        assert!(!result.contains(r#"engine = "parakeet""#));
        // Whisper model should be updated
        assert!(result.contains(r#"model = "base.en""#));
        // Parakeet section should be preserved
        assert!(result.contains("[parakeet]"));
        assert!(result.contains(r#"model = "parakeet-tdt-0.6b-v3""#));
    }

    #[test]
    fn test_update_model_in_config_preserves_other_sections() {
        let config = r#"[hotkey]
key = "SCROLLLOCK"

[whisper]
model = "base.en"
language = "en"

[output]
mode = "type"
"#;
        let result = update_model_in_config(config, "small.en");
        assert!(result.contains(r#"model = "small.en""#));
        assert!(result.contains(r#"key = "SCROLLLOCK""#));
        assert!(result.contains(r#"mode = "type""#));
        assert!(result.contains("[hotkey]"));
        assert!(result.contains("[output]"));
    }

    #[test]
    fn test_update_model_in_config_only_changes_whisper_section() {
        // If there's a "model" key in another section, it should not be changed
        let config = r#"[some_other_section]
model = "should_not_change"

[whisper]
model = "base.en"
"#;
        let result = update_model_in_config(config, "large-v3");
        assert!(result.contains(r#"model = "should_not_change""#));
        assert!(result.contains(r#"model = "large-v3""#));
    }

    #[test]
    fn test_update_model_in_config_handles_comments() {
        let config = r#"[whisper]
# Model to use
model = "base.en"
# Language setting
language = "en"
"#;
        let result = update_model_in_config(config, "medium.en");
        assert!(result.contains(r#"model = "medium.en""#));
        assert!(result.contains("# Model to use"));
        assert!(result.contains("# Language setting"));
    }

    #[test]
    fn test_models_list_contains_expected_models() {
        let model_names: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        // Multilingual models
        assert!(model_names.contains(&"tiny"));
        assert!(model_names.contains(&"base"));
        assert!(model_names.contains(&"small"));
        assert!(model_names.contains(&"medium"));
        // English-only models
        assert!(model_names.contains(&"tiny.en"));
        assert!(model_names.contains(&"base.en"));
        assert!(model_names.contains(&"small.en"));
        assert!(model_names.contains(&"medium.en"));
        // Large models (multilingual only)
        assert!(model_names.contains(&"large-v3"));
        assert!(model_names.contains(&"large-v3-turbo"));
    }

    #[test]
    fn test_model_info_sizes_are_reasonable() {
        for model in MODELS {
            // All models should have positive size
            assert!(model.size_mb > 0, "Model {} has invalid size", model.name);
            // Tiny models should be smallest, large should be biggest
            if model.name.starts_with("tiny") {
                assert!(model.size_mb < 100);
            }
            if model.name == "large-v3" {
                assert!(model.size_mb > 2000);
            }
        }
    }

    #[test]
    fn test_is_valid_model() {
        // Valid multilingual models
        assert!(is_valid_model("tiny"));
        assert!(is_valid_model("base"));
        assert!(is_valid_model("small"));
        assert!(is_valid_model("medium"));
        // Valid English-only models
        assert!(is_valid_model("tiny.en"));
        assert!(is_valid_model("base.en"));
        assert!(is_valid_model("small.en"));
        assert!(is_valid_model("medium.en"));
        // Valid large models
        assert!(is_valid_model("large-v3"));
        assert!(is_valid_model("large-v3-turbo"));

        // Invalid models
        assert!(!is_valid_model("invalid"));
        assert!(!is_valid_model("large"));
        assert!(!is_valid_model(""));
        assert!(!is_valid_model("LARGE-V3")); // case sensitive
    }

    #[test]
    fn test_valid_model_names() {
        let names = valid_model_names();
        assert!(names.contains(&"tiny.en"));
        assert!(names.contains(&"large-v3-turbo"));
        assert_eq!(names.len(), MODELS.len());
    }

    // =========================================================================
    // Parakeet Model Tests
    // =========================================================================

    #[test]
    fn test_parakeet_models_list_contains_expected_models() {
        let model_names: Vec<&str> = PARAKEET_MODELS.iter().map(|m| m.name).collect();
        assert!(model_names.contains(&"parakeet-tdt-0.6b-v3"));
        assert!(model_names.contains(&"parakeet-tdt-0.6b-v3-int8"));
    }

    #[test]
    fn test_parakeet_model_info_sizes_are_reasonable() {
        for model in PARAKEET_MODELS {
            // All models should have positive size
            assert!(model.size_mb > 0, "Model {} has invalid size", model.name);
            // Full model should be larger than quantized
            if model.name == "parakeet-tdt-0.6b-v3" {
                assert!(model.size_mb > 2000);
            }
            if model.name == "parakeet-tdt-0.6b-v3-int8" {
                assert!(model.size_mb < 1000);
            }
        }
    }

    #[test]
    fn test_parakeet_models_have_files() {
        for model in PARAKEET_MODELS {
            assert!(
                !model.files.is_empty(),
                "Model {} should have file definitions",
                model.name
            );
            // All TDT models should have vocab.txt
            assert!(
                model.files.iter().any(|(f, _)| *f == "vocab.txt"),
                "Model {} should have vocab.txt",
                model.name
            );
        }
    }

    #[test]
    fn test_is_parakeet_model() {
        // Valid Parakeet models
        assert!(is_parakeet_model("parakeet-tdt-0.6b-v3"));
        assert!(is_parakeet_model("parakeet-tdt-0.6b-v3-int8"));

        // Invalid models
        assert!(!is_parakeet_model("base.en"));
        assert!(!is_parakeet_model("large-v3"));
        assert!(!is_parakeet_model("parakeet")); // Not a full model name
        assert!(!is_parakeet_model(""));
    }

    #[test]
    fn test_valid_parakeet_model_names() {
        let names = valid_parakeet_model_names();
        assert!(names.contains(&"parakeet-tdt-0.6b-v3"));
        assert!(names.contains(&"parakeet-tdt-0.6b-v3-int8"));
        assert_eq!(names.len(), PARAKEET_MODELS.len());
    }

    #[test]
    fn test_update_parakeet_in_config_basic() {
        let config = r#"[hotkey]
key = "SCROLLLOCK"

[whisper]
model = "base.en"
language = "en"

[output]
mode = "type"
"#;
        let result = update_parakeet_in_config(config, "parakeet-tdt-0.6b-v3");

        // Should add engine = "parakeet"
        assert!(result.contains(r#"engine = "parakeet""#));
        // Should add [parakeet] section with model
        assert!(result.contains("[parakeet]"));
        assert!(result.contains(r#"model = "parakeet-tdt-0.6b-v3""#));
        // Should preserve existing sections
        assert!(result.contains("[whisper]"));
        assert!(result.contains("[hotkey]"));
        assert!(result.contains("[output]"));
    }

    #[test]
    fn test_update_parakeet_in_config_updates_existing() {
        let config = r#"engine = "whisper"

[hotkey]
key = "SCROLLLOCK"

[whisper]
model = "base.en"
language = "en"

[parakeet]
model = "old-model"

[output]
mode = "type"
"#;
        let result = update_parakeet_in_config(config, "parakeet-tdt-0.6b-v3-int8");

        // Should update engine to parakeet
        assert!(result.contains(r#"engine = "parakeet""#));
        assert!(!result.contains(r#"engine = "whisper""#));
        // Should update existing parakeet model
        assert!(result.contains(r#"model = "parakeet-tdt-0.6b-v3-int8""#));
        assert!(!result.contains(r#"model = "old-model""#));
    }

    #[test]
    fn test_update_parakeet_preserves_whisper_section() {
        let config = r#"[whisper]
model = "large-v3"
language = "en"
translate = false
"#;
        let result = update_parakeet_in_config(config, "parakeet-tdt-0.6b-v3");

        // Whisper section should be preserved
        assert!(result.contains("[whisper]"));
        assert!(result.contains(r#"model = "large-v3""#));
        assert!(result.contains(r#"language = "en""#));
        // Parakeet section should be added separately
        assert!(result.contains("[parakeet]"));
    }

    #[test]
    fn test_whisper_and_parakeet_models_dont_overlap() {
        // Ensure no model name is valid for both Whisper and Parakeet
        let whisper_names = valid_model_names();
        let parakeet_names = valid_parakeet_model_names();

        for name in &whisper_names {
            assert!(
                !parakeet_names.contains(name),
                "Model '{}' should not be in both Whisper and Parakeet lists",
                name
            );
        }

        for name in &parakeet_names {
            assert!(
                !whisper_names.contains(name),
                "Model '{}' should not be in both Whisper and Parakeet lists",
                name
            );
        }
    }

    // =========================================================================
    // Star Indicator Tests (for model selection menu)
    // =========================================================================

    #[test]
    fn test_star_indicator_whisper_model_selected() {
        use crate::config::TranscriptionEngine;

        // Simulate: engine=Whisper, current model="base.en"
        let is_whisper_engine =
            matches!(TranscriptionEngine::Whisper, TranscriptionEngine::Whisper);
        let current_whisper_model = "base.en";

        // "base.en" should have star
        let is_current = is_whisper_engine && "base.en" == current_whisper_model;
        assert!(
            is_current,
            "base.en should show star when it's the current Whisper model"
        );

        // "small.en" should NOT have star
        let is_current = is_whisper_engine && "small.en" == current_whisper_model;
        assert!(
            !is_current,
            "small.en should not show star when base.en is current"
        );
    }

    #[test]
    fn test_star_indicator_parakeet_model_selected() {
        use crate::config::TranscriptionEngine;

        // Simulate: engine=Parakeet, current model="parakeet-tdt-0.6b-v3"
        let is_parakeet_engine =
            matches!(TranscriptionEngine::Parakeet, TranscriptionEngine::Parakeet);
        let current_parakeet_model: Option<&str> = Some("parakeet-tdt-0.6b-v3");

        // "parakeet-tdt-0.6b-v3" should have star
        let is_current =
            is_parakeet_engine && current_parakeet_model == Some("parakeet-tdt-0.6b-v3");
        assert!(
            is_current,
            "parakeet-tdt-0.6b-v3 should show star when it's the current Parakeet model"
        );

        // "parakeet-tdt-0.6b-v3-int8" should NOT have star
        let is_current =
            is_parakeet_engine && current_parakeet_model == Some("parakeet-tdt-0.6b-v3-int8");
        assert!(
            !is_current,
            "parakeet-tdt-0.6b-v3-int8 should not show star when other model is current"
        );
    }

    #[test]
    fn test_star_indicator_engine_mismatch() {
        use crate::config::TranscriptionEngine;

        // When engine is Parakeet, Whisper models should NOT show star
        let is_whisper_engine =
            matches!(TranscriptionEngine::Parakeet, TranscriptionEngine::Whisper);
        let current_whisper_model = "base.en";

        let is_current = is_whisper_engine && "base.en" == current_whisper_model;
        assert!(
            !is_current,
            "Whisper models should not show star when engine is Parakeet"
        );

        // When engine is Whisper, Parakeet models should NOT show star
        let is_parakeet_engine =
            matches!(TranscriptionEngine::Whisper, TranscriptionEngine::Parakeet);
        let current_parakeet_model: Option<&str> = Some("parakeet-tdt-0.6b-v3");

        let is_current =
            is_parakeet_engine && current_parakeet_model == Some("parakeet-tdt-0.6b-v3");
        assert!(
            !is_current,
            "Parakeet models should not show star when engine is Whisper"
        );
    }

    #[test]
    fn test_star_indicator_no_parakeet_config() {
        use crate::config::TranscriptionEngine;

        // When parakeet config is None (not configured)
        let is_parakeet_engine =
            matches!(TranscriptionEngine::Parakeet, TranscriptionEngine::Parakeet);
        let current_parakeet_model: Option<&str> = None;

        // No model should show star when no parakeet config exists
        let is_current =
            is_parakeet_engine && current_parakeet_model == Some("parakeet-tdt-0.6b-v3");
        assert!(
            !is_current,
            "No star should show when parakeet config is not set"
        );
    }

    // =========================================================================
    // Moonshine Model Tests
    // =========================================================================

    #[test]
    fn test_moonshine_models_list_contains_expected_models() {
        let model_names: Vec<&str> = MOONSHINE_MODELS.iter().map(|m| m.name).collect();
        assert!(model_names.contains(&"base"));
        assert!(model_names.contains(&"tiny"));
    }

    #[test]
    fn test_moonshine_model_info_sizes_are_reasonable() {
        for model in MOONSHINE_MODELS {
            assert!(model.size_mb > 0, "Model {} has invalid size", model.name);
            if model.name.contains("tiny") {
                assert!(model.size_mb <= 150);
            }
            if model.name == "base" {
                assert!(model.size_mb > 150);
            }
        }
    }

    #[test]
    fn test_moonshine_models_have_files() {
        for model in MOONSHINE_MODELS {
            assert!(
                !model.files.is_empty(),
                "Model {} should have file definitions",
                model.name
            );
            // All models should have tokenizer.json
            assert!(
                model
                    .files
                    .iter()
                    .any(|(_, local)| *local == "tokenizer.json"),
                "Model {} should have tokenizer.json",
                model.name
            );
            // All models should have encoder
            assert!(
                model
                    .files
                    .iter()
                    .any(|(_, local)| *local == "encoder_model.onnx"),
                "Model {} should have encoder_model.onnx",
                model.name
            );
        }
    }

    #[test]
    fn test_is_moonshine_model() {
        // Valid Moonshine models
        assert!(is_moonshine_model("base"));
        assert!(is_moonshine_model("tiny"));
        assert!(is_moonshine_model("base-ja"));
        assert!(is_moonshine_model("tiny-ko"));

        // Invalid models
        assert!(!is_moonshine_model("base.en"));
        assert!(!is_moonshine_model("large-v3"));
        assert!(!is_moonshine_model("moonshine"));
        assert!(!is_moonshine_model(""));
    }

    #[test]
    fn test_valid_moonshine_model_names() {
        let names = valid_moonshine_model_names();
        assert!(names.contains(&"base"));
        assert!(names.contains(&"tiny"));
        assert_eq!(names.len(), MOONSHINE_MODELS.len());
    }

    #[test]
    fn test_moonshine_english_models_are_mit() {
        for model in MOONSHINE_MODELS {
            if model.language == "en" {
                assert_eq!(
                    model.license, "MIT",
                    "English model {} should be MIT licensed",
                    model.name
                );
            }
        }
    }

    #[test]
    fn test_moonshine_multilingual_models_are_community() {
        for model in MOONSHINE_MODELS {
            if model.language != "en" {
                assert_eq!(
                    model.license, "Community",
                    "Non-English model {} should be Community licensed",
                    model.name
                );
            }
        }
    }

    #[test]
    fn test_update_moonshine_in_config_basic() {
        let config = r#"engine = "whisper"

[whisper]
model = "base.en"
language = "en"

[output]
mode = "type"
"#;
        let result = update_moonshine_in_config(config, "base");

        // Should update engine to moonshine
        assert!(result.contains(r#"engine = "moonshine""#));
        assert!(!result.contains(r#"engine = "whisper""#));
        // Should add [moonshine] section with model
        assert!(result.contains("[moonshine]"));
        assert!(result.contains(r#"model = "base""#));
        // Should preserve existing sections
        assert!(result.contains("[whisper]"));
        assert!(result.contains("[output]"));
    }

    #[test]
    fn test_update_moonshine_in_config_updates_existing() {
        let config = r#"engine = "whisper"

[whisper]
model = "base.en"

[moonshine]
model = "tiny"
quantized = false

[output]
mode = "type"
"#;
        let result = update_moonshine_in_config(config, "base-ja");

        // Should update engine to moonshine
        assert!(result.contains(r#"engine = "moonshine""#));
        // Should update existing moonshine model
        assert!(result.contains(r#"model = "base-ja""#));
        assert!(!result.contains(r#"model = "tiny""#));
        // Should preserve quantized setting
        assert!(result.contains("quantized = false"));
    }

    #[test]
    fn test_moonshine_and_parakeet_models_dont_overlap() {
        // Moonshine and Parakeet model names should not overlap
        // (Whisper and Moonshine CAN share short names like "tiny" and "base"
        // because they're in different config sections)
        let parakeet_names = valid_parakeet_model_names();
        let moonshine_names = valid_moonshine_model_names();

        for name in &parakeet_names {
            assert!(
                !moonshine_names.contains(name),
                "Model '{}' should not be in both Parakeet and Moonshine lists",
                name
            );
        }
    }

    #[test]
    fn test_moonshine_dir_names_match_convention() {
        for model in MOONSHINE_MODELS {
            assert!(
                model.dir_name.starts_with("moonshine-"),
                "Model dir_name '{}' should start with 'moonshine-'",
                model.dir_name
            );
        }
    }

    #[test]
    fn test_validate_cohere_model_accepts_all_variants() {
        // Regression test for #357: validator hard-coded legacy filenames
        // (cohere-encoder.int8.onnx, tokens.txt) that the post-Optimum
        // downloader never produces. Every variant must validate against
        // the files it actually writes to disk.
        for model in COHERE_MODELS {
            let tmp = tempfile::tempdir().unwrap();
            let model_dir = tmp.path().join(model.dir_name);
            std::fs::create_dir_all(&model_dir).unwrap();
            for (_remote, local) in model.files {
                std::fs::write(model_dir.join(local), b"").unwrap();
            }
            validate_cohere_model(&model_dir)
                .unwrap_or_else(|e| panic!("variant {} failed validation: {}", model.name, e));
        }
    }

    #[test]
    fn test_validate_cohere_model_reports_missing_files() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("cohere-transcribe-fp16");
        std::fs::create_dir_all(&model_dir).unwrap();
        // Missing every file
        let err = validate_cohere_model(&model_dir).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("encoder_model.onnx"), "got: {}", msg);
        assert!(msg.contains("tokenizer.json"), "got: {}", msg);
    }

    #[test]
    fn test_valid_cached_gtcrn_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let good = b"gtcrn-good";
        let hash = leaked_hash(good);
        let spec = test_helper_asset(GTCRN_MODEL_FILENAME, hash);
        let model_path = tmp.path().join(GTCRN_MODEL_FILENAME);
        std::fs::write(&model_path, good).unwrap();
        let download_calls = Cell::new(0);

        let result = ensure_runtime_helper_asset_with(tmp.path(), &spec, |_, _, _| {
            download_calls.set(download_calls.get() + 1);
            Ok(())
        });

        assert_eq!(result, Some(model_path));
        assert_eq!(download_calls.get(), 0);
    }

    #[test]
    fn test_gtcrn_mismatch_repaired_silently() {
        let tmp = tempfile::tempdir().unwrap();
        let good = b"gtcrn-good";
        let hash = leaked_hash(good);
        let spec = test_helper_asset(GTCRN_MODEL_FILENAME, hash);
        let model_path = tmp.path().join(GTCRN_MODEL_FILENAME);
        std::fs::write(&model_path, b"bad").unwrap();
        let download_calls = Cell::new(0);
        let progress_flags = Cell::new(0);

        let result =
            ensure_runtime_helper_asset_with(tmp.path(), &spec, |path, _, show_progress| {
                download_calls.set(download_calls.get() + 1);
                progress_flags.set(progress_flags.get() + usize::from(show_progress));
                std::fs::write(path, good).unwrap();
                Ok(())
            });

        assert_eq!(result, Some(model_path.clone()));
        assert_eq!(download_calls.get(), 1);
        assert_eq!(progress_flags.get(), 0, "cached repair should stay silent");
        assert_eq!(std::fs::read(&model_path).unwrap(), good);
    }

    #[test]
    fn test_gtcrn_repair_failure_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let hash = leaked_hash(b"gtcrn-good");
        let spec = test_helper_asset(GTCRN_MODEL_FILENAME, hash);
        let model_path = tmp.path().join(GTCRN_MODEL_FILENAME);
        std::fs::write(&model_path, b"bad").unwrap();
        let download_calls = Cell::new(0);

        let result = ensure_runtime_helper_asset_with(tmp.path(), &spec, |path, _, _| {
            download_calls.set(download_calls.get() + 1);
            std::fs::write(path, b"still-bad").unwrap();
            Ok(())
        });

        assert!(result.is_none());
        assert_eq!(download_calls.get(), 1);
        assert!(
            !model_path.exists(),
            "failed repair should remove the bad helper asset"
        );
    }

    #[test]
    fn test_valid_cached_ecapa_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let good = b"ecapa-good";
        let hash = leaked_hash(good);
        let spec = test_helper_asset(ECAPA_MODEL_FILENAME, hash);
        let model_path = tmp.path().join(ECAPA_MODEL_FILENAME);
        std::fs::write(&model_path, good).unwrap();
        let download_calls = Cell::new(0);

        let result = ensure_runtime_helper_asset_with(tmp.path(), &spec, |_, _, _| {
            download_calls.set(download_calls.get() + 1);
            Ok(())
        });

        assert_eq!(result, Some(model_path));
        assert_eq!(download_calls.get(), 0);
    }

    #[test]
    fn test_ecapa_mismatch_repaired_silently() {
        let tmp = tempfile::tempdir().unwrap();
        let good = b"ecapa-good";
        let hash = leaked_hash(good);
        let spec = test_helper_asset(ECAPA_MODEL_FILENAME, hash);
        let model_path = tmp.path().join(ECAPA_MODEL_FILENAME);
        std::fs::write(&model_path, b"bad").unwrap();
        let download_calls = Cell::new(0);

        let result =
            ensure_runtime_helper_asset_with(tmp.path(), &spec, |path, _, show_progress| {
                download_calls.set(download_calls.get() + 1);
                assert!(!show_progress, "cached helper repair should stay silent");
                std::fs::write(path, good).unwrap();
                Ok(())
            });

        assert_eq!(result, Some(model_path.clone()));
        assert_eq!(download_calls.get(), 1);
        assert_eq!(std::fs::read(&model_path).unwrap(), good);
    }

    #[test]
    fn test_ecapa_repair_failure_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let hash = leaked_hash(b"ecapa-good");
        let spec = test_helper_asset(ECAPA_MODEL_FILENAME, hash);
        let model_path = tmp.path().join(ECAPA_MODEL_FILENAME);
        std::fs::write(&model_path, b"bad").unwrap();
        let download_calls = Cell::new(0);

        let result = ensure_runtime_helper_asset_with(tmp.path(), &spec, |path, _, _| {
            download_calls.set(download_calls.get() + 1);
            std::fs::write(path, b"still-bad").unwrap();
            Ok(())
        });

        assert!(result.is_none());
        assert_eq!(download_calls.get(), 1);
        assert!(
            !model_path.exists(),
            "failed repair should remove the bad helper asset"
        );
    }

    #[test]
    fn test_explicit_setup_paths_reverify_cached_files_before_skipping() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("cached.bin");
        let good = b"setup-good";
        std::fs::write(&file_path, good).unwrap();
        let expected = sha256_hex(good);
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);
        let mut prompt_cache = IntegrityPromptCache::default();

        let outcome = ensure_setup_file_with_prompt(
            &file_path,
            Some(&expected),
            &mut prompt_cache,
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                false
            },
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::ReusedVerified);
        assert_eq!(download_calls.get(), 0);
        assert_eq!(prompt_calls.get(), 0);
    }

    #[test]
    fn test_explicit_setup_paths_only_prompt_after_repair_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("cached.bin");
        std::fs::write(&file_path, b"bad").unwrap();
        let expected = sha256_hex(b"setup-good");
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);
        let mut prompt_cache = IntegrityPromptCache::default();

        let err = ensure_setup_file_with_prompt(
            &file_path,
            Some(&expected),
            &mut prompt_cache,
            || {
                download_calls.set(download_calls.get() + 1);
                std::fs::write(&file_path, b"still-bad").unwrap();
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                false
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("SHA-256 mismatch"));
        assert_eq!(
            download_calls.get(),
            1,
            "cached mismatch should get one repair attempt before prompting"
        );
        assert_eq!(prompt_calls.get(), 1);
        assert!(
            !file_path.exists(),
            "declined mismatch should remove the bad cached file"
        );
    }

    // =========================================================================
    // resolve_hash_status tests
    // =========================================================================

    #[test]
    fn test_resolve_hash_status_known_hash_returns_some_str() {
        let hash = "a".repeat(64);
        let status = CanonicalHashStatus::KnownHash(hash.clone());
        assert_eq!(resolve_hash_status(&status), Some(hash.as_str()));
    }

    #[test]
    fn test_resolve_hash_status_no_canonical_hash_returns_none() {
        let status = CanonicalHashStatus::NoCanonicalHash;
        assert_eq!(resolve_hash_status(&status), None);
    }

    // =========================================================================
    // ensure_model_file_with_fetcher tests
    // =========================================================================

    /// Builds a temporary file path inside `dir` with the given content.
    fn write_tmp(dir: &tempfile::TempDir, name: &str, content: &[u8]) -> std::path::PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn test_ensure_model_file_with_fetcher_known_hash_verifies() {
        // File already on disk with correct content; KnownHash should verify it.
        let dir = tempfile::tempdir().unwrap();
        let data = b"correct data";
        let hash = sha256_hex(data);
        let file_path = write_tmp(&dir, "model.bin", data);

        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);

        let outcome = ensure_model_file_with_fetcher(
            &file_path,
            "test/repo",
            "model.bin",
            "model.bin",
            &mut prompt_cache,
            |_repo, _filename| Ok(CanonicalHashStatus::KnownHash(hash.clone())),
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::ReusedVerified);
        assert_eq!(
            download_calls.get(),
            0,
            "no download needed for already-correct file"
        );
    }

    #[test]
    fn test_ensure_model_file_with_fetcher_no_canonical_hash_proceeds_without_verification() {
        // NoCanonicalHash means the file is inline (small), no LFS hash.
        // The engine should proceed without verification (hash=None path).
        let dir = tempfile::tempdir().unwrap();
        let file_path = write_tmp(&dir, "config.json", b"{\"key\":\"val\"}");
        let download_calls = Cell::new(0);
        let mut prompt_cache = IntegrityPromptCache::default();

        let outcome = ensure_model_file_with_fetcher(
            &file_path,
            "test/repo",
            "config.json",
            "config.json",
            &mut prompt_cache,
            |_repo, _filename| Ok(CanonicalHashStatus::NoCanonicalHash),
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
        )
        .unwrap();

        // No hash → no verification → ReusedWithoutHash (file already existed)
        assert_eq!(outcome, IntegrityOutcome::ReusedWithoutHash);
        assert_eq!(download_calls.get(), 0);
    }

    #[test]
    fn test_ensure_model_file_with_fetcher_metadata_error_fails_closed() {
        // Transport/parse error: hash_fetcher returns Err.
        // The function MUST propagate the error, not proceed as if hash=None.
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.bin");
        // File doesn't exist — would need to be downloaded.
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);

        let err = ensure_model_file_with_fetcher(
            &file_path,
            "test/repo",
            "model.bin",
            "model.bin",
            &mut prompt_cache,
            |_repo, _filename| Err(anyhow::anyhow!("network timeout")),
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("network timeout")
                || err.to_string().contains("canonical hash"),
            "error must carry context about the metadata failure; got: {err}"
        );
        assert_eq!(
            download_calls.get(),
            0,
            "must not attempt download after metadata failure"
        );
    }

    #[test]
    fn test_ensure_model_file_with_fetcher_missing_in_metadata_fails_closed() {
        // File not found in HF metadata → fetch_hf_lfs_sha256 returns Err("not in siblings").
        // Simulated by returning Err here. This must not silently proceed.
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("missing.onnx");
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);

        let err = ensure_model_file_with_fetcher(
            &file_path,
            "test/repo",
            "missing.onnx",
            "missing.onnx",
            &mut prompt_cache,
            |_repo, _filename| {
                Err(anyhow::anyhow!(
                    "'missing.onnx' not found in HuggingFace repository metadata for 'test/repo'"
                ))
            },
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
        )
        .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("missing.onnx") || msg.contains("not found"),
            "error should reference the missing file; got: {msg}"
        );
        assert_eq!(download_calls.get(), 0);
    }

    #[test]
    fn test_multi_file_download_fails_on_metadata_error_for_one_file() {
        // Simulate a multi-file download loop: one file has a network failure
        // fetching its hash. The entire loop should stop (? propagation).
        let dir = tempfile::tempdir().unwrap();
        let mut prompt_cache = IntegrityPromptCache::default();
        let files = [
            ("encoder.onnx", b"enc-data" as &[u8]),
            ("vocab.txt", b"vocab"),
        ];
        let download_calls = Cell::new(0);
        let fetcher_calls = Cell::new(0);

        // Simulate: first file OK, second file → network error
        let result: anyhow::Result<()> = (|| {
            for (filename, content) in &files {
                let file_path = write_tmp(&dir, filename, content);
                let call_n = fetcher_calls.get();
                fetcher_calls.set(call_n + 1);

                let outcome = ensure_model_file_with_fetcher(
                    &file_path,
                    "test/repo",
                    filename,
                    filename,
                    &mut prompt_cache,
                    |_repo, _file| {
                        if *filename == "vocab.txt" {
                            Err(anyhow::anyhow!("timeout fetching hash for vocab.txt"))
                        } else {
                            Ok(CanonicalHashStatus::KnownHash(sha256_hex(content)))
                        }
                    },
                    || {
                        download_calls.set(download_calls.get() + 1);
                        Ok(())
                    },
                )?;
                let _ = outcome;
            }
            Ok(())
        })();

        assert!(result.is_err(), "loop must abort on metadata failure");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("vocab.txt") || msg.contains("timeout"),
            "error should identify which file caused the failure; got: {msg}"
        );
        assert_eq!(
            download_calls.get(),
            0,
            "no download should have started after the error"
        );
    }

    #[test]
    fn test_multi_file_download_succeeds_with_mixed_hash_statuses() {
        // One file has a KnownHash, another is NoCanonicalHash (small inline file).
        // Both should succeed without error.
        let dir = tempfile::tempdir().unwrap();
        let mut prompt_cache = IntegrityPromptCache::default();
        let data_onnx = b"onnx model data";
        let data_config = b"{\"version\":1}";
        let hash_onnx = sha256_hex(data_onnx);

        let files = [
            ("encoder.onnx", data_onnx as &[u8], Some(hash_onnx.clone())),
            ("config.json", data_config, None), // inline, NoCanonicalHash
        ];

        for (filename, content, maybe_hash) in &files {
            let file_path = write_tmp(&dir, filename, content);
            let hash_clone = maybe_hash.clone();

            let outcome = ensure_model_file_with_fetcher(
                &file_path,
                "test/repo",
                filename,
                filename,
                &mut prompt_cache,
                move |_repo, _file| {
                    Ok(match hash_clone {
                        Some(ref h) => CanonicalHashStatus::KnownHash(h.clone()),
                        None => CanonicalHashStatus::NoCanonicalHash,
                    })
                },
                || Ok(()),
            )
            .unwrap();

            match maybe_hash {
                Some(_) => assert_eq!(outcome, IntegrityOutcome::ReusedVerified),
                None => assert_eq!(outcome, IntegrityOutcome::ReusedWithoutHash),
            }
        }
    }

    // =========================================================================
    // classify_outcome / SetupOutcomeKind reporting-helper tests
    // =========================================================================

    #[test]
    fn test_classify_outcome_accepted_by_user_is_unverified_not_downloaded() {
        // AcceptedByUser must NOT collapse into the generic Downloaded or Other
        // kinds — it needs its own distinct warning category.
        let kind = classify_outcome(IntegrityOutcome::AcceptedByUser);
        assert_eq!(kind, SetupOutcomeKind::AcceptedByUserUnverified);
        assert_ne!(kind, SetupOutcomeKind::Downloaded);
        assert_ne!(kind, SetupOutcomeKind::Other);
    }

    #[test]
    fn test_classify_outcome_reused_verified_maps_correctly() {
        assert_eq!(
            classify_outcome(IntegrityOutcome::ReusedVerified),
            SetupOutcomeKind::AlreadyVerified
        );
    }

    #[test]
    fn test_classify_outcome_reused_without_hash_maps_correctly() {
        assert_eq!(
            classify_outcome(IntegrityOutcome::ReusedWithoutHash),
            SetupOutcomeKind::AlreadyCachedNoHash
        );
    }

    #[test]
    fn test_classify_outcome_repaired_maps_correctly() {
        assert_eq!(
            classify_outcome(IntegrityOutcome::Repaired),
            SetupOutcomeKind::Repaired
        );
    }

    #[test]
    fn test_classify_outcome_downloaded_verified_maps_correctly() {
        assert_eq!(
            classify_outcome(IntegrityOutcome::DownloadedVerified),
            SetupOutcomeKind::Downloaded
        );
    }

    // =========================================================================
    // Integration-style: explicit setup orchestration → AcceptedByUser
    // =========================================================================

    /// Drive `ensure_setup_file_with_prompt` so that every download writes
    /// wrong content, the user accepts the risk, and the outcome is
    /// `AcceptedByUser`.  Then assert that `classify_outcome` does NOT map
    /// that outcome to `Downloaded` or `Other` — i.e. the reporting layer
    /// preserves the distinction.
    #[test]
    fn test_explicit_setup_accepted_by_user_is_classified_as_unverified() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("model.bin");
        let expected = sha256_hex(b"correct-content");
        let mut prompt_cache = IntegrityPromptCache::default();

        // Both download attempts write the wrong content.
        let outcome = ensure_setup_file_with_prompt(
            &file_path,
            Some(&expected),
            &mut prompt_cache,
            || {
                std::fs::write(&file_path, b"wrong-content").unwrap();
                Ok(())
            },
            |_, _| true, // user accepts
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::AcceptedByUser);

        // The reporting helper must distinguish this from a normal download.
        let kind = classify_outcome(outcome);
        assert_eq!(kind, SetupOutcomeKind::AcceptedByUserUnverified);
        assert_ne!(kind, SetupOutcomeKind::Downloaded);
        assert_ne!(kind, SetupOutcomeKind::Other);
    }
}
