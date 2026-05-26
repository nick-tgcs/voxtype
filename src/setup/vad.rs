//! VAD model download and status

use super::{print_info, print_success, print_warning};
use crate::config::Config;
use crate::setup::verify::{
    ensure_file_integrity_with_prompt, fetch_hf_lfs_sha256, prompt_hash_mismatch_continue,
    CanonicalHashStatus, IntegrityContext, IntegrityOutcome, IntegrityPromptCache,
};
use crate::vad::{get_whisper_vad_model_filename, get_whisper_vad_model_url};
use std::path::Path;
use std::process::Command;

const WHISPER_VAD_HF_REPO: &str = "ggml-org/whisper-vad";
const WHISPER_VAD_REMOTE_FILENAME: &str = "ggml-silero-v6.2.0.bin";

/// Fetch the canonical hash status for the Silero VAD model from HuggingFace
/// metadata.
///
/// Returns `Ok(CanonicalHashStatus::KnownHash(hex))` when the file is found
/// and has an LFS SHA-256.
///
/// Returns `Ok(CanonicalHashStatus::NoCanonicalHash)` when the file exists in
/// metadata but has no LFS hash.
///
/// Returns `Err` on network failure, parse failure, or if the file is not
/// listed in the repository metadata. The caller **must propagate** this error
/// rather than silently treating it as "no hash available".
fn fetch_vad_expected_hash() -> anyhow::Result<CanonicalHashStatus> {
    fetch_hf_lfs_sha256(WHISPER_VAD_HF_REPO, WHISPER_VAD_REMOTE_FILENAME)
}

fn download_with_curl(path: &Path, url: &str) -> anyhow::Result<()> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Model path contains non-UTF-8 bytes: {:?}", path))?;

    match Command::new("curl")
        .args(["-L", "--progress-bar", "-o", path_str, url])
        .status()
    {
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

fn download_model_with_hash_and_prompt<F, P>(
    models_dir: &Path,
    expected_hash: Option<&str>,
    download_file: F,
    prompt: P,
) -> anyhow::Result<IntegrityOutcome>
where
    F: FnMut() -> anyhow::Result<()>,
    P: FnMut(&Path, &anyhow::Error) -> bool,
{
    let filename = get_whisper_vad_model_filename();
    let model_path = models_dir.join(filename);
    let mut prompt_cache = IntegrityPromptCache::default();

    ensure_file_integrity_with_prompt(
        &model_path,
        expected_hash,
        IntegrityContext::ExplicitSetup,
        &mut prompt_cache,
        download_file,
        prompt,
    )
}

/// Inner download orchestration with an injectable hash fetcher.
///
/// In production `hash_fetcher` wraps `fetch_vad_expected_hash`; in tests a
/// stub is injected so the test runs offline.
pub(crate) fn download_model_with_hash_fetcher<H, F, P>(
    models_dir: &Path,
    hash_fetcher: H,
    download_file: F,
    prompt: P,
) -> anyhow::Result<IntegrityOutcome>
where
    H: FnOnce() -> anyhow::Result<CanonicalHashStatus>,
    F: FnMut() -> anyhow::Result<()>,
    P: FnMut(&Path, &anyhow::Error) -> bool,
{
    let hash_status = hash_fetcher()?;
    let expected_hash = match &hash_status {
        CanonicalHashStatus::KnownHash(h) => Some(h.as_str()),
        CanonicalHashStatus::NoCanonicalHash => None,
    };
    download_model_with_hash_and_prompt(models_dir, expected_hash, download_file, prompt)
}

/// Download the Silero VAD model
pub fn download_model() -> anyhow::Result<()> {
    let models_dir = Config::models_dir();
    let filename = get_whisper_vad_model_filename();
    let model_path = models_dir.join(filename);

    std::fs::create_dir_all(&models_dir)?;

    let url = get_whisper_vad_model_url();
    let mut announced_download = false;

    let outcome = download_model_with_hash_fetcher(
        &models_dir,
        fetch_vad_expected_hash,
        || {
            if !announced_download {
                println!("Downloading Silero VAD model...");
                println!("URL: {}", url);
                announced_download = true;
            }
            download_with_curl(&model_path, url)
        },
        prompt_hash_mismatch_continue,
    )?;

    match outcome {
        IntegrityOutcome::ReusedVerified => {
            print_success(&format!(
                "VAD model already installed and verified: {:?}",
                model_path
            ));
        }
        IntegrityOutcome::ReusedWithoutHash => {
            print_success(&format!("VAD model already installed: {:?}", model_path));
            print_warning("No canonical SHA-256 metadata was available for this cached file.");
        }
        IntegrityOutcome::AcceptedByUser => {
            print_warning(&format!(
                "Saved to {:?} \u{2014} WARNING: integrity verification failed. \
                 File kept only because you chose to continue. Treat as unverified.",
                model_path
            ));
        }
        _ => {
            print_success(&format!("Saved to {:?}", model_path));
        }
    }

    println!();
    print_info("Enable in config.toml:");
    println!("  [vad]");
    println!("  enabled = true");
    println!("  backend = \"whisper\"");
    Ok(())
}

/// Show VAD model status
pub fn show_status() {
    let models_dir = Config::models_dir();
    let filename = get_whisper_vad_model_filename();
    let model_path = models_dir.join(filename);

    println!("VAD Model Status\n");

    if model_path.exists() {
        let size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        print_success(&format!(
            "Silero VAD model installed: {:?} ({:.1} MB)",
            model_path,
            size as f64 / 1_048_576.0
        ));
    } else {
        print_warning("Silero VAD model not installed");
        print_info("Download with: voxtype setup vad");
        print_info("Energy VAD (no model needed) is available as an alternative.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    #[test]
    fn test_vad_setup_reuses_verified_cached_file() {
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        let good = b"vad-good";
        std::fs::write(&model_path, good).unwrap();
        let expected = sha256_hex(good);
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = download_model_with_hash_and_prompt(
            tmp.path(),
            Some(&expected),
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
    fn test_vad_setup_repairs_cached_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        let good = b"vad-good";
        std::fs::write(&model_path, b"bad").unwrap();
        let expected = sha256_hex(good);
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = download_model_with_hash_and_prompt(
            tmp.path(),
            Some(&expected),
            || {
                download_calls.set(download_calls.get() + 1);
                std::fs::write(&model_path, good).unwrap();
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                false
            },
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::Repaired);
        assert_eq!(download_calls.get(), 1);
        assert_eq!(prompt_calls.get(), 0);
        assert_eq!(std::fs::read(&model_path).unwrap(), good);
    }

    #[test]
    fn test_vad_setup_only_prompts_after_repair_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        std::fs::write(&model_path, b"bad").unwrap();
        let expected = sha256_hex(b"vad-good");
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let err = download_model_with_hash_and_prompt(
            tmp.path(),
            Some(&expected),
            || {
                download_calls.set(download_calls.get() + 1);
                std::fs::write(&model_path, b"still-bad").unwrap();
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                false
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("SHA-256 mismatch"));
        assert_eq!(download_calls.get(), 1);
        assert_eq!(prompt_calls.get(), 1);
        assert!(
            !model_path.exists(),
            "declined mismatch should remove the bad VAD file"
        );
    }

    // =========================================================================
    // download_model_with_hash_fetcher tests
    // =========================================================================

    #[test]
    fn test_vad_hash_fetcher_known_hash_verifies() {
        // KnownHash: existing file matches → ReusedVerified, no download.
        let tmp = tempfile::tempdir().unwrap();
        let data = b"vad-data";
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        std::fs::write(&model_path, data).unwrap();
        let expected = sha256_hex(data);
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = download_model_with_hash_fetcher(
            tmp.path(),
            || Ok(CanonicalHashStatus::KnownHash(expected)),
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
    fn test_vad_hash_fetcher_no_canonical_hash_proceeds() {
        // NoCanonicalHash: file exists, no hash available → proceeds unverified.
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        std::fs::write(&model_path, b"some-vad-data").unwrap();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = download_model_with_hash_fetcher(
            tmp.path(),
            || Ok(CanonicalHashStatus::NoCanonicalHash),
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

        assert_eq!(outcome, IntegrityOutcome::ReusedWithoutHash);
        assert_eq!(download_calls.get(), 0);
        assert_eq!(prompt_calls.get(), 0);
    }

    #[test]
    fn test_vad_hash_fetcher_metadata_error_fails_closed() {
        // Err from hash_fetcher must propagate — no silent fallback to no-hash.
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        std::fs::write(&model_path, b"existing-vad-data").unwrap();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let err = download_model_with_hash_fetcher(
            tmp.path(),
            || {
                Err(anyhow::anyhow!(
                    "network unreachable while fetching VAD hash"
                ))
            },
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                false
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("network unreachable") || err.to_string().contains("VAD hash"),
            "error must carry context about the metadata failure; got: {err}"
        );
        assert_eq!(
            download_calls.get(),
            0,
            "must not proceed to download after metadata failure"
        );
        assert_eq!(prompt_calls.get(), 0);
    }

    #[test]
    fn test_vad_hash_fetcher_file_not_in_metadata_fails_closed() {
        // Simulates the case where VAD model is not found in HF repo siblings →
        // fetch_hf_lfs_sha256 returns Err. Must fail closed.
        let tmp = tempfile::tempdir().unwrap();
        let download_calls = Cell::new(0);

        let err = download_model_with_hash_fetcher(
            tmp.path(),
            || {
                Err(anyhow::anyhow!(
                    "'silero_vad.onnx' not found in HuggingFace repository metadata"
                ))
            },
            || {
                download_calls.set(download_calls.get() + 1);
                Ok(())
            },
            |_, _| false,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("silero_vad.onnx") || msg.contains("not found"),
            "error should reference the missing file; got: {msg}"
        );
        assert_eq!(download_calls.get(), 0);
    }

    // =========================================================================
    // Integration-style: VAD explicit setup → AcceptedByUser
    // =========================================================================

    /// Force a persistent hash mismatch and have the user accept the risk.
    /// The outcome must be `AcceptedByUser`, and the reporting helper
    /// (`classify_outcome`) must classify it as `AcceptedByUserUnverified`
    /// rather than `Downloaded` or `Other`.
    #[test]
    fn test_vad_accepted_by_user_outcome_classified_as_unverified() {
        use crate::setup::model::{classify_outcome, SetupOutcomeKind};

        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join(get_whisper_vad_model_filename());
        let expected = sha256_hex(b"correct-vad-content");
        let prompt_calls = Cell::new(0);

        // Every download writes wrong content; the user accepts the mismatch.
        let outcome = download_model_with_hash_and_prompt(
            tmp.path(),
            Some(&expected),
            || {
                std::fs::write(&model_path, b"wrong-vad-content").unwrap();
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                true // user accepts
            },
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::AcceptedByUser);
        assert_eq!(prompt_calls.get(), 1);

        // Verify the reporting layer distinguishes this from a normal download.
        let kind = classify_outcome(outcome);
        assert_eq!(kind, SetupOutcomeKind::AcceptedByUserUnverified);
        assert_ne!(kind, SetupOutcomeKind::Downloaded);
        assert_ne!(kind, SetupOutcomeKind::Other);
    }
}
