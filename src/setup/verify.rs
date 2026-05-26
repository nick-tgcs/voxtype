//! Download integrity verification for voxtype models.
//!
//! Provides SHA-256 verification of downloaded model files and
//! integration with the HuggingFace LFS metadata API to fetch
//! expected hashes at download time without a hardcoded manifest.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum IntegrityContext {
    ExplicitSetup,
    RuntimeHelperAsset,
    RuntimePrimaryCompatibility,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum IntegrityOutcome {
    ReusedVerified,
    DownloadedVerified,
    Repaired,
    ReusedWithoutHash,
    DownloadedWithoutHash,
    AcceptedByUser,
    WarnedAndReused,
    Fallback,
}

#[derive(Debug, Default)]
pub struct IntegrityPromptCache {
    decisions: HashMap<PathBuf, bool>,
}

impl IntegrityPromptCache {
    pub fn decide_with<F>(&mut self, path: &Path, err: &anyhow::Error, prompt: F) -> bool
    where
        F: FnOnce(&Path, &anyhow::Error) -> bool,
    {
        if let Some(decision) = self.decisions.get(path) {
            return *decision;
        }

        let decision = prompt(path, err);
        self.decisions.insert(path.to_path_buf(), decision);
        decision
    }
}

enum FileIntegrityState {
    Verified,
    NoExpectedHash,
    Mismatch(anyhow::Error),
}

/// Verify a downloaded file's SHA-256 digest against an expected hex string.
///
/// The comparison is case-insensitive so that hashes produced by `sha256sum`
/// (lowercase) and those from other tools (uppercase) both work.
///
/// Returns `Ok(())` if the digest matches, or an error describing the
/// mismatch so the caller can clean up the file and surface a useful message.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> anyhow::Result<()> {
    // Validate the expected hex string before doing any file I/O.
    if expected_hex.len() != 64 || !expected_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!(
            "Invalid expected SHA-256 hash {:?}: must be 64 hex characters",
            expected_hex
        );
    }

    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Cannot open {:?} for SHA-256 verification: {}", path, e))?;

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let computed: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    if computed.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        anyhow::bail!(
            "SHA-256 mismatch for {:?}:\n  expected: {}\n  computed: {}",
            path,
            expected_hex.to_lowercase(),
            computed
        )
    }
}

// Rollout strategy: backwards compatibility is the primary constraint for the
// first integrity pass. Fresh downloads and explicit setup flows stay strict,
// because the user asked for that file and we can repair or prompt there
// without surprising them. Automatic runtime helpers are treated differently:
// they silently verify cached GTCRN/ECAPA-style assets, try one silent repair,
// then fall back so meeting mode keeps working without prompts or hard
// failures. Already-installed primary transcription models are different again:
// this rollout intentionally does not introduce a new runtime hard gate for
// Whisper or ONNX models, because breaking existing installs on upgrade would
// be worse than tolerating a warn-only compatibility path until a later,
// better-migrated rollout.
pub fn ensure_file_integrity<F>(
    path: &Path,
    expected_hash: Option<&str>,
    context: IntegrityContext,
    prompt_cache: &mut IntegrityPromptCache,
    download_file: F,
) -> anyhow::Result<IntegrityOutcome>
where
    F: FnMut() -> anyhow::Result<()>,
{
    ensure_file_integrity_with_prompt(
        path,
        expected_hash,
        context,
        prompt_cache,
        download_file,
        prompt_hash_mismatch_continue,
    )
}

pub(crate) fn ensure_file_integrity_with_prompt<F, P>(
    path: &Path,
    expected_hash: Option<&str>,
    context: IntegrityContext,
    prompt_cache: &mut IntegrityPromptCache,
    mut download_file: F,
    mut prompt: P,
) -> anyhow::Result<IntegrityOutcome>
where
    F: FnMut() -> anyhow::Result<()>,
    P: FnMut(&Path, &anyhow::Error) -> bool,
{
    let had_cached_file = path.exists();
    let mut repaired = false;

    if had_cached_file {
        match inspect_file_integrity(path, expected_hash) {
            FileIntegrityState::Verified => return Ok(IntegrityOutcome::ReusedVerified),
            FileIntegrityState::NoExpectedHash => {
                return Ok(IntegrityOutcome::ReusedWithoutHash);
            }
            FileIntegrityState::Mismatch(err) => match context {
                IntegrityContext::RuntimePrimaryCompatibility => {
                    tracing::warn!(
                        path = ?path,
                        error = %err,
                        "Integrity mismatch on existing primary model; keeping cached file for compatibility"
                    );
                    return Ok(IntegrityOutcome::WarnedAndReused);
                }
                IntegrityContext::ExplicitSetup | IntegrityContext::RuntimeHelperAsset => {
                    repaired = true;
                    remove_file_if_exists(path);
                }
            },
        }
    }

    let mut attempts_remaining = if expected_hash.is_some() && !repaired {
        2
    } else {
        1
    };
    let mut last_err = None;

    while attempts_remaining > 0 {
        download_file()?;

        match inspect_file_integrity(path, expected_hash) {
            FileIntegrityState::Verified => {
                return Ok(if repaired {
                    IntegrityOutcome::Repaired
                } else {
                    IntegrityOutcome::DownloadedVerified
                });
            }
            FileIntegrityState::NoExpectedHash => {
                return Ok(IntegrityOutcome::DownloadedWithoutHash);
            }
            FileIntegrityState::Mismatch(err) => {
                attempts_remaining -= 1;
                last_err = Some(err);

                if attempts_remaining > 0 {
                    repaired = true;
                    remove_file_if_exists(path);
                }
            }
        }
    }

    let err =
        last_err.expect("integrity verification exhausted download attempts without an error");
    match context {
        IntegrityContext::ExplicitSetup => {
            if prompt_cache.decide_with(path, &err, |path, err| prompt(path, err)) {
                Ok(IntegrityOutcome::AcceptedByUser)
            } else {
                remove_file_if_exists(path);
                Err(err)
            }
        }
        IntegrityContext::RuntimeHelperAsset => {
            remove_file_if_exists(path);
            Ok(IntegrityOutcome::Fallback)
        }
        IntegrityContext::RuntimePrimaryCompatibility => {
            tracing::warn!(
                path = ?path,
                error = %err,
                "Downloaded primary model failed integrity verification; keeping file for compatibility"
            );
            Ok(IntegrityOutcome::WarnedAndReused)
        }
    }
}

/// After a SHA-256 mismatch, show the user what happened and ask whether to
/// proceed anyway or abort.
///
/// Returns `true` if the user explicitly confirms they want to continue
/// despite the mismatch. Defaults to `false` (abort) on empty input, read
/// errors, or any input other than `y` / `yes`.
pub fn prompt_hash_mismatch_continue(_path: &Path, err: &anyhow::Error) -> bool {
    let stdin = std::io::stdin();
    prompt_with_reader(err, &mut stdin.lock())
}

fn inspect_file_integrity(path: &Path, expected_hash: Option<&str>) -> FileIntegrityState {
    if !path.exists() {
        return FileIntegrityState::Mismatch(anyhow::anyhow!(
            "Expected {:?} to exist for integrity verification",
            path
        ));
    }

    match expected_hash {
        Some(hash) => match verify_sha256(path, hash) {
            Ok(()) => FileIntegrityState::Verified,
            Err(err) => FileIntegrityState::Mismatch(err),
        },
        None => FileIntegrityState::NoExpectedHash,
    }
}

fn remove_file_if_exists(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Inner implementation of [`prompt_hash_mismatch_continue`] that reads from
/// any [`BufRead`], making it testable without real stdin.
///
/// [`BufRead`]: std::io::BufRead
fn prompt_with_reader(err: &anyhow::Error, reader: &mut dyn std::io::BufRead) -> bool {
    use std::io::Write;

    println!();
    println!("  WARNING: {err}");
    println!();
    println!("  This could mean:");
    println!("    - A corrupted or partial download (consider retrying)");
    println!("    - A file served from a compromised or unexpected source");
    println!("    - An updated model whose hash has changed since this release");
    println!();
    print!("  Proceed with this file anyway? [y/N]: ");
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    if reader.read_line(&mut input).is_err() {
        println!();
        println!("  (Could not read input — aborting for safety.)");
        return false;
    }

    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

// ---------------------------------------------------------------------------
// Canonical hash metadata
// ---------------------------------------------------------------------------

/// The result of a successful HuggingFace canonical-hash lookup for a single
/// file.
///
/// This type is deliberately richer than `Option<String>` so that callers
/// cannot accidentally conflate two distinct conditions:
///
/// - [`KnownHash`] – the file exists in the HF repo metadata *and* has an LFS
///   SHA-256 pointer. This hash should be used to verify the download.
/// - [`NoCanonicalHash`] – the file exists in the HF repo metadata but has no
///   LFS hash (typically a small inline file such as `config.json`). This is a
///   legitimate successful lookup; the file is just not LFS-tracked.
///
/// When the metadata fetch itself fails, or when the file is not listed in the
/// repository's `siblings` array at all, the lookup returns `Err` rather than
/// one of these variants. That means callers can distinguish every relevant
/// state without resorting to extra booleans or overloaded `Option`:
///
/// | Situation                              | Return value          |
/// |----------------------------------------|-----------------------|
/// | File found with LFS SHA-256            | `Ok(KnownHash(hash))` |
/// | File found, no LFS hash (inline file)  | `Ok(NoCanonicalHash)` |
/// | File absent from repo metadata         | `Err(...)`            |
/// | Network or parse failure               | `Err(...)`            |
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalHashStatus {
    /// The file is in HuggingFace metadata and has a known LFS SHA-256 hash.
    KnownHash(String),
    /// The file is in HuggingFace metadata but has no LFS hash.
    ///
    /// Typical for small config/vocab files stored inline by HF rather than
    /// via Git LFS. The file genuinely has no canonical hash; proceeding
    /// without verification is an intentional, accepted trade-off.
    NoCanonicalHash,
}

/// Parse the HuggingFace model-blobs API response and return the canonical
/// hash status for a specific file.
///
/// Returns `Some(CanonicalHashStatus::KnownHash(hex))` when the file is
/// found in the `siblings` list and has an LFS SHA-256 pointer.
///
/// Returns `Some(CanonicalHashStatus::NoCanonicalHash)` when the file is
/// found in the `siblings` list but has no LFS block (small inline file).
///
/// Returns `None` when:
/// - The filename is not listed in the `siblings` array at all, or
/// - The response structure is unexpected/missing the `siblings` key.
///
/// Crucially, `None` means *the file is absent from metadata*, which is
/// distinct from `NoCanonicalHash` (file present but inline). Callers that
/// convert this `Option` to `Result` must map `None` to an error.
pub fn parse_hf_blobs_sha256(
    response: &serde_json::Value,
    filename: &str,
) -> Option<CanonicalHashStatus> {
    let siblings = response.get("siblings")?.as_array()?;
    for entry in siblings {
        if entry.get("rfilename")?.as_str()? == filename {
            // File found. Return KnownHash if an LFS block is present,
            // NoCanonicalHash if the file is stored inline (no LFS).
            let status = match entry
                .get("lfs")
                .and_then(|lfs| lfs.get("sha256"))
                .and_then(|v| v.as_str())
            {
                Some(sha256) => CanonicalHashStatus::KnownHash(sha256.to_owned()),
                None => CanonicalHashStatus::NoCanonicalHash,
            };
            return Some(status);
        }
    }
    // File not found in siblings — deliberately returns None so the caller
    // can convert it to a meaningful error.
    None
}

/// Fetch the canonical hash status for a file hosted in a HuggingFace
/// repository by calling the model blobs API.
///
/// Returns `Ok(CanonicalHashStatus::KnownHash(hex))` when the file is found
/// in metadata and has an LFS SHA-256.
///
/// Returns `Ok(CanonicalHashStatus::NoCanonicalHash)` when the file is found
/// but is stored inline with no LFS hash (e.g. small `config.json` files).
///
/// Returns `Err` on:
/// - Network failure
/// - API response parse failure
/// - Unexpected response structure (e.g., missing `siblings` key)
/// - The file is **not listed** in the repository's `siblings` metadata — this
///   is treated as an error because callers only request hashes for files
///   expected to exist; a missing entry indicates a misconfiguration or a
///   changed repository layout.
///
/// # Note on trust model
/// The hash is fetched from the same HF domain that serves the file, so
/// this defends against CDN / third-party cache poisoning but not against
/// a fully compromised HF account. For the highest assurance, pin hashes
/// in a local manifest (planned for the models.voxtype.io CDN work).
pub fn fetch_hf_lfs_sha256(repo: &str, filename: &str) -> anyhow::Result<CanonicalHashStatus> {
    let url = format!("https://huggingface.co/api/models/{}?blobs=true", repo);
    let response: serde_json::Value = ureq::get(&url)
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to fetch HF model metadata for {}: {}", repo, e))?
        .into_json()
        .map_err(|e| anyhow::anyhow!("Failed to parse HF API response for {}: {}", repo, e))?;
    parse_hf_blobs_sha256(&response, filename).ok_or_else(|| {
        anyhow::anyhow!(
            "File '{}' not found in HuggingFace metadata for repo '{}'; \
             the repository layout may have changed",
            filename,
            repo
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::Cell;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Known-good SHA-256 digests used across multiple tests.
    // Verified with: echo -n "hello world" | sha256sum
    const SHA256_HELLO_WORLD: &str =
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
    // Verified with: printf '' | sha256sum
    const SHA256_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    // Binary content: bytes 0x00..0xff repeated twice (512 bytes).
    // Verified with: python3 -c "import hashlib; print(hashlib.sha256(bytes(range(256))*2).hexdigest())"
    const SHA256_BINARY_512: &str =
        "110009dcee21620b166f3abfecb5eff7a873be729d1c2d53822e7acc5f34eb9b";

    // -----------------------------------------------------------------------
    // verify_sha256
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_sha256_correct_hash_passes() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"hello world").unwrap();
        assert!(
            verify_sha256(f.path(), SHA256_HELLO_WORLD).is_ok(),
            "correct hash must pass"
        );
    }

    #[test]
    fn test_verify_sha256_wrong_hash_returns_err() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"hello world").unwrap();
        let err = verify_sha256(f.path(), SHA256_EMPTY).unwrap_err();
        let msg = err.to_string();
        // The error should name both the expected and computed hashes
        assert!(
            msg.contains(SHA256_EMPTY),
            "error should include expected hash; got: {msg}"
        );
        assert!(
            msg.contains(SHA256_HELLO_WORLD),
            "error should include computed hash; got: {msg}"
        );
    }

    #[test]
    fn test_verify_sha256_empty_file_has_known_hash() {
        let f = NamedTempFile::new().unwrap();
        // File is empty by default after creation
        assert!(
            verify_sha256(f.path(), SHA256_EMPTY).is_ok(),
            "empty file must verify against the SHA-256 of zero bytes"
        );
    }

    #[test]
    fn test_verify_sha256_nonexistent_file_returns_err() {
        let path = std::path::PathBuf::from("/tmp/voxtype_test_does_not_exist_xyz.bin");
        assert!(
            verify_sha256(&path, SHA256_HELLO_WORLD).is_err(),
            "missing file must return an error"
        );
    }

    #[test]
    fn test_verify_sha256_hash_is_case_insensitive() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"hello world").unwrap();
        let upper = SHA256_HELLO_WORLD.to_uppercase();
        assert!(
            verify_sha256(f.path(), &upper).is_ok(),
            "uppercase expected hash must be accepted"
        );
    }

    #[test]
    fn test_verify_sha256_binary_content() {
        // 512 bytes: 0x00..0xff twice
        let data: Vec<u8> = (0u8..=255).chain(0u8..=255).collect();
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&data).unwrap();
        // We'll verify the hash matches what sha2 computes — the constant
        // SHA256_BINARY_512 was pre-computed; if the implementation is
        // correct this must pass.
        assert!(
            verify_sha256(f.path(), SHA256_BINARY_512).is_ok(),
            "binary content must hash correctly"
        );
    }

    #[test]
    fn test_verify_sha256_invalid_hex_string_returns_err() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"hello world").unwrap();
        assert!(
            verify_sha256(f.path(), "not-valid-hex!!!").is_err(),
            "garbage expected hash must be caught before file I/O"
        );
    }

    // -----------------------------------------------------------------------
    // parse_hf_blobs_sha256
    // -----------------------------------------------------------------------

    fn hf_fixture() -> serde_json::Value {
        json!({
            "siblings": [
                {
                    "rfilename": "encoder-model.onnx",
                    "size": 41770866,
                    "lfs": {
                        "sha256": "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb2222",
                        "size": 41770866,
                        "pointer_size": 134
                    }
                },
                {
                    // Small file — no LFS block
                    "rfilename": "config.json",
                    "size": 97
                },
                {
                    "rfilename": "vocab.txt",
                    "size": 9384,
                    "lfs": {
                        "sha256": "0000aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb",
                        "size": 9384,
                        "pointer_size": 131
                    }
                }
            ]
        })
    }

    // -----------------------------------------------------------------------
    // parse_hf_blobs_sha256 — CanonicalHashStatus variant
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_hf_blobs_lfs_file_returns_known_hash() {
        let fixture = hf_fixture();
        let result = parse_hf_blobs_sha256(&fixture, "encoder-model.onnx");
        assert_eq!(
            result,
            Some(CanonicalHashStatus::KnownHash(
                "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb2222".to_string()
            ))
        );
    }

    #[test]
    fn test_parse_hf_blobs_non_lfs_file_returns_no_canonical_hash() {
        let fixture = hf_fixture();
        // config.json is small and not LFS-tracked: it IS in the siblings list
        // but has no LFS block. This must return NoCanonicalHash, NOT None.
        let result = parse_hf_blobs_sha256(&fixture, "config.json");
        assert_eq!(
            result,
            Some(CanonicalHashStatus::NoCanonicalHash),
            "inline (non-LFS) file must return NoCanonicalHash, got {result:?}"
        );
    }

    #[test]
    fn test_parse_hf_blobs_unknown_filename_returns_none() {
        let fixture = hf_fixture();
        // A file that is not present in the siblings list at all.
        // This returns None ("file absent"), which is distinct from
        // NoCanonicalHash ("file present but inline").
        let result = parse_hf_blobs_sha256(&fixture, "does_not_exist.bin");
        assert!(
            result.is_none(),
            "absent filename must return None (file-not-in-metadata signal)"
        );
    }

    #[test]
    fn test_parse_hf_blobs_distinguishes_no_lfs_from_missing_file() {
        // Core property: NoCanonicalHash and "file absent" must not be the same.
        let fixture = hf_fixture();
        let inline_result = parse_hf_blobs_sha256(&fixture, "config.json"); // inline file
        let absent_result = parse_hf_blobs_sha256(&fixture, "nonexistent.bin"); // not in siblings
        assert_eq!(inline_result, Some(CanonicalHashStatus::NoCanonicalHash));
        assert!(absent_result.is_none());
        assert_ne!(
            inline_result, absent_result,
            "inline file with no LFS hash must be distinct from an absent file"
        );
    }

    #[test]
    fn test_parse_hf_blobs_empty_siblings_returns_none() {
        let fixture = json!({ "siblings": [] });
        let result = parse_hf_blobs_sha256(&fixture, "encoder-model.onnx");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_hf_blobs_multiple_files_returns_correct_one() {
        let fixture = hf_fixture();
        // vocab.txt has a different hash from encoder-model.onnx
        let encoder_status = parse_hf_blobs_sha256(&fixture, "encoder-model.onnx").unwrap();
        let vocab_status = parse_hf_blobs_sha256(&fixture, "vocab.txt").unwrap();
        assert_ne!(
            encoder_status, vocab_status,
            "each file must return its own hash"
        );
        assert_eq!(
            vocab_status,
            CanonicalHashStatus::KnownHash(
                "0000aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb".to_string()
            )
        );
    }

    #[test]
    fn test_parse_hf_blobs_missing_siblings_key_returns_none() {
        // Malformed response without "siblings" key
        let fixture = json!({ "modelId": "some/repo" });
        let result = parse_hf_blobs_sha256(&fixture, "encoder-model.onnx");
        assert!(
            result.is_none(),
            "missing siblings key must not panic and must return None"
        );
    }

    #[test]
    fn test_canonical_hash_status_variants_are_distinct() {
        let known = CanonicalHashStatus::KnownHash("abc123".repeat(10).chars().take(64).collect());
        let no_hash = CanonicalHashStatus::NoCanonicalHash;
        assert_ne!(known, no_hash);
    }

    #[test]
    fn test_parse_hf_blobs_missing_file_maps_to_error_in_fetch() {
        // parse_hf_blobs_sha256 returns None for missing files, and
        // fetch_hf_lfs_sha256 converts that None to Err. Verify the
        // semantics at the parse level: the None from a missing file
        // is the signal that the caller converts to Err.
        let fixture = hf_fixture();
        let absent = parse_hf_blobs_sha256(&fixture, "missing.bin");
        assert!(
            absent.is_none(),
            "missing file returns None from parse, which fetch converts to Err"
        );
        // A file that exists inline (NoCanonicalHash) must NOT be None.
        let inline = parse_hf_blobs_sha256(&fixture, "config.json");
        assert!(
            inline.is_some(),
            "inline file returns Some(NoCanonicalHash), not None"
        );
    }

    // -----------------------------------------------------------------------
    // prompt_with_reader
    // -----------------------------------------------------------------------

    fn make_err() -> anyhow::Error {
        anyhow::anyhow!("SHA-256 mismatch for test.bin: expected aabb, computed ccdd")
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    #[test]
    fn test_prompt_y_returns_true() {
        let mut reader = std::io::Cursor::new(b"y\n".to_vec());
        assert!(
            prompt_with_reader(&make_err(), &mut reader),
            "'y' must continue"
        );
    }

    #[test]
    fn test_prompt_yes_returns_true() {
        let mut reader = std::io::Cursor::new(b"yes\n".to_vec());
        assert!(
            prompt_with_reader(&make_err(), &mut reader),
            "'yes' must continue"
        );
    }

    #[test]
    fn test_prompt_uppercase_y_returns_true() {
        let mut reader = std::io::Cursor::new(b"Y\n".to_vec());
        assert!(
            prompt_with_reader(&make_err(), &mut reader),
            "'Y' must continue"
        );
    }

    #[test]
    fn test_prompt_uppercase_yes_returns_true() {
        let mut reader = std::io::Cursor::new(b"YES\n".to_vec());
        assert!(
            prompt_with_reader(&make_err(), &mut reader),
            "'YES' must continue"
        );
    }

    #[test]
    fn test_prompt_n_returns_false() {
        let mut reader = std::io::Cursor::new(b"n\n".to_vec());
        assert!(
            !prompt_with_reader(&make_err(), &mut reader),
            "'n' must abort"
        );
    }

    #[test]
    fn test_prompt_no_returns_false() {
        let mut reader = std::io::Cursor::new(b"no\n".to_vec());
        assert!(
            !prompt_with_reader(&make_err(), &mut reader),
            "'no' must abort"
        );
    }

    #[test]
    fn test_prompt_empty_returns_false() {
        let mut reader = std::io::Cursor::new(b"\n".to_vec());
        assert!(
            !prompt_with_reader(&make_err(), &mut reader),
            "empty input must abort (safe default)"
        );
    }

    #[test]
    fn test_prompt_garbage_returns_false() {
        let mut reader = std::io::Cursor::new(b"maybe\n".to_vec());
        assert!(
            !prompt_with_reader(&make_err(), &mut reader),
            "arbitrary text must abort"
        );
    }

    #[test]
    fn test_prompt_eof_returns_false() {
        // Empty reader — read_line returns Ok(0), input stays ""
        let mut reader = std::io::Cursor::new(b"".to_vec());
        assert!(
            !prompt_with_reader(&make_err(), &mut reader),
            "EOF must abort (safe default)"
        );
    }

    #[test]
    fn test_prompt_whitespace_around_y_returns_true() {
        // "  y  \n" — leading/trailing spaces should be trimmed
        let mut reader = std::io::Cursor::new(b"  y  \n".to_vec());
        assert!(
            prompt_with_reader(&make_err(), &mut reader),
            "whitespace-padded 'y' must continue"
        );
    }

    #[test]
    fn test_prompt_cache_reuses_decision_for_same_path() {
        let err = make_err();
        let path = PathBuf::from("/tmp/voxtype-prompt-cache.bin");
        let calls = Cell::new(0);
        let mut prompt_cache = IntegrityPromptCache::default();

        assert!(prompt_cache.decide_with(&path, &err, |_, _| {
            calls.set(calls.get() + 1);
            true
        }));
        assert!(prompt_cache.decide_with(&path, &err, |_, _| {
            calls.set(calls.get() + 1);
            false
        }));
        assert_eq!(calls.get(), 1, "prompt should only run once per path");
    }

    #[test]
    fn test_explicit_setup_reuses_verified_cache_without_download() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("verified.bin");
        let good = b"verified";
        std::fs::write(&path, good).unwrap();
        let expected = sha256_hex(good);
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = ensure_file_integrity_with_prompt(
            &path,
            Some(&expected),
            IntegrityContext::ExplicitSetup,
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
    fn test_explicit_setup_repairs_cached_mismatch_before_prompting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("repair.bin");
        let good = b"good";
        std::fs::write(&path, b"bad").unwrap();
        let expected = sha256_hex(good);
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = ensure_file_integrity_with_prompt(
            &path,
            Some(&expected),
            IntegrityContext::ExplicitSetup,
            &mut prompt_cache,
            || {
                download_calls.set(download_calls.get() + 1);
                std::fs::write(&path, good).unwrap();
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
        assert_eq!(std::fs::read(&path).unwrap(), good);
    }

    #[test]
    fn test_explicit_setup_retries_fresh_mismatch_then_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fresh.bin");
        let expected = sha256_hex(b"expected");
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = ensure_file_integrity_with_prompt(
            &path,
            Some(&expected),
            IntegrityContext::ExplicitSetup,
            &mut prompt_cache,
            || {
                let next = download_calls.get() + 1;
                download_calls.set(next);
                std::fs::write(&path, b"wrong").unwrap();
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                true
            },
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::AcceptedByUser);
        assert_eq!(
            download_calls.get(),
            2,
            "fresh mismatch should retry once before prompting"
        );
        assert_eq!(prompt_calls.get(), 1);
        assert!(
            path.exists(),
            "accepted mismatch should keep the downloaded file"
        );
    }

    #[test]
    fn test_runtime_helper_mismatch_falls_back_without_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("helper.bin");
        std::fs::write(&path, b"bad").unwrap();
        let expected = sha256_hex(b"good");
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = ensure_file_integrity_with_prompt(
            &path,
            Some(&expected),
            IntegrityContext::RuntimeHelperAsset,
            &mut prompt_cache,
            || {
                download_calls.set(download_calls.get() + 1);
                std::fs::write(&path, b"still-bad").unwrap();
                Ok(())
            },
            |_, _| {
                prompt_calls.set(prompt_calls.get() + 1);
                true
            },
        )
        .unwrap();

        assert_eq!(outcome, IntegrityOutcome::Fallback);
        assert_eq!(download_calls.get(), 1);
        assert_eq!(prompt_calls.get(), 0, "runtime helpers must not prompt");
        assert!(
            !path.exists(),
            "fallback should remove the mismatched helper asset"
        );
    }

    #[test]
    fn test_primary_model_compatibility_warns_and_reuses_cached_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("primary.bin");
        std::fs::write(&path, b"bad").unwrap();
        let expected = sha256_hex(b"good");
        let mut prompt_cache = IntegrityPromptCache::default();
        let download_calls = Cell::new(0);
        let prompt_calls = Cell::new(0);

        let outcome = ensure_file_integrity_with_prompt(
            &path,
            Some(&expected),
            IntegrityContext::RuntimePrimaryCompatibility,
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

        assert_eq!(outcome, IntegrityOutcome::WarnedAndReused);
        assert_eq!(download_calls.get(), 0);
        assert_eq!(prompt_calls.get(), 0);
        assert!(
            path.exists(),
            "compatibility path must not remove existing primary models"
        );
    }
}
