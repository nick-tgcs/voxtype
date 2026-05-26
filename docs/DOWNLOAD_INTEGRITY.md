# Download Integrity Strategy

This page documents the compatibility-first rollout for model and helper-asset integrity checks in voxtype.

The goal is to tighten cached-file handling without breaking existing installs, without adding surprise runtime prompts, and without turning a routine upgrade into a hard failure for users who already have primary transcription models installed.

## Rollout Policy

Backwards compatibility is the primary constraint for this rollout.

- Fresh downloads stay strict.
- Explicit setup and download commands verify cached files before reusing them when a canonical hash is available.
- Optional runtime helper assets silently verify cached files, repair once, then fall back without prompting.
- Already-installed primary Whisper and ONNX models are not newly hard-gated at runtime in this release.

## Flow Types

### Explicit setup and download flows

This includes commands such as:

- `voxtype setup model`
- `voxtype setup vad`
- other explicit setup-time model downloads under `src/setup/model.rs`

Behavior:

- If the cached file has a canonical SHA-256, voxtype verifies it before skipping the download.
- If the cached file fails verification, voxtype deletes it and re-downloads once automatically.
- If the repaired file still fails verification, voxtype prompts once in that explicit setup flow and lets the user continue or abort.
- The same file is not prompted repeatedly in a single run.

This keeps setup strict while still giving users a recovery path when the upstream file changed or the local cache is damaged.

### Automatic runtime helper assets

This rollout treats small optional helper assets differently from primary models.

Current runtime helper assets covered by this policy:

- GTCRN echo-cancellation model used by meeting mode
- ECAPA-TDNN speaker-embedding model used by ML diarization

Behavior:

- Cached helper files are verified silently before reuse.
- A mismatched cached helper is re-downloaded once automatically and silently.
- If the repair still fails, voxtype does not prompt and does not hard-fail the feature entry point.
- Meeting mode falls back gracefully instead:
  - GTCRN falls back to no enhancement.
  - ECAPA falls back to simple diarization.

This avoids surprise questions during `voxtype meeting start` while still tightening correctness for helper assets that were previously trusted by existence alone.

### Runtime primary transcription models

Primary Whisper and ONNX transcription models are intentionally handled more conservatively in this release.

Behavior in this rollout:

- Fresh downloads in explicit setup remain strict.
- Already-installed primary models are not newly hard-gated during normal runtime.
- The compatibility path is warn-not-break, not prompt-or-block.

Why this is deferred: a new runtime hard gate would risk breaking working installs immediately after upgrade if a cached primary model fails a check that older versions never enforced. That is not acceptable for the first rollout.

### VAD

VAD is covered, but only in its explicit setup flow for this rollout.

Behavior:

- `voxtype setup vad` verifies a cached VAD file before reusing it when hash metadata is available.
- A mismatched cached VAD file is re-downloaded once automatically.
- Prompting is limited to the explicit setup command, and only after automatic repair fails.
- The runtime VAD load path does not start auto-downloading and does not start prompting.

## Trust Limits Of HuggingFace LFS Metadata

Many setup flows use the HuggingFace model metadata API to fetch SHA-256 values for LFS-tracked files.

This improves integrity checking, but it is not a full independent trust anchor.

- It defends against corrupted local caches, partial downloads, and some bad cached-file reuse.
- It does not independently defend against a fully compromised HuggingFace account or a malicious upstream repo update, because the hash and the file come from the same service.
- Small inline files on HuggingFace do not always expose an LFS SHA-256. Those files cannot be authenticated the same way in this rollout.

The long-term direction is a voxtype-controlled manifest and CDN flow so voxtype can pin and ship hashes independently of the download host.

## What Is Deferred

These pieces are intentionally left for a later rollout:

- Hard runtime enforcement for already-installed primary Whisper and ONNX models
- A persistent trust database
- Broader manifest-based verification for files that lack HuggingFace LFS hashes today
- A stronger independent trust root for primary model downloads

That staged approach keeps current installs working while still closing the most obvious cache-trust gap for setup flows and optional runtime helper assets.