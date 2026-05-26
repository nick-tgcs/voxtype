#!/bin/bash
# Build AppImage packages for voxtype
# Uses pre-built release binaries from releases/{version}/
#
# Produces 3 AppImages:
#   voxtype-{ver}-x86_64.AppImage          Whisper (avx2 + avx512 + vulkan)
#   voxtype-{ver}-onnx-x86_64.AppImage     ONNX engines (onnx-avx2 + onnx-avx512 + vulkan)
#   voxtype-{ver}-onnx-cuda-x86_64.AppImage  ONNX CUDA (onnx-cuda + vulkan)
#
# Usage:
#   ./scripts/build-appimage.sh [options] VERSION
#   ./scripts/build-appimage.sh 0.6.5
#   ./scripts/build-appimage.sh --variant whisper 0.6.5
#
# Options:
#   --variant NAME   whisper, onnx, onnx-cuda, all (default: all)
#   --skip-build     Accepted for compatibility (binaries must already exist)

set -euo pipefail

# ---------------------------------------------------------------------------
# Pinned appimagetool release
# To update: download the new release, run sha256sum on it, and update both
# constants below. The URL must contain the version string.
#
# To skip verification (not recommended — you accept the risk):
#   APPIMAGETOOL_SKIP_VERIFY=1 ./scripts/build-appimage.sh VERSION
# To allow an unpinned system appimagetool only if the pinned artifact cannot
# be reused or downloaded (not recommended — bypasses version pinning and
# SHA-256 verification):
#   APPIMAGETOOL_ALLOW_SYSTEM=1 ./scripts/build-appimage.sh VERSION
# ---------------------------------------------------------------------------
APPIMAGETOOL_VERSION="1.9.1"
APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage"

# ---------------------------------------------------------------------------
# verify_download_sha256 is defined before the source guard so that tests can
# source this script (with APPIMAGETOOL_SOURCE_ONLY=1) and call it directly.
# ---------------------------------------------------------------------------

# Verify the SHA-256 of a downloaded file against an expected hex string.
#
# Usage: verify_download_sha256 FILE EXPECTED_HEX
# Returns 0 on match. On mismatch: prints an error to stderr showing the
# actual hash, deletes FILE, and returns 1.
# Comparison is case-insensitive.
verify_download_sha256() {
    local file="$1" expected="$2"

    if [[ -z "$expected" ]]; then
        echo "  Error: expected SHA-256 is empty" >&2
        return 1
    fi

    if [[ ! -f "$file" ]]; then
        echo "  Error: file not found: $file" >&2
        return 1
    fi

    local actual
    actual=$(sha256sum "$file" | cut -d' ' -f1)

    if [[ "${actual,,}" != "${expected,,}" ]]; then
        echo "  SHA-256 mismatch for $(basename "$file")" >&2
        echo "    expected: ${expected,,}" >&2
        echo "    computed: $actual" >&2
        echo "  To skip verification (not recommended): set APPIMAGETOOL_SKIP_VERIFY=1" >&2
        rm -f "$file"
        return 1
    fi
}

# Find or download appimagetool, pinned to APPIMAGETOOL_VERSION.
#
# Resolution order:
#   1. Cached versioned binary    — verified against pinned SHA-256 on each use.
#   2. Fresh download             — verified before chmod +x or use.
#   3. Optional system fallback   — only when APPIMAGETOOL_ALLOW_SYSTEM=1;
#                                    bypasses the pinned version and verification.
#
# Set APPIMAGETOOL_SKIP_VERIFY=1 to bypass hash checking (accepts the risk).
find_appimagetool() {
    local cache_dir="$HOME/.local/bin"
    local cached="$cache_dir/appimagetool-${APPIMAGETOOL_VERSION}"

    # 1. Already cached — verify before reusing
    if [[ -x "$cached" ]]; then
        if [[ -n "${APPIMAGETOOL_SKIP_VERIFY:-}" ]]; then
            echo "  Warning: SHA-256 verification skipped (APPIMAGETOOL_SKIP_VERIFY is set)" >&2
            echo "$cached"
            return
        fi
        if verify_download_sha256 "$cached" "$APPIMAGETOOL_SHA256" 2>/dev/null; then
            echo "$cached"
            return
        fi
        echo "  Cached appimagetool failed verification; re-downloading..." >&2
        # verify_download_sha256 already deleted the corrupted cache file
    fi

    # 2. Download to a temp file, verify, then move into place
    echo "  Downloading appimagetool ${APPIMAGETOOL_VERSION}..." >&2
    mkdir -p "$cache_dir"
    local tmp
    tmp=$(mktemp "${cache_dir}/appimagetool-${APPIMAGETOOL_VERSION}.XXXXXX")

    if ! curl -fsSL -o "$tmp" "$APPIMAGETOOL_URL"; then
        rm -f "$tmp"
        echo "  Error: failed to download pinned appimagetool ${APPIMAGETOOL_VERSION}" >&2
        if [[ -n "${APPIMAGETOOL_ALLOW_SYSTEM:-}" ]] && command -v appimagetool >/dev/null 2>&1; then
            echo "  Warning: using system appimagetool from PATH because APPIMAGETOOL_ALLOW_SYSTEM is set" >&2
            echo "  Warning: this bypasses the pinned version and SHA-256 verification" >&2
            echo "appimagetool"
            return 0
        fi
        return 1
    fi

    if [[ -z "${APPIMAGETOOL_SKIP_VERIFY:-}" ]]; then
        if ! verify_download_sha256 "$tmp" "$APPIMAGETOOL_SHA256"; then
            # verify_download_sha256 already deleted the temp file
            echo "  Download aborted: SHA-256 verification failed." >&2
            echo "  Update APPIMAGETOOL_VERSION and APPIMAGETOOL_SHA256 in this script" >&2
            echo "  if appimagetool has been updated, or set APPIMAGETOOL_SKIP_VERIFY=1" >&2
            echo "  to bypass (not recommended)." >&2
            return 1
        fi
    else
        echo "  Warning: SHA-256 verification skipped (APPIMAGETOOL_SKIP_VERIFY is set)" >&2
    fi

    mv "$tmp" "$cached"
    chmod +x "$cached"
    echo "$cached"
}

# Allow sourcing this script to load functions without executing the build.
# Used by tests/test-build-appimage-verify.sh.
[[ -n "${APPIMAGETOOL_SOURCE_ONLY:-}" ]] && return 0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
VARIANT="all"
VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --skip-build)
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--variant NAME] VERSION"
            echo ""
            echo "Variants: whisper, onnx, onnx-cuda, all"
            echo ""
            echo "Uses pinned appimagetool (auto-downloaded and SHA-256 verified by default)"
            exit 0
            ;;
        *)
            VERSION="$1"
            shift
            ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    echo "Error: VERSION is required" >&2
    echo "Usage: $0 [--variant NAME] VERSION" >&2
    exit 1
fi

RELEASE_DIR="$PROJECT_DIR/releases/$VERSION"
APPIMAGE_DIR="$PROJECT_DIR/packaging/appimage"

if [[ ! -d "$RELEASE_DIR" ]]; then
    echo "Error: Release directory not found: $RELEASE_DIR" >&2
    echo "Build binaries first or check the version number." >&2
    exit 1
fi

APPIMAGETOOL="$(find_appimagetool)"
echo "Using appimagetool: $APPIMAGETOOL"

# Populate shared files (docs, completions, config) into an AppDir
populate_shared_files() {
    local appdir="$1"

    mkdir -p "$appdir/usr/share/doc/voxtype"
    cp "$PROJECT_DIR/README.md" "$appdir/usr/share/doc/voxtype/"
    cp "$PROJECT_DIR/LICENSE" "$appdir/usr/share/doc/voxtype/"

    # Default config
    mkdir -p "$appdir/etc/voxtype"
    cp "$PROJECT_DIR/config/default.toml" "$appdir/etc/voxtype/config.toml"

    # Shell completions
    mkdir -p "$appdir/usr/share/bash-completion/completions"
    mkdir -p "$appdir/usr/share/zsh/site-functions"
    mkdir -p "$appdir/usr/share/fish/vendor_completions.d"
    cp "$PROJECT_DIR/packaging/completions/voxtype.bash" "$appdir/usr/share/bash-completion/completions/voxtype"
    cp "$PROJECT_DIR/packaging/completions/voxtype.zsh" "$appdir/usr/share/zsh/site-functions/_voxtype"
    cp "$PROJECT_DIR/packaging/completions/voxtype.fish" "$appdir/usr/share/fish/vendor_completions.d/voxtype.fish"

    # Man pages (if available from a prior cargo build --release)
    local man_dir
    man_dir=$(find "$PROJECT_DIR/target/release/build" -name "man" -type d -path "*/voxtype-*/out/man" 2>/dev/null | head -1)
    if [[ -n "$man_dir" && -d "$man_dir" ]]; then
        mkdir -p "$appdir/usr/share/man/man1"
        cp "$man_dir"/*.1 "$appdir/usr/share/man/man1/"
    fi

    # Desktop entry and icon at AppDir root (AppImage spec)
    cp "$APPIMAGE_DIR/voxtype.desktop" "$appdir/"
    cp "$APPIMAGE_DIR/voxtype.svg" "$appdir/"
}

# Copy a binary into the AppDir if it exists
copy_binary() {
    local src="$1"
    local dst="$2"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        chmod 755 "$dst"
        return 0
    fi
    return 1
}

# Build a single AppImage from a prepared AppDir
build_appimage() {
    local appdir="$1"
    local output_name="$2"
    local output_path="$RELEASE_DIR/$output_name"

    echo "  Building $output_name..."
    ARCH=x86_64 "$APPIMAGETOOL" "$appdir" "$output_path" 2>&1 | tail -1
    chmod +x "$output_path"
    echo "  Created: $output_path ($(du -h "$output_path" | cut -f1))"
}

# Whisper AppImage: avx2 + avx512 + vulkan
# Use VOXTYPE_GPU=1 to select the Vulkan binary at runtime
build_whisper() {
    echo ""
    echo "Building Whisper AppImage (avx2 + avx512 + vulkan)..."

    local avx2="$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-avx2"
    if [[ ! -f "$avx2" ]]; then
        echo "  Skipping: $avx2 not found" >&2
        return 1
    fi

    local appdir
    appdir="$(mktemp -d "${TMPDIR:-/tmp}/voxtype-appimage.XXXXXX")"
    trap 'rm -rf "$appdir"' RETURN

    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/voxtype"

    # CPU-adaptive wrapper (handles GPU dispatch via VOXTYPE_GPU=1)
    cp "$SCRIPT_DIR/voxtype-wrapper.sh" "$appdir/usr/bin/voxtype"
    chmod 755 "$appdir/usr/bin/voxtype"

    # Whisper CPU binaries
    copy_binary "$avx2" "$appdir/usr/lib/voxtype/voxtype-avx2"
    copy_binary "$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-avx512" \
        "$appdir/usr/lib/voxtype/voxtype-avx512" || true

    # Whisper Vulkan GPU binary
    copy_binary "$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-vulkan" \
        "$appdir/usr/lib/voxtype/voxtype-vulkan" || true

    cp "$APPIMAGE_DIR/AppRun" "$appdir/"
    chmod 755 "$appdir/AppRun"

    populate_shared_files "$appdir"
    build_appimage "$appdir" "voxtype-${VERSION}-x86_64.AppImage"
}

# ONNX AppImage: onnx-avx2 + onnx-avx512 + vulkan
# Wrapper detects engine from config/CLI/env and dispatches accordingly
build_onnx() {
    echo ""
    echo "Building ONNX AppImage (onnx-avx2 + onnx-avx512 + vulkan)..."

    local onnx_avx2="$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-onnx-avx2"
    if [[ ! -f "$onnx_avx2" ]]; then
        echo "  Skipping: $onnx_avx2 not found" >&2
        return 1
    fi

    local appdir
    appdir="$(mktemp -d "${TMPDIR:-/tmp}/voxtype-appimage.XXXXXX")"
    trap 'rm -rf "$appdir"' RETURN

    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/voxtype"

    # Multi-engine wrapper (dispatches between ONNX and Vulkan)
    cp "$APPIMAGE_DIR/voxtype-onnx-wrapper.sh" "$appdir/usr/bin/voxtype"
    chmod 755 "$appdir/usr/bin/voxtype"

    # ONNX CPU binaries
    copy_binary "$onnx_avx2" "$appdir/usr/lib/voxtype/voxtype-onnx-avx2"
    copy_binary "$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-onnx-avx512" \
        "$appdir/usr/lib/voxtype/voxtype-onnx-avx512" || true

    # Vulkan binary for whisper engine fallback
    copy_binary "$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-vulkan" \
        "$appdir/usr/lib/voxtype/voxtype-vulkan" || true

    cp "$APPIMAGE_DIR/AppRun" "$appdir/"
    chmod 755 "$appdir/AppRun"

    populate_shared_files "$appdir"
    build_appimage "$appdir" "voxtype-${VERSION}-onnx-x86_64.AppImage"
}

# ONNX CUDA AppImage: onnx-cuda + vulkan
build_onnx_cuda() {
    echo ""
    echo "Building ONNX CUDA AppImage (onnx-cuda + vulkan)..."

    local onnx_cuda="$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-onnx-cuda"
    if [[ ! -f "$onnx_cuda" ]]; then
        echo "  Skipping: $onnx_cuda not found" >&2
        return 1
    fi

    local appdir
    appdir="$(mktemp -d "${TMPDIR:-/tmp}/voxtype-appimage.XXXXXX")"
    trap 'rm -rf "$appdir"' RETURN

    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/voxtype"

    # Multi-engine wrapper
    cp "$APPIMAGE_DIR/voxtype-onnx-wrapper.sh" "$appdir/usr/bin/voxtype"
    chmod 755 "$appdir/usr/bin/voxtype"

    # ONNX CUDA binary
    copy_binary "$onnx_cuda" "$appdir/usr/lib/voxtype/voxtype-onnx-cuda"

    # Vulkan binary for whisper engine fallback
    copy_binary "$RELEASE_DIR/voxtype-${VERSION}-linux-x86_64-vulkan" \
        "$appdir/usr/lib/voxtype/voxtype-vulkan" || true

    cp "$APPIMAGE_DIR/AppRun" "$appdir/"
    chmod 755 "$appdir/AppRun"

    populate_shared_files "$appdir"
    build_appimage "$appdir" "voxtype-${VERSION}-onnx-cuda-x86_64.AppImage"
}

# Main
echo "Building voxtype AppImage packages v${VERSION}"
echo "Release dir: $RELEASE_DIR"

failed=0

case "$VARIANT" in
    whisper)
        build_whisper || failed=1
        ;;
    onnx)
        build_onnx || failed=1
        ;;
    onnx-cuda)
        build_onnx_cuda || failed=1
        ;;
    all)
        build_whisper || failed=1
        build_onnx || failed=1
        build_onnx_cuda || failed=1
        ;;
    *)
        echo "Error: Unknown variant '$VARIANT'" >&2
        echo "Valid variants: whisper, onnx, onnx-cuda, all" >&2
        exit 1
        ;;
esac

echo ""
if [[ "$failed" -eq 0 ]]; then
    echo "AppImage builds complete."
else
    echo "Some AppImage builds were skipped (missing binaries)."
fi

# List generated AppImages
echo ""
echo "Generated AppImages:"
ls -lh "$RELEASE_DIR"/*.AppImage 2>/dev/null || echo "  (none)"
