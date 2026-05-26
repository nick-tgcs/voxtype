#!/bin/bash
# Tests for the appimagetool verification and selection logic in scripts/build-appimage.sh
#
# Run with:  bash tests/test-build-appimage-verify.sh
#
# Returns exit code 0 if all tests pass, 1 if any fail.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/../scripts/build-appimage.sh"

# Source the script in function-only mode (guard added to the main script).
# This loads verify_download_sha256 and the constants without running the build.
APPIMAGETOOL_SOURCE_ONLY=1 VERSION=0 source "$BUILD_SCRIPT"

PASS=0
FAIL=0
ORIGINAL_PATH="$PATH"

pass() { echo "PASS: $1"; ((PASS++)) || true; }
fail() { echo "FAIL: $1"; ((FAIL++)) || true; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

make_temp_file() {
    local f
    f=$(mktemp)
    # Write unique content so the hash is non-trivial
    printf 'voxtype test content %s %s\n' "$$" "$RANDOM" > "$f"
    echo "$f"
}

sha256_of() {
    sha256sum "$1" | cut -d' ' -f1
}

make_test_appimagetool() {
    local path="$1" label="$2"
    printf '#!/bin/sh\necho "%s"\n' "$label" > "$path"
    chmod +x "$path"
}

# ---------------------------------------------------------------------------
# verify_download_sha256 tests
# ---------------------------------------------------------------------------

test_correct_hash_passes() {
    local f
    f=$(make_temp_file)
    local hash
    hash=$(sha256_of "$f")
    if verify_download_sha256 "$f" "$hash" 2>/dev/null; then
        pass "correct hash returns 0"
    else
        fail "correct hash returns 0"
    fi
    rm -f "$f"
}

test_wrong_hash_fails() {
    local f
    f=$(make_temp_file)
    local bad_hash="aaaa0000bbbb1111cccc2222dddd3333eeee4444ffff5555aaaa0000bbbb1111"
    if ! verify_download_sha256 "$f" "$bad_hash" 2>/dev/null; then
        pass "wrong hash returns 1"
    else
        fail "wrong hash returns 1"
        rm -f "$f"
    fi
}

test_file_deleted_on_mismatch() {
    local f
    f=$(make_temp_file)
    local bad_hash="aaaa0000bbbb1111cccc2222dddd3333eeee4444ffff5555aaaa0000bbbb1111"
    verify_download_sha256 "$f" "$bad_hash" 2>/dev/null || true
    if [[ ! -f "$f" ]]; then
        pass "file is deleted on hash mismatch"
    else
        fail "file is deleted on hash mismatch"
        rm -f "$f"
    fi
}

test_missing_file_fails() {
    local f="/tmp/voxtype_test_nonexistent_$$.bin"
    local any_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    if ! verify_download_sha256 "$f" "$any_hash" 2>/dev/null; then
        pass "missing file returns 1"
    else
        fail "missing file returns 1"
    fi
}

test_empty_hash_fails() {
    local f
    f=$(make_temp_file)
    if ! verify_download_sha256 "$f" "" 2>/dev/null; then
        pass "empty expected hash returns 1"
    else
        fail "empty expected hash returns 1"
    fi
    rm -f "$f"
}

test_uppercase_hash_accepted() {
    local f
    f=$(make_temp_file)
    local hash_upper
    hash_upper=$(sha256_of "$f" | tr '[:lower:]' '[:upper:]')
    if verify_download_sha256 "$f" "$hash_upper" 2>/dev/null; then
        pass "uppercase expected hash is accepted"
    else
        fail "uppercase expected hash is accepted"
    fi
    rm -f "$f"
}

test_mixed_case_hash_accepted() {
    local f
    f=$(make_temp_file)
    local hash
    hash=$(sha256_of "$f")
    # Flip every other character to uppercase
    local mixed
    mixed=$(echo "$hash" | sed 's/./\u&/g; s/../\l&/2g; s/..../\l&/3g')
    # Simpler: just uppercase first half
    local first_half second_half
    first_half=$(echo "${hash:0:32}" | tr '[:lower:]' '[:upper:]')
    second_half="${hash:32}"
    if verify_download_sha256 "$f" "${first_half}${second_half}" 2>/dev/null; then
        pass "mixed-case expected hash is accepted"
    else
        fail "mixed-case expected hash is accepted"
    fi
    rm -f "$f"
}

test_error_output_shows_computed_hash() {
    local f
    f=$(make_temp_file)
    local actual_hash
    actual_hash=$(sha256_of "$f")
    local bad_hash="aaaa0000bbbb1111cccc2222dddd3333eeee4444ffff5555aaaa0000bbbb1111"
    local output
    output=$(verify_download_sha256 "$f" "$bad_hash" 2>&1 || true)
    if echo "$output" | grep -qF "$actual_hash"; then
        pass "error output includes the computed hash"
    else
        fail "error output includes the computed hash (got: $output)"
    fi
}

test_error_output_shows_skip_hint() {
    local f
    f=$(make_temp_file)
    local bad_hash="aaaa0000bbbb1111cccc2222dddd3333eeee4444ffff5555aaaa0000bbbb1111"
    local output
    output=$(verify_download_sha256 "$f" "$bad_hash" 2>&1 || true)
    if echo "$output" | grep -q "APPIMAGETOOL_SKIP_VERIFY"; then
        pass "error output mentions APPIMAGETOOL_SKIP_VERIFY escape hatch"
    else
        fail "error output mentions APPIMAGETOOL_SKIP_VERIFY escape hatch (got: $output)"
    fi
}

test_pinned_version_constant_is_set() {
    if [[ -n "${APPIMAGETOOL_VERSION:-}" ]]; then
        pass "APPIMAGETOOL_VERSION constant is set (got: $APPIMAGETOOL_VERSION)"
    else
        fail "APPIMAGETOOL_VERSION constant is set"
    fi
}

test_pinned_sha256_constant_is_64_hex_chars() {
    local hash="${APPIMAGETOOL_SHA256:-}"
    if [[ ${#hash} -eq 64 ]] && [[ "$hash" =~ ^[0-9a-fA-F]+$ ]]; then
        pass "APPIMAGETOOL_SHA256 constant is a valid 64-char hex string"
    else
        fail "APPIMAGETOOL_SHA256 constant is a valid 64-char hex string (got: '${hash}')"
    fi
}

test_pinned_url_contains_version() {
    local url="${APPIMAGETOOL_URL:-}"
    local ver="${APPIMAGETOOL_VERSION:-}"
    if [[ "$url" == *"$ver"* ]]; then
        pass "APPIMAGETOOL_URL contains the pinned version"
    else
        fail "APPIMAGETOOL_URL contains the pinned version (url: $url, ver: $ver)"
    fi
}

# ---------------------------------------------------------------------------
# find_appimagetool selection tests
# ---------------------------------------------------------------------------

test_default_prefers_verified_cached_pinned_binary_over_path() {
    local home bin_dir cached path_tool cached_hash resolved
    home=$(mktemp -d)
    bin_dir="$home/test-bin"
    cached="$home/.local/bin/appimagetool-${APPIMAGETOOL_VERSION}"
    path_tool="$bin_dir/appimagetool"

    mkdir -p "$home/.local/bin" "$bin_dir"
    make_test_appimagetool "$cached" "cached"
    make_test_appimagetool "$path_tool" "path"
    cached_hash=$(sha256_of "$cached")

    if resolved=$({
        export HOME="$home"
        export PATH="$bin_dir:$ORIGINAL_PATH"
        APPIMAGETOOL_SHA256="$cached_hash"
        unset APPIMAGETOOL_ALLOW_SYSTEM APPIMAGETOOL_SKIP_VERIFY
        find_appimagetool
    } 2>/dev/null); then
        if [[ "$resolved" == "$cached" ]]; then
            pass "default selection ignores PATH and uses the verified pinned cache"
        else
            fail "default selection ignores PATH and uses the verified pinned cache (got: $resolved)"
        fi
    else
        fail "default selection ignores PATH and uses the verified pinned cache"
    fi

    rm -rf "$home"
}

test_downloads_and_caches_pinned_appimagetool() {
    local home source cached source_hash resolved
    home=$(mktemp -d)
    source="$home/source-appimagetool"
    cached="$home/.local/bin/appimagetool-${APPIMAGETOOL_VERSION}"

    mkdir -p "$home/.local/bin"
    make_test_appimagetool "$source" "downloaded"
    source_hash=$(sha256_of "$source")

    if resolved=$({
        export HOME="$home"
        export PATH="$ORIGINAL_PATH"
        APPIMAGETOOL_URL="file://$source"
        APPIMAGETOOL_SHA256="$source_hash"
        unset APPIMAGETOOL_ALLOW_SYSTEM APPIMAGETOOL_SKIP_VERIFY
        find_appimagetool
    } 2>/dev/null); then
        if [[ "$resolved" == "$cached" ]] && [[ -x "$cached" ]] && cmp -s "$source" "$cached"; then
            pass "missing cache downloads, verifies, and caches the pinned appimagetool"
        else
            fail "missing cache downloads, verifies, and caches the pinned appimagetool"
        fi
    else
        fail "missing cache downloads, verifies, and caches the pinned appimagetool"
    fi

    rm -rf "$home"
}

test_download_failure_does_not_silently_use_path_binary() {
    local home bin_dir path_tool stderr_path resolved
    home=$(mktemp -d)
    bin_dir="$home/test-bin"
    path_tool="$bin_dir/appimagetool"
    stderr_path=$(mktemp)

    mkdir -p "$home/.local/bin" "$bin_dir"
    make_test_appimagetool "$path_tool" "path"

    if resolved=$({
        export HOME="$home"
        export PATH="$bin_dir:$ORIGINAL_PATH"
        APPIMAGETOOL_URL="file://$home/does-not-exist"
        unset APPIMAGETOOL_ALLOW_SYSTEM APPIMAGETOOL_SKIP_VERIFY
        find_appimagetool
    } 2>"$stderr_path"); then
        fail "download failure without opt-in does not silently use PATH appimagetool (got: $resolved)"
    else
        pass "download failure without opt-in does not silently use PATH appimagetool"
    fi

    rm -rf "$home" "$stderr_path"
}

test_system_fallback_is_opt_in_and_warns() {
    local home bin_dir path_tool stderr_path resolved
    home=$(mktemp -d)
    bin_dir="$home/test-bin"
    path_tool="$bin_dir/appimagetool"
    stderr_path=$(mktemp)

    mkdir -p "$home/.local/bin" "$bin_dir"
    make_test_appimagetool "$path_tool" "path"

    if resolved=$({
        export HOME="$home"
        export PATH="$bin_dir:$ORIGINAL_PATH"
        APPIMAGETOOL_URL="file://$home/does-not-exist"
        APPIMAGETOOL_ALLOW_SYSTEM=1
        unset APPIMAGETOOL_SKIP_VERIFY
        find_appimagetool
    } 2>"$stderr_path"); then
        if [[ "$resolved" == "appimagetool" ]] \
            && grep -q "APPIMAGETOOL_ALLOW_SYSTEM" "$stderr_path" \
            && grep -q "bypasses the pinned version and SHA-256 verification" "$stderr_path"; then
            pass "system PATH fallback is explicit and warns about the trust tradeoff"
        else
            fail "system PATH fallback is explicit and warns about the trust tradeoff"
        fi
    else
        fail "system PATH fallback is explicit and warns about the trust tradeoff"
    fi

    rm -rf "$home" "$stderr_path"
}

# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

echo "Running build-appimage verification tests..."
echo ""

test_correct_hash_passes
test_wrong_hash_fails
test_file_deleted_on_mismatch
test_missing_file_fails
test_empty_hash_fails
test_uppercase_hash_accepted
test_mixed_case_hash_accepted
test_error_output_shows_computed_hash
test_error_output_shows_skip_hint
test_pinned_version_constant_is_set
test_pinned_sha256_constant_is_64_hex_chars
test_pinned_url_contains_version
test_default_prefers_verified_cached_pinned_binary_over_path
test_downloads_and_caches_pinned_appimagetool
test_download_failure_does_not_silently_use_path_binary
test_system_fallback_is_opt_in_and_warns

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
