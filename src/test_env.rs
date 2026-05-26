//! Shared test-only utilities for safe environment-variable mutation.
//!
//! Environment variables are process-global. `cargo test` runs tests in
//! parallel by default, so any test that calls `std::env::set_var` or
//! `std::env::remove_var` races with every other test that reads or writes
//! the environment at the same time — even if they target different keys,
//! because the underlying env table is shared.
//!
//! This module provides two primitives:
//!
//! - [`EnvGuard`] — a RAII guard that acquires the process-wide lock, sets
//!   (or removes) an env var, and restores the previous value on drop.
//!   This is the primary tool for test code that needs to temporarily change
//!   an env var.
//!
//! - [`EnvLockGuard`] — a RAII guard that acquires the process-wide lock
//!   *without* modifying any env var. Use this when you need to perform
//!   ad-hoc env reads or writes (via the `unsafe` `std::env::set_var` /
//!   `remove_var` calls) while holding the lock, for example in tests that
//!   verify [`EnvGuard`] itself.
//!
//! # Usage
//!
//! ```ignore
//! // Inside a #[test] function:
//! let _guard = EnvGuard::set("MY_VAR", "value");
//! // ... test code that reads MY_VAR ...
//! // _guard dropped here → MY_VAR restored to its prior state
//! ```
//!
//! Multiple guards on the same thread are supported — the lock is held
//! until the last guard on that thread is dropped:
//!
//! ```ignore
//! let _a = EnvGuard::set("VAR_A", "x");
//! let _b = EnvGuard::set("VAR_B", "y"); // no deadlock — reentrant
//! // ... both vars set, lock held, other threads serialised ...
//! ```
//!
//! # Why a single global lock?
//!
//! Earlier iterations used per-file mutexes (e.g. a lock local to
//! `config::tests` or `dotool::tests`). Those only serialise tests *within
//! the same module* — they cannot prevent races with tests in other modules
//! that mutate env vars at the same time, because env mutation is
//! inherently process-global. A single global lock is the minimum
//! coordination needed for correctness.
//!
//! # Lock design — reentrant per-thread
//!
//! The global env lock is a *reentrant* (recursive) mutex. A thread that
//! already holds the lock can acquire it again without deadlocking; the
//! lock is released only when every nested acquisition on that thread has
//! been released. This mirrors the semantics of POSIX
//! `PTHREAD_MUTEX_RECURSIVE`.
//!
//! Reentrancy is implemented with a `Condvar`-backed state machine:
//! `(Option<ThreadId>, depth)` tracks which thread owns the lock and how
//! many times it has acquired it. No external dependencies are required.
//!
//! # Safety of raw `std::env::set_var` / `remove_var`
//!
//! Starting with Rust edition 2024, `std::env::set_var` and
//! `std::env::remove_var` are `unsafe`. This module wraps all env mutation
//! in `unsafe` blocks, with the safety invariant that the process-wide
//! `ENV_LOCK` is held at the time of the call. Any code that calls these
//! functions outside this module **must** also hold `ENV_LOCK` (via
//! [`EnvLockGuard`]) or accept the risk of a data race.

use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

/// A reentrant (recursive) process-wide mutex for env-var access.
///
/// The same thread may acquire it multiple times. Other threads block
/// until every acquisition on the current thread has been released.
struct ReentrantEnvLock {
    inner: Mutex<(Option<ThreadId>, usize)>,
    cond: Condvar,
}

impl ReentrantEnvLock {
    const fn new() -> Self {
        Self {
            inner: Mutex::new((None, 0)),
            cond: Condvar::new(),
        }
    }

    /// Acquire the lock (blocking). Reentrant: same thread can call multiple
    /// times; the lock is held until a matching number of `release` calls.
    fn acquire(&self) {
        let tid = std::thread::current().id();
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match *state {
                (None, _) => {
                    *state = (Some(tid), 1);
                    return;
                }
                (Some(owner), count) if owner == tid => {
                    *state = (Some(owner), count + 1);
                    return;
                }
                _ => {
                    // Another thread owns the lock — wait for it to release.
                    state = self.cond.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
    }

    /// Release one acquisition. Only the owning thread may call this.
    /// When the depth reaches zero, other threads are notified.
    fn release(&self) {
        let tid = std::thread::current().id();
        let notify = {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            match *state {
                (Some(owner), count) if owner == tid => {
                    if count == 1 {
                        *state = (None, 0);
                        true // notify waiting threads after releasing inner lock
                    } else {
                        *state = (Some(owner), count - 1);
                        false
                    }
                }
                _ => panic!("ENV_LOCK released by non-owning thread"),
            }
            // `state` (MutexGuard for inner) is dropped here, releasing inner
        };
        if notify {
            self.cond.notify_all();
        }
    }
}

static ENV_LOCK: ReentrantEnvLock = ReentrantEnvLock::new();

/// RAII guard that acquires the process-wide [`ENV_LOCK`] without modifying
/// any environment variable.
///
/// Use this when you need to perform ad-hoc env reads or writes while
/// holding the global lock, for example in tests that verify [`EnvGuard`]
/// itself. The lock is released on drop.
///
/// Acquiring while [`EnvGuard`]s are active on the same thread is safe —
/// the lock is reentrant.
///
/// # Example
///
/// ```ignore
/// {
///     let _lock = EnvLockGuard::acquire();
///     // SAFETY: We hold ENV_LOCK, so no other test can race.
///     unsafe { std::env::set_var("MY_TEST_KEY", "setup_value") };
///     assert_eq!(std::env::var("MY_TEST_KEY").unwrap(), "setup_value");
///     // Clean up before releasing the lock.
///     unsafe { std::env::remove_var("MY_TEST_KEY") };
/// } // _lock dropped, lock released
/// ```
pub struct EnvLockGuard;

impl EnvLockGuard {
    /// Acquire the process-wide env lock.
    ///
    /// Blocks until the lock is available (or until the calling thread
    /// already owns it, in which case it returns immediately).
    pub fn acquire() -> Self {
        ENV_LOCK.acquire();
        Self
    }
}

impl Drop for EnvLockGuard {
    fn drop(&mut self) {
        ENV_LOCK.release();
    }
}

/// RAII guard that sets (or removes) an environment variable for the
/// duration of its lifetime, and restores the previous value on drop.
///
/// The process-wide [`ENV_LOCK`] is held for the entire lifetime of the
/// guard, preventing any other thread from mutating env vars concurrently.
/// Multiple `EnvGuard`s can be active on the same thread simultaneously —
/// the lock is reentrant; it is released only when the last guard on that
/// thread is dropped.
///
/// Create via [`EnvGuard::set`], [`EnvGuard::remove`], or
/// [`EnvGuard::set_opt`].
pub struct EnvGuard {
    key: String,
    prior: Option<String>,
}

impl EnvGuard {
    /// Set `key` to `value` for the duration of the guard.
    ///
    /// Acquires the global env lock (reentrant — safe even if the calling
    /// thread already holds it). The lock is held until the guard is dropped.
    ///
    /// The guard restores the previous value of `key` on drop.
    pub fn set(key: &str, value: &str) -> Self {
        ENV_LOCK.acquire();
        let prior = std::env::var(key).ok();
        // SAFETY: We hold the process-wide ENV_LOCK, so no other test
        // can concurrently mutate env vars through EnvGuard or EnvLockGuard.
        unsafe { std::env::set_var(key, value) };
        Self {
            key: key.to_owned(),
            prior,
        }
    }

    /// Remove `key` from the environment for the duration of the guard.
    ///
    /// Acquires the global env lock (reentrant). The lock is held until the
    /// guard is dropped. The guard restores the previous value of `key` on drop.
    pub fn remove(key: &str) -> Self {
        ENV_LOCK.acquire();
        let prior = std::env::var(key).ok();
        // SAFETY: We hold the process-wide ENV_LOCK.
        unsafe { std::env::remove_var(key) };
        Self {
            key: key.to_owned(),
            prior,
        }
    }

    /// Convenience: set `key` to `Some(value)` or remove it if `None`.
    ///
    /// This mirrors the `Option<&str>` pattern used by several existing
    /// test helpers and makes it easy to express "set or clear".
    pub fn set_opt(key: &str, value: Option<&str>) -> Self {
        match value {
            Some(v) => Self::set(key, v),
            None => Self::remove(key),
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: We hold the process-wide ENV_LOCK — acquired in the
        // constructor and released via release() below.
        match self.prior.take() {
            Some(v) => unsafe { std::env::set_var(&self.key, &v) },
            None => unsafe { std::env::remove_var(&self.key) },
        }
        ENV_LOCK.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // SAFETY RULE (regression guard): every direct `set_var` / `remove_var`
    // call in these self-tests must occur while ENV_LOCK is held — either via
    // an enclosing `EnvLockGuard` or via an `EnvGuard` constructor.  The
    // outer `EnvLockGuard` acquired at the start of each test below covers
    // precondition setup, `EnvGuard` construction, and post-drop assertions
    // as a single uninterrupted critical section, preventing races with other
    // parallel lib tests that also mutate env vars.
    //
    // Incorrect pattern (what the original tests did — do not reintroduce):
    //
    //   { let _lock = EnvLockGuard::acquire(); set_var(...); }  // lock released!
    //   { let _guard = EnvGuard::set(...); }                    // gap — race window
    //   assert!(std::env::var(...).is_err());                   // outside any lock!
    //
    // Correct pattern used throughout this module:
    //
    //   let _lock = EnvLockGuard::acquire();     // held for entire test body
    //   unsafe { set_var(...) };                 // precondition setup, under lock
    //   { let _guard = EnvGuard::set(...); ... } // reentrant acquire/release
    //   assert!(...);                            // post-drop check, still under lock

    /// Unique key prefix to avoid colliding with real env vars or with
    /// each other (tests run in parallel, all using the same process env).
    const TEST_PREFIX: &str = "VOXTYPE_TEST_ENV_GUARD_";

    #[test]
    fn set_absent_var_restores_absence_on_drop() {
        let key = format!("{TEST_PREFIX}ABSENT");
        // Hold ENV_LOCK for the entire test: precondition setup, guard
        // lifetime, and post-drop assertion are all one critical section.
        let _lock = EnvLockGuard::acquire();

        // Precondition: key is absent.
        // SAFETY: We hold ENV_LOCK via _lock.
        unsafe { std::env::remove_var(&key) };
        assert!(std::env::var(&key).is_err());

        {
            let _guard = EnvGuard::set(&key, "hello"); // reentrant lock
            assert_eq!(std::env::var(&key).unwrap(), "hello");
        } // EnvGuard dropped → var removed, lock depth decremented

        // Post-drop: var must be absent again.  Still inside _lock.
        assert!(std::env::var(&key).is_err());
    }

    #[test]
    fn set_present_var_restores_original_on_drop() {
        let key = format!("{TEST_PREFIX}PRESENT");
        let _lock = EnvLockGuard::acquire();

        // SAFETY: We hold ENV_LOCK via _lock.
        unsafe { std::env::set_var(&key, "original") };
        assert_eq!(std::env::var(&key).unwrap(), "original");

        {
            let _guard = EnvGuard::set(&key, "overridden"); // reentrant
            assert_eq!(std::env::var(&key).unwrap(), "overridden");
        } // EnvGuard dropped → var restored to "original"

        // Post-drop assertion still inside _lock.
        assert_eq!(std::env::var(&key).unwrap(), "original");

        // Cleanup while still holding the lock.
        // SAFETY: We hold ENV_LOCK.
        unsafe { std::env::remove_var(&key) };
    }

    #[test]
    fn remove_present_var_restores_original_on_drop() {
        let key = format!("{TEST_PREFIX}TO_REMOVE");
        let _lock = EnvLockGuard::acquire();

        // SAFETY: We hold ENV_LOCK via _lock.
        unsafe { std::env::set_var(&key, "will_be_removed") };
        assert_eq!(std::env::var(&key).unwrap(), "will_be_removed");

        {
            let _guard = EnvGuard::remove(&key); // reentrant
            assert!(std::env::var(&key).is_err());
        } // EnvGuard dropped → var restored to "will_be_removed"

        // Post-drop assertion still inside _lock.
        assert_eq!(std::env::var(&key).unwrap(), "will_be_removed");

        // SAFETY: We hold ENV_LOCK.
        unsafe { std::env::remove_var(&key) };
    }

    #[test]
    fn remove_absent_var_stays_absent_on_drop() {
        let key = format!("{TEST_PREFIX}ALREADY_ABSENT");
        let _lock = EnvLockGuard::acquire();

        // SAFETY: We hold ENV_LOCK via _lock.
        unsafe { std::env::remove_var(&key) };
        assert!(std::env::var(&key).is_err());

        {
            let _guard = EnvGuard::remove(&key); // reentrant
            assert!(std::env::var(&key).is_err());
        } // EnvGuard dropped → var stays absent

        // Post-drop assertion still inside _lock.
        assert!(std::env::var(&key).is_err());
    }

    #[test]
    fn set_opt_some_sets_and_restores() {
        let key = format!("{TEST_PREFIX}OPT_SOME");
        let _lock = EnvLockGuard::acquire();

        // SAFETY: We hold ENV_LOCK via _lock.
        unsafe { std::env::remove_var(&key) };

        {
            let _guard = EnvGuard::set_opt(&key, Some("value")); // reentrant
            assert_eq!(std::env::var(&key).unwrap(), "value");
        }

        // Post-drop assertion still inside _lock.
        assert!(std::env::var(&key).is_err());
    }

    #[test]
    fn set_opt_none_removes_and_restores() {
        let key = format!("{TEST_PREFIX}OPT_NONE");
        let _lock = EnvLockGuard::acquire();

        // SAFETY: We hold ENV_LOCK via _lock.
        unsafe { std::env::set_var(&key, "prior") };

        {
            let _guard = EnvGuard::set_opt(&key, None); // reentrant
            assert!(std::env::var(&key).is_err());
        }

        // Post-drop assertion still inside _lock.
        assert_eq!(std::env::var(&key).unwrap(), "prior");

        // SAFETY: We hold ENV_LOCK.
        unsafe { std::env::remove_var(&key) };
    }

    #[test]
    fn env_lock_guard_acquires_and_releases() {
        // Verify that EnvLockGuard can be acquired and dropped without
        // deadlocking, and that env vars can be read while held.
        let _lock = EnvLockGuard::acquire();
        // Reading env vars under the lock is always safe.
        assert!(std::env::var("PATH").is_ok() || std::env::var("PATH").is_err());
        // Lock is released when _lock is dropped at end of scope.
    }

    #[test]
    fn multiple_guards_on_same_thread_no_deadlock() {
        // Verify that multiple EnvGuard instances can coexist on the same
        // thread without deadlocking — the reentrant lock must allow this.
        let key_a = format!("{TEST_PREFIX}MULTI_A");
        let key_b = format!("{TEST_PREFIX}MULTI_B");

        let _a = EnvGuard::set(&key_a, "alpha");
        let _b = EnvGuard::set(&key_b, "beta"); // reentrant — must not deadlock
        assert_eq!(std::env::var(&key_a).unwrap(), "alpha");
        assert_eq!(std::env::var(&key_b).unwrap(), "beta");
        // Both restored on drop (in reverse declaration order).
    }
}
