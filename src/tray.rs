//! Linux system tray via StatusNotifierItem (SNI) D-Bus protocol.
//!
//! Uses the `ksni` crate which implements both the SNI spec and the
//! `com.canonical.dbusmenu` context menu protocol. Works with KDE Plasma,
//! GNOME (AppIndicator/KStatusNotifierItem extension), waybar, swaybar, and
//! any SNI-compatible panel.
//!
//! Enable with `[tray] enabled = true` in config (default on Linux). The
//! daemon spawns a tokio task that watches an in-process watch channel and
//! reflects state changes in the icon, tooltip, and context menu.

use crate::config::TrayIconStyle;
use ksni::{menu::*, Icon, Status, ToolTip, Tray, TrayMethods};
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};
use tokio::{process::Command as TokioCommand, sync::watch};

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
    Stopped,
}

impl TrayState {
    pub fn from_state_name(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "idle" => TrayState::Idle,
            "recording" | "streaming" => TrayState::Recording,
            "transcribing" => TrayState::Transcribing,
            "stopped" => TrayState::Stopped,
            other => {
                tracing::warn!(
                    state = other,
                    "Unrecognized daemon state; tray will show Idle"
                );
                TrayState::Idle
            }
        }
    }

    fn sni_status(self) -> Status {
        match self {
            TrayState::Recording => Status::NeedsAttention,
            TrayState::Stopped => Status::Passive,
            _ => Status::Active,
        }
    }

    fn tooltip_body(self) -> &'static str {
        match self {
            TrayState::Idle => "Ready",
            TrayState::Recording => "Recording...",
            TrayState::Transcribing => "Transcribing...",
            TrayState::Stopped => "Daemon not running",
        }
    }
}

// ── ksni::Tray implementation ─────────────────────────────────────────────────

struct VoxtypeTray {
    state: TrayState,
    /// Cached once at construction: true when started by systemd (INVOCATION_ID is set).
    under_systemd: bool,
    style: TrayIconStyle,
    levels: [f32; 8],
    phase: usize,
}

impl VoxtypeTray {
    fn new(style: TrayIconStyle) -> Self {
        Self {
            state: TrayState::Idle,
            under_systemd: std::env::var_os("INVOCATION_ID").is_some(),
            style,
            levels: [0.0; 8],
            phase: 0,
        }
    }

    fn set_state(&mut self, state: TrayState) {
        self.state = state;
        self.levels = [0.0; 8];
        self.phase = 0;
    }

    fn tick(&mut self, peak_dbfs: f32) {
        self.levels.rotate_left(1);
        // Silence (-inf) and malformed frames must never produce full bars.
        let level = if peak_dbfs.is_finite() {
            ((peak_dbfs + 60.0) / 50.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.levels[7] = level;
        self.phase = (self.phase + 1) % 8;
    }
}

/// A small transparent ARGB pixmap. SNI requires A,R,G,B bytes in network
/// order, independent of host endianness. The host scales the 24px icon.
fn waveform_icon(state: TrayState, levels: &[f32; 8], phase: usize) -> Icon {
    const SIZE: usize = 24;
    let mut pixels = vec![0u8; SIZE * SIZE * 4];
    for (bar, level) in levels.iter().enumerate() {
        let height = match state {
            TrayState::Recording => 3 + (17.0 * level.clamp(0.0, 1.0)).round() as usize,
            TrayState::Transcribing => {
                let distance = (bar + 8 - phase % 8) % 8;
                [16, 12, 8, 5, 3, 3, 3, 3][distance]
            }
            _ => 3,
        };
        let alpha = if state == TrayState::Idle { 180 } else { 255 };
        let color = [alpha, 128 + bar as u8 * 9, 134 - bar as u8 * 2, 255];
        let top = (SIZE - height) / 2;
        for y in top..top + height {
            for x in 3 * bar..3 * bar + 2 {
                let offset = (y * SIZE + x) * 4;
                pixels[offset..offset + 4].copy_from_slice(&color);
            }
        }
    }
    Icon {
        width: SIZE as i32,
        height: SIZE as i32,
        data: pixels,
    }
}

impl Tray for VoxtypeTray {
    fn id(&self) -> String {
        "voxtype".into()
    }

    fn title(&self) -> String {
        "Voxtype".into()
    }

    // Icon name used when status is Active or Transcribing.
    fn icon_name(&self) -> String {
        match self.style {
            TrayIconStyle::Microphone => "voxtype".into(),
            // Named icons take precedence over pixmaps on many SNI hosts.
            TrayIconStyle::Waveform => String::new(),
        }
    }

    fn icon_pixmap(&self) -> Vec<Icon> {
        match self.style {
            TrayIconStyle::Microphone => Vec::new(),
            TrayIconStyle::Waveform => vec![waveform_icon(self.state, &self.levels, self.phase)],
        }
    }

    // Icon name used when status is NeedsAttention (Recording).
    fn attention_icon_name(&self) -> String {
        match self.style {
            TrayIconStyle::Microphone => "voxtype-recording".into(),
            TrayIconStyle::Waveform => String::new(),
        }
    }

    fn attention_icon_pixmap(&self) -> Vec<Icon> {
        self.icon_pixmap()
    }

    fn status(&self) -> Status {
        self.state.sni_status()
    }

    fn tool_tip(&self) -> ToolTip {
        ToolTip {
            title: "Voxtype".into(),
            description: self.state.tooltip_body().into(),
            ..Default::default()
        }
    }

    /// Left-click toggles recording.
    fn activate(&mut self, _x: i32, _y: i32) {
        spawn_voxtype(&["record", "toggle"]);
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        let is_active = matches!(self.state, TrayState::Recording | TrayState::Transcribing);
        vec![
            StandardItem {
                label: "Toggle Recording".into(),
                activate: Box::new(|_| spawn_voxtype(&["record", "toggle"])),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Cancel".into(),
                enabled: is_active,
                activate: Box::new(|_| spawn_voxtype(&["record", "cancel"])),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Restart Daemon".into(),
                // Only enable when running under systemd (INVOCATION_ID is set by
                // systemd for every service it manages). On manually-launched
                // daemons, systemctl would either fail or target the wrong unit.
                enabled: self.under_systemd,
                activate: Box::new(|_| restart_daemon()),
                ..Default::default()
            }
            .into(),
        ]
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Spawn a voxtype sub-command using the current executable (unified binary).
fn spawn_voxtype(args: &[&str]) {
    let bin = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("Failed to resolve current exe for tray command: {e}");
            return;
        }
    };
    let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
    tokio::task::spawn(async move {
        match TokioCommand::new(&bin).args(&args).spawn() {
            Ok(mut child) => {
                if let Err(e) = child.wait().await {
                    tracing::warn!(bin = %bin.display(), "Failed to wait for voxtype: {e}");
                }
            }
            Err(e) => {
                tracing::warn!(bin = %bin.display(), args = ?args, "Failed to spawn voxtype: {e}");
            }
        }
    });
}

/// Restart the voxtype systemd user service. Only call when running under
/// systemd (`INVOCATION_ID` is set); the menu item is disabled otherwise.
fn restart_daemon() {
    // Fire-and-forget: systemd will SIGTERM this process before any child.wait()
    // could complete, so we just spawn and drop the child immediately.
    // kill_on_drop(false) ensures the systemctl process isn't killed when the
    // Child handle is dropped.
    match TokioCommand::new("systemctl")
        .args(["--user", "restart", "voxtype"])
        .kill_on_drop(false)
        .spawn()
    {
        Ok(_child) => {} // dropped immediately — fire and forget
        Err(e) => {
            tracing::warn!("Failed to restart voxtype via systemctl: {e}");
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Spawn the SNI tray as a tokio task. The task watches the in-process state
/// channel and reflects changes via ksni's handle. Exits when the sender is
/// dropped (daemon shutting down).
pub fn spawn(
    mut rx: watch::Receiver<TrayState>,
    style: TrayIconStyle,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let tray = VoxtypeTray::new(style);
        let handle = match tray.spawn().await {
            Ok(h) => h,
            Err(e) => {
                tracing::error!("Failed to start tray icon: {e}");
                return;
            }
        };
        tracing::info!("Tray icon active (SNI via ksni)");
        let mut ticker = time::interval(Duration::from_millis(80));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let (peak_tx, peak_rx) = watch::channel(f32::NEG_INFINITY);
        let mut reader: Option<tokio::task::JoinHandle<()>> = None;
        let mut state = *rx.borrow_and_update();
        // Apply the latest state even when it arrived during the D-Bus handshake.
        handle.update(|tray| tray.set_state(state)).await;
        if style == TrayIconStyle::Waveform && state == TrayState::Recording {
            reader = Some(spawn_peak_reader(peak_tx.clone()));
        }
        loop {
            let animated = style == TrayIconStyle::Waveform
                && matches!(state, TrayState::Recording | TrayState::Transcribing);
            tokio::select! {
                changed = rx.changed() => {
                    if changed.is_err() { break; }
                    state = *rx.borrow_and_update();
                    if let Some(task) = reader.take() { task.abort(); }
                    peak_tx.send_replace(f32::NEG_INFINITY);
                    handle.update(|tray| tray.set_state(state)).await;
                    if style == TrayIconStyle::Waveform && state == TrayState::Recording {
                        reader = Some(spawn_peak_reader(peak_tx.clone()));
                    }
                    ticker.reset();
                }
                _ = ticker.tick(), if animated => {
                    let peak = *peak_rx.borrow();
                    handle.update(|tray| tray.tick(peak)).await;
                }
            }
        }
        if let Some(task) = reader {
            task.abort();
        }
        // A closed state channel must not leave a stale "recording" icon.
        handle.shutdown().await;
    })
}

/// Reuse the OSD's best-effort audio stream, without opening another microphone
/// or depending on layer-shell. Only runs during waveform-style recording.
fn spawn_peak_reader(peak_tx: watch::Sender<f32>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let path = crate::audio::levels::default_socket_path();
        loop {
            tokio::select! {
                _ = peak_tx.closed() => break,
                _ = crate::osd::ipc::run_one_connection(&path, |frame| {
                    peak_tx.send_replace(frame.peak_dbfs);
                }) => {}
            }
            peak_tx.send_replace(f32::NEG_INFINITY);
            time::sleep(Duration::from_secs(1)).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TrayState::from_state_name ────────────────────────────────────────────

    #[test]
    fn from_str_idle() {
        assert_eq!(TrayState::from_state_name("idle"), TrayState::Idle);
    }

    #[test]
    fn from_str_recording() {
        assert_eq!(
            TrayState::from_state_name("recording"),
            TrayState::Recording
        );
    }

    #[test]
    fn from_str_transcribing() {
        assert_eq!(
            TrayState::from_state_name("transcribing"),
            TrayState::Transcribing
        );
    }

    #[test]
    fn from_str_unknown_is_idle() {
        // Unknown state names default to Idle (not Stopped) so the tray does
        // not falsely report "Daemon not running" if a new state is added.
        assert_eq!(TrayState::from_state_name("whatever"), TrayState::Idle);
        assert_eq!(TrayState::from_state_name(""), TrayState::Idle);
    }

    #[test]
    fn from_str_stopped_is_explicit() {
        assert_eq!(TrayState::from_state_name("stopped"), TrayState::Stopped);
    }

    #[test]
    fn from_str_case_insensitive() {
        assert_eq!(TrayState::from_state_name("IDLE"), TrayState::Idle);
        assert_eq!(
            TrayState::from_state_name("Recording"),
            TrayState::Recording
        );
        assert_eq!(
            TrayState::from_state_name("TRANSCRIBING"),
            TrayState::Transcribing
        );
    }

    #[test]
    fn from_str_trims_whitespace() {
        assert_eq!(TrayState::from_state_name("  idle  "), TrayState::Idle);
        assert_eq!(
            TrayState::from_state_name("recording\n"),
            TrayState::Recording
        );
    }

    // ── icon names ────────────────────────────────────────────────────────────

    #[test]
    fn idle_icon_name_matches_installed_theme_name() {
        use ksni::Tray;
        // Must match the filename stem written by setup::icons::install().
        let t = VoxtypeTray {
            state: TrayState::Idle,
            under_systemd: false,
            ..VoxtypeTray::new(TrayIconStyle::Microphone)
        };
        assert_eq!(t.icon_name(), "voxtype");
    }

    #[test]
    fn attention_icon_name_matches_installed_theme_name() {
        use ksni::Tray;
        let t = VoxtypeTray {
            state: TrayState::Idle,
            under_systemd: false,
            ..VoxtypeTray::new(TrayIconStyle::Microphone)
        };
        assert_eq!(t.attention_icon_name(), "voxtype-recording");
    }

    // ── sni_status ────────────────────────────────────────────────────────────

    #[test]
    fn sni_status_recording_is_needs_attention() {
        assert!(matches!(
            TrayState::Recording.sni_status(),
            Status::NeedsAttention
        ));
    }

    #[test]
    fn sni_status_stopped_is_passive() {
        assert!(matches!(TrayState::Stopped.sni_status(), Status::Passive));
    }

    #[test]
    fn sni_status_idle_and_transcribing_are_active() {
        assert!(matches!(TrayState::Idle.sni_status(), Status::Active));
        assert!(matches!(
            TrayState::Transcribing.sni_status(),
            Status::Active
        ));
    }

    // ── tooltip_body ──────────────────────────────────────────────────────────

    #[test]
    fn tooltip_body_non_empty_for_all_states() {
        for state in [
            TrayState::Idle,
            TrayState::Recording,
            TrayState::Transcribing,
            TrayState::Stopped,
        ] {
            assert!(
                !state.tooltip_body().is_empty(),
                "{:?} tooltip was empty",
                state
            );
        }
    }
    #[test]
    fn waveform_style_uses_pixmaps_for_both_normal_and_attention_icons() {
        let mut tray = VoxtypeTray::new(TrayIconStyle::Waveform);
        tray.set_state(TrayState::Recording);
        tray.tick(-10.0);
        assert!(tray.icon_name().is_empty());
        assert!(tray.attention_icon_name().is_empty());
        let normal = tray.icon_pixmap();
        let attention = tray.attention_icon_pixmap();
        assert_eq!(normal.len(), 1);
        assert_eq!(normal[0].data, attention[0].data);
        assert_eq!(normal[0].data.len(), 24 * 24 * 4);
    }

    #[test]
    fn louder_audio_draws_taller_bars_in_argb_network_order() {
        let quiet = waveform_icon(TrayState::Recording, &[0.0; 8], 0);
        let loud = waveform_icon(TrayState::Recording, &[1.0; 8], 0);
        let painted = |icon: &Icon| icon.data.chunks(4).filter(|p| p[0] != 0).count();
        assert!(painted(&loud) > painted(&quiet));
        let first = loud.data.chunks(4).find(|p| p[0] != 0).unwrap();
        assert_eq!(first, &[255, 128, 134, 255]);
        assert!(loud.data.chunks(4).any(|p| p == [0, 0, 0, 0]));
    }

    #[test]
    fn idle_clears_recording_history_and_stops_animation() {
        let mut tray = VoxtypeTray::new(TrayIconStyle::Waveform);
        tray.set_state(TrayState::Recording);
        tray.tick(-10.0);
        let active = tray.icon_pixmap().remove(0).data;
        tray.set_state(TrayState::Idle);
        assert_eq!(tray.levels, [0.0; 8]);
        assert_eq!(tray.phase, 0);
        assert_ne!(active, tray.icon_pixmap()[0].data);
        assert_eq!(
            waveform_icon(TrayState::Idle, &[0.0; 8], 0).data,
            waveform_icon(TrayState::Idle, &[1.0; 8], 5).data
        );
    }

    #[test]
    fn transcribing_sweeps_without_using_stale_microphone_levels() {
        let first = waveform_icon(TrayState::Transcribing, &[0.0; 8], 0);
        let next = waveform_icon(TrayState::Transcribing, &[0.0; 8], 1);
        assert_ne!(first.data, next.data);
        assert_eq!(
            first.data,
            waveform_icon(TrayState::Transcribing, &[1.0; 8], 0).data
        );
    }

    #[test]
    fn nonfinite_and_out_of_range_audio_levels_are_safe() {
        let mut tray = VoxtypeTray::new(TrayIconStyle::Waveform);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            tray.tick(invalid);
            assert_eq!(tray.levels[7], 0.0);
        }
        tray.tick(100.0);
        assert_eq!(tray.levels[7], 1.0);
        tray.tick(-200.0);
        assert_eq!(tray.levels[7], 0.0);
    }

    #[test]
    fn config_keeps_microphone_default_and_accepts_opt_in_waveform() {
        use crate::config::TrayConfig;
        let default: TrayConfig = toml::from_str("enabled = true").unwrap();
        assert_eq!(default.icon_style, TrayIconStyle::Microphone);
        let waveform: TrayConfig =
            toml::from_str("enabled = true\nicon_style = \"waveform\"").unwrap();
        assert_eq!(waveform.icon_style, TrayIconStyle::Waveform);
        let invalid = toml::from_str::<TrayConfig>("icon_style = \"other\"");
        assert!(invalid.is_err());
    }

    #[test]
    fn streaming_is_a_recording_state_for_the_waveform() {
        assert_eq!(
            TrayState::from_state_name("streaming"),
            TrayState::Recording
        );
    }
}
