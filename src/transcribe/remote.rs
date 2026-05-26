//! Remote speech-to-text transcription via OpenAI-compatible API
//!
//! Sends audio to a remote whisper.cpp server or OpenAI-compatible endpoint
//! for transcription, enabling use of GPU servers for faster inference.
//!
//! Note: Remote APIs don't support language arrays. When a language array is
//! configured, the first/primary language is used.

use super::Transcriber;
use crate::config::{LanguageConfig, WhisperConfig};
use crate::error::TranscribeError;
use std::io::Cursor;
use std::time::Duration;
use ureq::serde_json;

/// Remote transcriber using OpenAI-compatible Whisper API
#[derive(Debug)]
pub struct RemoteTranscriber {
    /// Base endpoint URL (e.g., "http://192.168.1.100:8080")
    endpoint: String,
    /// Model name to send to server
    model: String,
    /// Language configuration
    language: LanguageConfig,
    /// Whether to translate to English
    translate: bool,
    /// Optional API key for authentication
    api_key: Option<String>,
    /// Optional initial prompt for transcription context
    initial_prompt: Option<String>,
    /// Request timeout
    timeout: Duration,
}

/// Parse a URL and check whether its host component is a loopback address.
///
/// Uses the `url` crate to correctly extract the hostname, which prevents
/// userinfo-based bypasses (e.g. `http://localhost@evil.com` where the real
/// host is `evil.com`, not `localhost`).
///
/// Returns `Ok(true)` for `localhost`, any address in 127.0.0.0/8, and `::1`.
/// Returns `Ok(false)` for non-loopback hosts.
/// Returns `Err` for malformed URLs (fails closed).
fn host_is_loopback(url_str: &str) -> Result<bool, TranscribeError> {
    let parsed = url::Url::parse(url_str).map_err(|e| {
        TranscribeError::ConfigError(format!(
            "remote_endpoint is not a valid URL: {} (error: {})",
            url_str, e
        ))
    })?;

    let host = parsed.host().ok_or_else(|| {
        TranscribeError::ConfigError(format!(
            "remote_endpoint has no host: {}",
            url_str
        ))
    })?;

    match host {
        url::Host::Domain(domain) => {
            // Exact match for "localhost" (case-insensitive)
            Ok(domain.eq_ignore_ascii_case("localhost"))
        }
        url::Host::Ipv4(addr) => Ok(addr.is_loopback()),
        url::Host::Ipv6(addr) => Ok(addr.is_loopback()),
    }
}

impl RemoteTranscriber {
    /// Create a new remote transcriber from config
    pub fn new(config: &WhisperConfig) -> Result<Self, TranscribeError> {
        let endpoint = config
            .remote_endpoint
            .as_ref()
            .ok_or_else(|| {
                TranscribeError::ConfigError(
                    "remote_endpoint is required when mode = 'remote'".into(),
                )
            })?
            .clone();

        // Validate endpoint URL format
        if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            return Err(TranscribeError::ConfigError(format!(
                "remote_endpoint must start with http:// or https://, got: {}",
                endpoint
            )));
        }

        // Reject non-loopback http:// unless explicitly opted in.
        // http:// to loopback addresses (127.x.x.x / ::1 / localhost) is fine —
        // traffic never leaves the machine. Non-loopback http:// sends audio
        // (and any API key) in cleartext.
        if endpoint.starts_with("http://") {
            match host_is_loopback(&endpoint)? {
                true => { /* loopback — allowed without opt-in */ }
                false => {
                    if config.remote_allow_insecure_http {
                        tracing::warn!(
                            "Remote endpoint uses HTTP over a non-loopback address. \
                             Audio and API keys will be transmitted unencrypted. \
                             Proceeding because remote_allow_insecure_http = true."
                        );
                    } else {
                        return Err(TranscribeError::ConfigError(format!(
                            "Remote endpoint '{}' uses HTTP over a non-loopback address. \
                             Audio data would be transmitted unencrypted.\n\
                             Use https:// instead, or set remote_allow_insecure_http = true \
                             in your config (or --allow-insecure-http on the command line) \
                             to accept this risk.",
                            endpoint
                        )));
                    }
                }
            }
        }

        // Check for API key in config or environment
        let api_key = config
            .remote_api_key
            .clone()
            .or_else(|| std::env::var("VOXTYPE_WHISPER_API_KEY").ok());

        let model = config
            .remote_model
            .clone()
            .unwrap_or_else(|| "whisper-1".to_string());

        let timeout = Duration::from_secs(config.remote_timeout_secs.unwrap_or(30));

        // Warn if language array is configured (remote APIs don't support arrays)
        if config.language.is_multiple() {
            tracing::warn!(
                "Remote backend doesn't support language arrays. Using primary language '{}' from {:?}",
                config.language.primary(),
                config.language.as_vec()
            );
        }

        tracing::info!(
            "Configured remote transcriber: endpoint={}, model={}, timeout={}s",
            endpoint,
            model,
            timeout.as_secs()
        );

        let initial_prompt = config
            .initial_prompt
            .as_ref()
            .filter(|s| !s.is_empty())
            .cloned();

        Ok(Self {
            endpoint,
            model,
            language: config.language.clone(),
            translate: config.translate,
            api_key,
            initial_prompt,
            timeout,
        })
    }

    /// Encode f32 samples to WAV format
    fn encode_wav(&self, samples: &[f32]) -> Result<Vec<u8>, TranscribeError> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut buffer, spec).map_err(|e| {
            TranscribeError::AudioFormat(format!("Failed to create WAV writer: {}", e))
        })?;

        // Convert f32 [-1.0, 1.0] to i16
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let scaled = (clamped * i16::MAX as f32) as i16;
            writer.write_sample(scaled).map_err(|e| {
                TranscribeError::AudioFormat(format!("Failed to write sample: {}", e))
            })?;
        }

        writer
            .finalize()
            .map_err(|e| TranscribeError::AudioFormat(format!("Failed to finalize WAV: {}", e)))?;

        Ok(buffer.into_inner())
    }

    /// Build the multipart form body for the API request
    fn build_multipart_body(&self, wav_data: &[u8]) -> (String, Vec<u8>) {
        let boundary = format!(
            "----VoxtypeBoundary{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        let mut body = Vec::new();

        // Add file field
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(wav_data);
        body.extend_from_slice(b"\r\n");

        // Add model field
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        body.extend_from_slice(self.model.as_bytes());
        body.extend_from_slice(b"\r\n");

        // Add language field (if not auto-detect mode)
        // For language arrays, use the primary language since remote APIs don't support arrays
        if !self.language.is_auto() {
            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
            body.extend_from_slice(self.language.primary().as_bytes());
            body.extend_from_slice(b"\r\n");
        }

        // Add prompt field (if initial_prompt is configured)
        if let Some(ref prompt) = self.initial_prompt {
            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"prompt\"\r\n\r\n");
            body.extend_from_slice(prompt.as_bytes());
            body.extend_from_slice(b"\r\n");
        }

        // Add response_format field
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
        body.extend_from_slice(b"json\r\n");

        // End boundary
        body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

        (boundary, body)
    }
}

impl Transcriber for RemoteTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat("Empty audio buffer".into()));
        }

        let duration_secs = samples.len() as f32 / 16000.0;
        tracing::debug!(
            "Sending {:.2}s of audio to remote server ({} samples)",
            duration_secs,
            samples.len()
        );

        let start = std::time::Instant::now();

        // Encode audio to WAV
        let wav_data = self.encode_wav(samples)?;
        tracing::debug!("Encoded WAV: {} bytes", wav_data.len());

        // Build multipart form
        let (boundary, body) = self.build_multipart_body(&wav_data);

        // Determine the API path based on whether we're doing transcription or translation
        let path = if self.translate {
            "/v1/audio/translations"
        } else {
            "/v1/audio/transcriptions"
        };

        let url = format!("{}{}", self.endpoint.trim_end_matches('/'), path);

        // Build request
        let mut request = ureq::post(&url).timeout(self.timeout).set(
            "Content-Type",
            &format!("multipart/form-data; boundary={}", boundary),
        );

        // Add authorization if API key is configured
        if let Some(ref key) = self.api_key {
            request = request.set("Authorization", &format!("Bearer {}", key));
        }

        // Send request
        let response = request.send_bytes(&body).map_err(|e| match e {
            ureq::Error::Status(code, resp) => {
                let body = resp.into_string().unwrap_or_default();
                TranscribeError::RemoteError(format!("Server returned {}: {}", code, body))
            }
            ureq::Error::Transport(t) => {
                TranscribeError::NetworkError(format!("Request failed: {}", t))
            }
        })?;

        // Parse JSON response
        let json: serde_json::Value = response.into_json().map_err(|e| {
            TranscribeError::RemoteError(format!("Failed to parse response: {}", e))
        })?;

        // Extract text from response
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TranscribeError::RemoteError(format!("Response missing 'text' field: {}", json))
            })?
            .trim()
            .to_string();

        tracing::info!(
            "Remote transcription completed in {:.2}s: {:?}",
            start.elapsed().as_secs_f32(),
            if text.chars().count() > 50 {
                format!("{}...", text.chars().take(50).collect::<String>())
            } else {
                text.clone()
            }
        );

        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_wav_basic() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        // Create a simple sine wave
        let samples: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin() * 0.5)
            .collect();

        let wav = transcriber.encode_wav(&samples).unwrap();

        // WAV header is 44 bytes, then 16000 samples * 2 bytes = 32000 bytes
        assert_eq!(wav.len(), 44 + 32000);

        // Check WAV magic
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    fn test_config_validation_missing_endpoint() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: None, // Missing!
            ..Default::default()
        };

        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("remote_endpoint"));
    }

    #[test]
    fn test_config_validation_invalid_url() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("not-a-url".to_string()),
            ..Default::default()
        };

        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("http://"));
    }

    #[test]
    fn test_multipart_body_structure() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_model: Some("large-v3".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100]; // Dummy data

        let (boundary, body) = transcriber.build_multipart_body(&wav_data);

        let body_str = String::from_utf8_lossy(&body);

        // Verify boundary is used
        assert!(body_str.contains(&boundary));

        // Verify required fields
        assert!(body_str.contains("name=\"file\""));
        assert!(body_str.contains("filename=\"audio.wav\""));
        assert!(body_str.contains("name=\"model\""));
        assert!(body_str.contains("large-v3"));
        assert!(body_str.contains("name=\"language\""));
        assert!(body_str.contains("name=\"response_format\""));
        assert!(body_str.contains("json"));
    }

    #[test]
    fn test_multipart_body_includes_prompt() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            initial_prompt: Some("Technical discussion about Rust and Kubernetes.".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (_boundary, body) = transcriber.build_multipart_body(&wav_data);
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("name=\"prompt\""));
        assert!(body_str.contains("Technical discussion about Rust and Kubernetes."));
    }

    #[test]
    fn test_multipart_body_excludes_empty_prompt() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            initial_prompt: Some("".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (_boundary, body) = transcriber.build_multipart_body(&wav_data);
        let body_str = String::from_utf8_lossy(&body);

        assert!(!body_str.contains("name=\"prompt\""));
    }

    #[test]
    fn test_multipart_body_excludes_prompt_when_none() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            initial_prompt: None,
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (_boundary, body) = transcriber.build_multipart_body(&wav_data);
        let body_str = String::from_utf8_lossy(&body);

        assert!(!body_str.contains("name=\"prompt\""));
    }

    #[test]
    fn test_translate_false_uses_transcriptions_endpoint() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            translate: false,
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        // Verify translate flag is stored correctly
        assert!(!transcriber.translate);

        // The endpoint path logic: if !translate, use /v1/audio/transcriptions
        let path = if transcriber.translate {
            "/v1/audio/translations"
        } else {
            "/v1/audio/transcriptions"
        };
        assert_eq!(path, "/v1/audio/transcriptions");
    }

    #[test]
    fn test_translate_true_uses_translations_endpoint() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            translate: true,
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        // Verify translate flag is stored correctly
        assert!(transcriber.translate);

        // The endpoint path logic: if translate, use /v1/audio/translations
        let path = if transcriber.translate {
            "/v1/audio/translations"
        } else {
            "/v1/audio/transcriptions"
        };
        assert_eq!(path, "/v1/audio/translations");
    }

    #[test]
    fn test_api_key_from_config() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_api_key: Some("sk-test-key-123".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.api_key, Some("sk-test-key-123".to_string()));
    }

    #[test]
    fn test_custom_timeout() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_timeout_secs: Some(60),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_default_timeout() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.timeout, Duration::from_secs(30));
    }

    // --- host_is_loopback unit tests ---

    #[test]
    fn test_loopback_localhost() {
        assert!(host_is_loopback("http://localhost:8080").unwrap());
        assert!(host_is_loopback("http://localhost").unwrap());
        assert!(host_is_loopback("http://LOCALHOST:8080").unwrap());
        assert!(host_is_loopback("https://localhost/v1/audio").unwrap());
    }

    #[test]
    fn test_loopback_ipv4() {
        assert!(host_is_loopback("http://127.0.0.1:8080").unwrap());
        assert!(host_is_loopback("http://127.0.0.2:9000").unwrap());
        assert!(host_is_loopback("http://127.1.2.3").unwrap());
    }

    #[test]
    fn test_loopback_ipv6() {
        assert!(host_is_loopback("http://[::1]:8080").unwrap());
        assert!(host_is_loopback("http://[::1]").unwrap());
    }

    #[test]
    fn test_not_loopback_remote_ip() {
        assert!(!host_is_loopback("http://192.168.1.100:8080").unwrap());
        assert!(!host_is_loopback("http://10.0.0.1").unwrap());
        assert!(!host_is_loopback("http://8.8.8.8").unwrap());
    }

    #[test]
    fn test_not_loopback_hostname() {
        assert!(!host_is_loopback("http://example.com").unwrap());
        assert!(!host_is_loopback("http://example.com:8080").unwrap());
    }

    #[test]
    fn test_not_loopback_localhost_subdomain() {
        // "localhost" must match the whole host, not as a substring
        assert!(!host_is_loopback("http://localhost.example.com:8080").unwrap());
        assert!(!host_is_loopback("http://notlocalhost.com").unwrap());
    }

    // --- userinfo bypass regression tests ---

    #[test]
    fn test_userinfo_bypass_localhost_at_evil() {
        // http://localhost@evil.com — real host is evil.com, not localhost
        assert!(!host_is_loopback("http://localhost@evil.com").unwrap());
    }

    #[test]
    fn test_userinfo_bypass_localhost_port_at_evil() {
        // http://localhost:8080@evil.com — "localhost:8080" is userinfo, host is evil.com
        assert!(!host_is_loopback("http://localhost:8080@evil.com").unwrap());
    }

    #[test]
    fn test_userinfo_bypass_127_at_evil() {
        // http://127.0.0.1@evil.com — real host is evil.com
        assert!(!host_is_loopback("http://127.0.0.1@evil.com").unwrap());
    }

    #[test]
    fn test_userinfo_bypass_127_port_at_evil() {
        // http://127.0.0.1:9000@evil.com — real host is evil.com
        assert!(!host_is_loopback("http://127.0.0.1:9000@evil.com").unwrap());
    }

    // --- malformed URL tests ---

    #[test]
    fn test_malformed_url_fails_closed() {
        // A URL with no host should fail (not default to loopback)
        assert!(host_is_loopback("http://").is_err());
        assert!(host_is_loopback("not-a-url").is_err());
    }

    // --- non-loopback HTTP rejection tests ---

    #[test]
    fn test_non_loopback_http_rejected_by_default() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://192.168.1.100:8080".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("remote_allow_insecure_http"),
            "error should mention the opt-in field: {msg}"
        );
    }

    #[test]
    fn test_non_loopback_http_allowed_with_flag() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://192.168.1.100:8080".to_string()),
            remote_allow_insecure_http: true,
            ..Default::default()
        };
        // Should not return an error
        assert!(RemoteTranscriber::new(&config).is_ok());
    }

    #[test]
    fn test_https_non_loopback_always_allowed() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://api.openai.com".to_string()),
            ..Default::default()
        };
        assert!(RemoteTranscriber::new(&config).is_ok());
    }

    #[test]
    fn test_localhost_http_always_allowed() {
        // Loopback http:// must never require the opt-in flag
        for url in &[
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
        ] {
            let config = WhisperConfig {
                mode: Some(crate::config::WhisperMode::Remote),
                remote_endpoint: Some(url.to_string()),
                remote_allow_insecure_http: false,
                ..Default::default()
            };
            assert!(
                RemoteTranscriber::new(&config).is_ok(),
                "loopback URL should be accepted without flag: {url}"
            );
        }
    }

    // --- integration-style tests for userinfo bypass ---

    #[test]
    fn test_userinfo_bypass_rejected_by_default() {
        // http://localhost@evil.com must be treated as non-loopback
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost@evil.com".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err(), "userinfo bypass should be rejected by default");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("remote_allow_insecure_http"),
            "error should mention the opt-in field: {msg}"
        );
    }

    #[test]
    fn test_userinfo_bypass_with_port_rejected_by_default() {
        // http://localhost:8080@evil.com must be treated as non-loopback
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080@evil.com".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err(), "userinfo bypass with port should be rejected by default");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("remote_allow_insecure_http"),
            "error should mention the opt-in field: {msg}"
        );
    }

    #[test]
    fn test_userinfo_bypass_allowed_with_flag() {
        // http://localhost@evil.com with opt-in should succeed
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost@evil.com".to_string()),
            remote_allow_insecure_http: true,
            ..Default::default()
        };
        assert!(
            RemoteTranscriber::new(&config).is_ok(),
            "userinfo bypass should be allowed with opt-in flag"
        );
    }

    #[test]
    fn test_real_loopback_http_remains_allowed() {
        // http://localhost:8080 (genuine loopback) must still work without the flag
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        assert!(
            RemoteTranscriber::new(&config).is_ok(),
            "real loopback should work without flag"
        );
    }

    #[test]
    fn test_real_loopback_ip_remains_allowed() {
        // http://127.0.0.1:8080 (genuine loopback) must still work without the flag
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://127.0.0.1:8080".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        assert!(
            RemoteTranscriber::new(&config).is_ok(),
            "real loopback IP should work without flag"
        );
    }

    #[test]
    fn test_non_loopback_http_remains_rejected() {
        // http://192.168.1.100:8080 must still be rejected by default
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://192.168.1.100:8080".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        assert!(RemoteTranscriber::new(&config).is_err());
    }

    #[test]
    fn test_https_non_loopback_remains_allowed() {
        // https://example.com must still be allowed without the flag
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://example.com".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        assert!(RemoteTranscriber::new(&config).is_ok());
    }

    #[test]
    fn test_malformed_endpoint_config_error() {
        // A malformed URL should produce a clear config error, not be silently accepted
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("not-a-url".to_string()),
            remote_allow_insecure_http: false,
            ..Default::default()
        };
        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        // The error should mention the URL is invalid, not the insecure HTTP opt-in
        assert!(
            msg.contains("http://") || msg.contains("valid URL") || msg.contains("not a valid"),
            "malformed URL error should describe the problem clearly: {msg}"
        );
    }
}
