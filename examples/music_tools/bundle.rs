//! Local listening bundles; no network, model weights, latents or arbitrary files.
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use yue2_rs::{
    Result, audio,
    json::{self, Value},
};

pub fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
fn credentials(value: &Value) -> bool {
    match value {
        Value::Object(fields) => fields.iter().any(|(k, v)| {
            let key = k.to_ascii_lowercase().replace('-', "_");
            ([
                "token",
                "access_token",
                "refresh_token",
                "api_key",
                "apikey",
                "hf_token",
                "password",
                "passwd",
                "secret",
                "client_secret",
                "authorization",
                "cookie",
                "credentials",
                "credential",
                "aws_secret_access_key",
                "aws_session_token",
            ]
            .contains(&key.as_str())
                && !matches!(v, Value::Null | Value::Bool(false))
                && !matches!(v,Value::String(s) if s.is_empty()))
                || credentials(v)
        }),
        Value::Array(values) => values.iter().any(credentials),
        Value::String(s) => {
            let words: Vec<_> = s.split_whitespace().collect();
            words
                .windows(2)
                .any(|pair| pair[0].eq_ignore_ascii_case("bearer") && pair[1].len() >= 8)
                || words.iter().any(|word| {
                    (word.starts_with("hf_") && word.len() > 22)
                        || (word.starts_with("sk-") && word.len() > 22)
                        || word.split_once("://").is_some_and(|(_, rest)| {
                            rest.split('/').next().is_some_and(|authority| {
                                authority.contains(':') && authority.contains('@')
                            })
                        })
                })
        }
        _ => false,
    }
}
fn receipt(path: &Path) -> Result<Value> {
    Ok(json::object([
        ("sha256", audio::sha256_file(path)?.into()),
        ("bytes", fs::metadata(path)?.len().into()),
    ]))
}
fn metadata(path: &Path) -> Result<Value> {
    if fs::metadata(path)?.len() > 8 * 1024 * 1024 {
        return Err("Metadata exceeds 8 MiB".into());
    }
    let value = json::read(path)?;
    value.object()?;
    if credentials(&value) {
        return Err("Credential-like metadata withheld".into());
    }
    Ok(value)
}
pub fn build(sources: &[PathBuf], output: &Path) -> Result<bool> {
    if sources.is_empty() {
        return Err("listen requires at least one song directory".into());
    }
    if output.exists() {
        return Err("Choose a fresh listening directory".into());
    }
    let sources: Vec<_> = sources
        .iter()
        .map(fs::canonicalize)
        .collect::<std::io::Result<_>>()?;
    let absolute = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    let parent = absolute.parent().ok_or("Output needs a parent directory")?;
    fn resolve(path: &Path) -> Result<PathBuf> {
        if path.exists() {
            return Ok(fs::canonicalize(path)?);
        }
        Ok(resolve(path.parent().ok_or("Invalid output parent")?)?
            .join(path.file_name().ok_or("Invalid output directory")?))
    }
    let destination = resolve(parent)?.join(
        absolute
            .file_name()
            .ok_or("Output needs a directory name")?,
    );
    if sources.iter().any(|p| destination.starts_with(p)) {
        return Err("Keep listening bundles outside source song directories".into());
    }
    fs::create_dir_all(destination.parent().unwrap())?;
    fs::create_dir(&destination)?;
    let mut cases = Vec::new();
    let mut files = BTreeMap::new();
    let mut complete = true;
    let mut html = String::from(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; media-src 'self' file:; base-uri 'none'; form-action 'none'\"><title>YuE2 listening comparison</title><style>body{font:16px/1.5 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;background:#f5f6f8;color:#17212b}article{background:white;border:1px solid #ddd;border-radius:12px;padding:24px;margin:24px 0}audio{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}li{color:#842b19}</style></head><body><h1>YuE2 listening comparison</h1><p>Local audio, exact requests and recorded metadata. Hash checks verify copied bytes; they do not assess musical quality. No weights, latents or logs are included.</p>",
    );
    for (i, source) in sources.iter().enumerate() {
        let directory = format!("case-{}", i + 1);
        let dest = destination.join(&directory);
        fs::create_dir(&dest)?;
        let mut issues = Vec::<String>::new();
        let mut objects = BTreeMap::new();
        for name in ["request.json", "config.json", "result.json"] {
            match metadata(&source.join(name)) {
                Ok(v) => {
                    objects.insert(name, v);
                }
                Err(_) => issues.push(format!(
                    "{name}: missing, invalid or credential-like metadata; withheld"
                )),
            }
        }
        let request = objects
            .get("request.json")
            .cloned()
            .unwrap_or_else(|| json::object([] as [(&str, Value); 0]));
        let result = objects
            .get("result.json")
            .cloned()
            .unwrap_or_else(|| json::object([] as [(&str, Value); 0]));
        if result.optional("status").and_then(|v| v.string().ok()) != Some("complete") {
            issues.push("Generation does not report complete status".into());
        }
        match result.optional("truncated").and_then(|v| v.object().ok()) {
            Some(t)
                if ["abc", "semantic"]
                    .iter()
                    .all(|k| t.get(*k).is_some_and(|v| v.boolean().ok() == Some(false))) => {}
            _ => issues.push("Generation is truncated or lacks valid truncation metadata".into()),
        }
        if source.join("failure.json").exists() {
            issues.push("Source contains a failure record; review it locally".into());
        }
        let mut copied = BTreeMap::new();
        let mut integrity_failed = false;
        let mut names: Vec<_> = objects.keys().copied().collect();
        if source.join("score.abc").is_file() {
            names.push("score.abc");
        }
        let audio_name = if source.join("audio.flac").is_file() {
            Some("audio.flac")
        } else if source.join("audio.wav").is_file() {
            Some("audio.wav")
        } else {
            None
        };
        if let Some(name) = audio_name {
            names.push(name);
        } else {
            issues.push("Missing audio.flac/audio.wav".into());
        }
        for name in names {
            let input = source.join(name);
            let target = dest.join(name);
            if input.is_symlink() {
                issues.push(format!("{name}: symlink withheld"));
                continue;
            }
            if name == "score.abc" {
                if fs::metadata(&input)?.len() > 8 * 1024 * 1024 {
                    issues.push("score.abc: oversized; withheld".into());
                    continue;
                }
                let text = fs::read_to_string(&input)?;
                if credentials(&Value::from(text.clone())) {
                    issues.push("score.abc: credential-like text withheld".into());
                    continue;
                }
                match super::abc::Score::parse(&text) {
                    Ok(_) => {}
                    Err(_) => issues.push("Score did not pass native ABC structural checks".into()),
                }
            }
            fs::copy(&input, &target)?;
            let observed = receipt(&target)?;
            if name != "result.json" {
                let expected = result.optional("artifacts").and_then(|a| a.optional(name));
                let matching = expected.is_some_and(|e| {
                    e.optional("sha256") == observed.optional("sha256")
                        && e.optional("bytes") == observed.optional("bytes")
                });
                if !matching {
                    integrity_failed = true;
                    issues.push(format!("{name}: no matching native hash/size receipt"));
                }
            }
            files.insert(format!("{directory}/{name}"), observed.clone());
            copied.insert(name.to_string(), observed);
        }
        let label = request
            .optional("id")
            .and_then(|v| v.string().ok())
            .unwrap_or("Song");
        html.push_str(&format!("<article><h2>{}</h2>", escape(label)));
        if let Some(name) = audio_name.filter(|_| !integrity_failed) {
            let target = dest.join(name);
            let signature_ok = if target.is_file() {
                let mut bytes = [0; 12];
                let count = fs::File::open(&target)?.read(&mut bytes)?;
                (name.ends_with("flac") && count >= 4 && &bytes[..4] == b"fLaC")
                    || (name.ends_with("wav")
                        && count >= 12
                        && (&bytes[..4] == b"RIFF" || &bytes[..4] == b"RF64")
                        && &bytes[8..12] == b"WAVE")
            } else {
                false
            };
            if signature_ok {
                html.push_str(&format!("<audio controls preload=\"none\" src=\"{directory}/{name}\"></audio><p><a href=\"{directory}/{name}\" download>Download audio</a></p>"));
            } else {
                issues.push("Audio container signature is missing/invalid; player withheld".into());
            }
        } else {
            html.push_str("<p>Verified player unavailable; review the issues below.</p>");
        }
        for (heading, key) in [("Style prompt", "style"), ("Lyrics", "lyrics")] {
            html.push_str(&format!(
                "<h3>{heading}</h3><pre>{}</pre>",
                escape(
                    request
                        .optional(key)
                        .and_then(|v| v.string().ok())
                        .unwrap_or("Unavailable")
                )
            ));
        }
        if let Some(duration) = result.optional("audio_seconds") {
            html.push_str(&format!(
                "<p>Recorded duration: {} seconds</p>",
                escape(&json::stringify(duration))
            ));
        }
        if !issues.is_empty() {
            complete = false;
            html.push_str("<ul>");
            for issue in &issues {
                html.push_str(&format!("<li>{}</li>", escape(issue)));
            }
            html.push_str("</ul>");
        }
        for (name, value) in &objects {
            html.push_str(&format!("<details><summary>{name}</summary><pre>{}</pre><a href=\"{directory}/{name}\">Download</a></details>",escape(&json::stringify(value))));
        }
        if copied.contains_key("score.abc") {
            html.push_str(&format!(
                "<details><summary>Score ABC</summary><pre>{}</pre></details>",
                escape(&fs::read_to_string(dest.join("score.abc"))?)
            ));
        }
        html.push_str("</article>");
        cases.push(json::object([
            ("directory", directory.into()),
            ("request", request),
            (
                "status",
                if issues.is_empty() {
                    "complete"
                } else {
                    "needs_review"
                }
                .into(),
            ),
            (
                "issues",
                Value::Array(issues.into_iter().map(Value::from).collect()),
            ),
            ("files", Value::Object(copied)),
        ]));
    }
    html.push_str(
        "<p><a href=\"manifest.json\">Bundle manifest and SHA-256 hashes</a></p></body></html>\n",
    );
    fs::write(destination.join("index.html"), html)?;
    files.insert(
        "index.html".into(),
        receipt(&destination.join("index.html"))?,
    );
    json::write(
        destination.join("manifest.json"),
        &json::object([
            ("schema", "yue2-rs-listening-bundle-v1".into()),
            ("cases", Value::Array(cases)),
            ("files", Value::Object(files)),
            (
                "hash_scope",
                "copied files and index.html; excludes manifest.json's recursive self-hash".into(),
            ),
            ("network_access", false.into()),
            ("weights_or_latents_copied", false.into()),
        ]),
    )?;
    println!("{}", destination.join("index.html").display());
    Ok(complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundle_escapes_requests_omits_latents_and_detects_tampering() {
        let root = std::env::temp_dir().join(format!(
            "yue2-listen-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let source = root.join("song");
        fs::create_dir_all(&source).unwrap();
        json::write(
            source.join("request.json"),
            &json::object([
                ("id", "test".into()),
                ("style", "<script>alert(1)</script>".into()),
                ("lyrics", "line & line".into()),
            ]),
        )
        .unwrap();
        json::write(
            source.join("config.json"),
            &json::object([] as [(&str, Value); 0]),
        )
        .unwrap();
        fs::write(source.join("audio.flac"), b"fLaC fixture bytes").unwrap();
        fs::write(source.join("latent.npy"), b"not copied").unwrap();
        let mut artifacts = BTreeMap::new();
        for name in ["request.json", "config.json", "audio.flac"] {
            artifacts.insert(name.to_string(), receipt(&source.join(name)).unwrap());
        }
        json::write(
            source.join("result.json"),
            &json::object([
                ("status", "complete".into()),
                (
                    "truncated",
                    json::object([("abc", false.into()), ("semantic", false.into())]),
                ),
                ("artifacts", Value::Object(artifacts)),
            ]),
        )
        .unwrap();
        let output = root.join("listen");
        assert!(build(std::slice::from_ref(&source), &output).unwrap());
        let page = fs::read_to_string(output.join("index.html")).unwrap();
        assert!(page.contains("&lt;script&gt;"));
        assert!(!page.contains("<script>"));
        assert!(page.contains("<audio controls"));
        assert!(!output.join("case-1/latent.npy").exists());
        assert!(build(std::slice::from_ref(&source), &output).is_err());
        assert!(build(std::slice::from_ref(&source), &source.join("inside")).is_err());
        assert!(
            build(
                std::slice::from_ref(&source),
                &source.join("new-parent/inside")
            )
            .is_err()
        );
        assert!(!source.join("new-parent").exists());
        fs::write(source.join("audio.flac"), b"fLaC changed bytes").unwrap();
        let altered = root.join("altered");
        assert!(!build(std::slice::from_ref(&source), &altered).unwrap());
        assert!(
            !fs::read_to_string(altered.join("index.html"))
                .unwrap()
                .contains("<audio controls")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
