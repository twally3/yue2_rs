//! Staged score planning → semantic tokens → acoustic latents → stereo audio.
use crate::{
    Result, audio, invalid,
    json::{self, Value},
    model::Model,
    protocol::{self, Cot, GenerationConfig, Sampling, SongRequest},
    sampling::{self, Generated, Phase},
    tensor::Tensor,
    tokenizer::Tokenizer,
    vae::Vae,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone)]
pub struct SymbolicPlan {
    pub request: SongRequest,
    pub abc: Option<String>,
    pub abc_ids: Vec<usize>,
    pub prefix: Vec<usize>,
    pub timing: Value,
    pub truncated: bool,
}
impl SymbolicPlan {
    pub fn save(&self, directory: impl AsRef<Path>) -> Result<()> {
        let d = directory.as_ref();
        std::fs::create_dir_all(d)?;
        if let Some(abc) = &self.abc {
            std::fs::write(d.join("score.abc"), abc)?;
        }
        audio::save_ids(d.join("abc_tokens.npy"), &self.abc_ids)?;
        audio::save_ids(d.join("prefix.npy"), &self.prefix)?;
        json::write(
            d.join("plan.json"),
            &json::object([
                ("request", self.request.json()),
                ("timing", self.timing.clone()),
                ("truncated", self.truncated.into()),
                ("prefix", json::ids(&self.prefix)),
                ("abc_ids", json::ids(&self.abc_ids)),
                ("abc", self.abc.clone().map_or(Value::Null, Value::from)),
            ]),
        )?;
        let mut manifest = BTreeMap::new();
        for name in ["plan.json", "abc_tokens.npy", "prefix.npy"]
            .into_iter()
            .chain(self.abc.as_ref().map(|_| "score.abc"))
        {
            manifest.insert(name.into(), Value::from(audio::sha256_file(d.join(name))?));
        }
        json::write(d.join("plan_manifest.json"), &Value::Object(manifest))
    }
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let d = directory.as_ref();
        let manifest = json::read(d.join("plan_manifest.json"))?;
        for name in ["plan.json", "abc_tokens.npy", "prefix.npy"] {
            manifest.get(name)?;
        }
        for (name, digest) in manifest.object()? {
            if !["plan.json", "abc_tokens.npy", "prefix.npy", "score.abc"].contains(&name.as_str())
                || d.join(name).is_symlink()
                || audio::sha256_file(d.join(name))? != digest.string()?
            {
                return Err(invalid(
                    "Saved plan changed; supply edited ABC as an external score",
                ));
            }
        }
        let v = json::read(d.join("plan.json"))?;
        let ids = |name: &str| -> Result<Vec<usize>> {
            v.get(name)?.array()?.iter().map(Value::usize).collect()
        };
        let abc = match v.get("abc")? {
            Value::Null => None,
            s => Some(s.string()?.into()),
        };
        if let Some(s) = &abc {
            manifest.get("score.abc")?;
            if std::fs::read_to_string(d.join("score.abc"))? != *s {
                return Err(invalid("Saved ABC text mismatch"));
            }
        }
        let abc_ids = ids("abc_ids")?;
        let prefix = ids("prefix")?;
        if audio::load_ids(d.join("abc_tokens.npy"))? != abc_ids
            || audio::load_ids(d.join("prefix.npy"))? != prefix
        {
            return Err(invalid("Saved plan token array mismatch"));
        }
        Ok(Self {
            request: SongRequest::from_json(v.get("request")?)?,
            abc,
            abc_ids,
            prefix,
            timing: v.get("timing")?.clone(),
            truncated: v.get("truncated")?.boolean()?,
        })
    }
}
pub struct SemanticResult {
    pub plan: SymbolicPlan,
    pub tokens: Vec<usize>,
    pub timing: Value,
    pub truncated: bool,
}
pub struct SongResult {
    pub audio: Tensor,
    pub sample_rate: usize,
    pub semantic: SemanticResult,
    pub latents: Tensor,
    pub config: Value,
    pub weights: Value,
    pub timing: Value,
}
impl SongResult {
    pub fn truncated(&self) -> bool {
        self.semantic.plan.truncated || self.semantic.truncated
    }
    pub fn save_artifacts(&self, directory: impl AsRef<Path>) -> Result<()> {
        let d = directory.as_ref();
        if d.exists() {
            return Err(invalid(
                "Choose a fresh output directory to retain each version",
            ));
        }
        std::fs::create_dir_all(d)?;
        self.semantic.plan.save(d)?;
        audio::save_ids(d.join("semantic.npy"), &self.semantic.tokens)?;
        audio::save_tensor(d.join("latent.npy"), &self.latents)?;
        audio::save_flac(d.join("audio.flac"), &self.audio, self.sample_rate)?;
        json::write(d.join("request.json"), &self.semantic.plan.request.json())?;
        json::write(d.join("config.json"), &self.config)?;
        let mut artifacts = BTreeMap::new();
        for entry in std::fs::read_dir(d)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() {
                artifacts.insert(
                    entry.file_name().to_string_lossy().into_owned(),
                    json::object([
                        ("sha256", audio::sha256_file(&p)?.into()),
                        ("bytes", entry.metadata()?.len().into()),
                    ]),
                );
            }
        }
        json::write(
            d.join("result.json"),
            &json::object([
                ("status", "complete".into()),
                (
                    "truncated",
                    json::object([
                        ("abc", self.semantic.plan.truncated.into()),
                        ("semantic", self.semantic.truncated.into()),
                    ]),
                ),
                ("sample_rate", self.sample_rate.into()),
                (
                    "audio_seconds",
                    (self.audio.rows as f64 / self.sample_rate as f64).into(),
                ),
                ("weights", self.weights.clone()),
                ("timing", self.timing.clone()),
                ("artifacts", Value::Object(artifacts)),
            ]),
        )
    }
}
fn timing(g: &Generated) -> Value {
    json::object([
        ("seconds", g.seconds.into()),
        ("prefill_seconds", g.prefill_seconds.into()),
        ("content_tokens", g.tokens.len().into()),
        ("execution", crate::tensor::backend_name().into()),
    ])
}
pub(crate) fn verify_weights(path: &Path) -> Result<Value> {
    let manifest = path.join("weights_manifest.json");
    let v = if manifest.exists() {
        json::read(manifest)?
    } else {
        let mut entries = BTreeMap::new();
        for e in std::fs::read_dir(path)? {
            let e = e?;
            let p = e.path();
            if p.extension().is_some_and(|s| s == "safetensors") {
                entries.insert(
                    e.file_name().to_string_lossy().into_owned(),
                    json::object([
                        ("bytes", e.metadata()?.len().into()),
                        ("sha256", audio::sha256_file(&p)?.into()),
                    ]),
                );
            }
        }
        json::object([("files", Value::Object(entries))])
    };
    if v.get("files")?.object()?.is_empty() {
        return Err(invalid("No model weights"));
    }
    for (name, entry) in v.get("files")?.object()? {
        if Path::new(name).components().count() != 1 {
            return Err(invalid("Invalid weight manifest filename"));
        }
        let p = path.join(name);
        if p.metadata()?.len() != entry.get("bytes")?.u64()?
            || audio::sha256_file(p)? != entry.get("sha256")?.string()?
        {
            return Err(invalid(format!("Checkpoint integrity failed for {name}")));
        }
    }
    Ok(json::object([
        ("files", v.get("files")?.clone()),
        (
            "config_sha256",
            audio::sha256_file(path.join("config.json"))?.into(),
        ),
    ]))
}
pub struct Pipeline {
    pub generation_config: GenerationConfig,
    pub vae_core_frames: usize,
    model_path: PathBuf,
    vae_path: PathBuf,
    pub tokenizer: Tokenizer,
    model: Option<Model>,
    vae: Option<Vae>,
    weights: Value,
}
impl Pipeline {
    pub fn close(&mut self) {
        self.model = None;
        self.vae = None;
    }
    pub fn result(&self, semantic: SemanticResult, latents: Tensor, audio: Tensor) -> SongResult {
        SongResult {
            audio,
            sample_rate: self.vae.as_ref().map_or(48000, |v| v.sample_rate),
            semantic,
            latents,
            config: self.generation_config.json(),
            weights: self.weights.clone(),
            timing: json::object([] as [(&str, Value); 0]),
        }
    }
    pub fn from_pretrained(model: impl AsRef<Path>, vae: impl AsRef<Path>) -> Result<Self> {
        Self::from_pretrained_revision(model, vae, None, None)
    }
    pub fn from_pretrained_revision(
        model: impl AsRef<Path>,
        vae: impl AsRef<Path>,
        revision: Option<&str>,
        vae_revision: Option<&str>,
    ) -> Result<Self> {
        let model_path = crate::checkpoint::resolve_model(model.as_ref(), revision)?;
        let vae_path = crate::checkpoint::resolve_model(vae.as_ref(), vae_revision)?;
        eprintln!("Verifying checkpoints");
        let weights = json::object([
            ("mot", verify_weights(&model_path)?),
            ("vae", verify_weights(&vae_path)?),
        ]);
        let config_path = model_path.join("yue2_generation_config.json");
        let generation_config = if config_path.exists() {
            GenerationConfig::from_json(&json::read(config_path)?)?
        } else {
            GenerationConfig::default()
        };
        let tokenizer = Tokenizer::open(model_path.join("qwen.tiktoken"))?;
        eprintln!("Matrix backend: {}", crate::tensor::backend_name());
        Ok(Self {
            model_path,
            vae_path,
            tokenizer,
            generation_config,
            vae_core_frames: 512,
            model: None,
            vae: None,
            weights,
        })
    }
    fn model(&mut self) -> Result<&Model> {
        if self.model.is_none() {
            eprintln!("Loading YuE2 transformer");
            self.model = Some(Model::open(&self.model_path)?);
        }
        Ok(self.model.as_ref().unwrap())
    }
    pub fn plan(
        &mut self,
        request: SongRequest,
        overrides: Option<&Sampling>,
    ) -> Result<SymbolicPlan> {
        request.validate()?;
        if request.cot == Cot::Off {
            return Ok(SymbolicPlan {
                prefix: protocol::prefix(&request, &self.tokenizer, None)?,
                request,
                abc: None,
                abc_ids: vec![],
                timing: json::object([("seconds", 0f64.into())]),
                truncated: false,
            });
        }
        if let Some(abc) = &request.abc {
            let ids = self.tokenizer.encode(abc)?;
            return Ok(SymbolicPlan {
                prefix: protocol::prefix(&request, &self.tokenizer, Some(&ids))?,
                abc: Some(abc.clone()),
                request,
                abc_ids: ids,
                timing: json::object([("seconds", 0f64.into())]),
                truncated: false,
            });
        }
        let prefix = protocol::prefix(&request, &self.tokenizer, None)?;
        let sample = overrides
            .cloned()
            .unwrap_or_else(|| self.generation_config.abc.clone());
        let generated = sampling::generate(
            self.model()?,
            &prefix,
            &sample,
            request.seed,
            Phase::Abc,
            None,
            1.,
            false,
        )?;
        let time = timing(&generated);
        let abc = self.tokenizer.decode(&generated.tokens);
        Ok(SymbolicPlan {
            prefix: protocol::prefix(&request, &self.tokenizer, Some(&generated.tokens))?,
            request,
            abc: Some(abc),
            abc_ids: generated.tokens,
            timing: time,
            truncated: generated.truncated,
        })
    }
    pub fn generate_semantic(
        &mut self,
        plan: SymbolicPlan,
        overrides: Option<&Sampling>,
    ) -> Result<SemanticResult> {
        if plan.prefix != protocol::prefix(&plan.request, &self.tokenizer, Some(&plan.abc_ids))? {
            return Err(invalid("Plan prefix disagrees with exact request/ABC IDs"));
        }
        let scale = plan.request.guidance();
        let negative = if scale != 1. {
            Some(protocol::negative_prefix(
                &plan.request,
                &self.tokenizer,
                &plan.abc_ids,
            )?)
        } else {
            None
        };
        let sample = overrides
            .cloned()
            .unwrap_or_else(|| self.generation_config.semantic.clone());
        let generated = sampling::generate(
            self.model()?,
            &plan.prefix,
            &sample,
            plan.request.seed,
            Phase::Semantic,
            negative.as_deref(),
            scale,
            plan.request.cot == Cot::Off,
        )?;
        let time = timing(&generated);
        Ok(SemanticResult {
            plan,
            tokens: generated
                .tokens
                .iter()
                .map(|&i| i - protocol::CODEC_OFFSET)
                .collect(),
            timing: time,
            truncated: generated.truncated,
        })
    }
    pub fn synthesize(&mut self, semantic: &SemanticResult) -> Result<Tensor> {
        if semantic.plan.prefix
            != protocol::prefix(
                &semantic.plan.request,
                &self.tokenizer,
                Some(&semantic.plan.abc_ids),
            )?
        {
            return Err(invalid("Semantic result lost exact prefix"));
        }
        let steps = self.generation_config.ode_steps;
        self.model()?.synthesize(
            &semantic.plan.prefix,
            &semantic.tokens,
            semantic.plan.request.seed,
            steps,
        )
    }
    pub fn decode(&mut self, latents: &Tensor, full: bool) -> Result<Tensor> {
        if self.vae.is_none() {
            eprintln!("Loading stereo VAE");
            self.vae = Some(Vae::open(&self.vae_path, true)?);
            eprintln!("VAE decoder: {}", self.vae.as_ref().unwrap().backend_name());
        }
        let vae = self.vae.as_ref().unwrap();
        let mut audio = if full {
            vae.decode(latents)?
        } else {
            vae.decode_tiled(latents, self.vae_core_frames, 16)?
        };
        for v in &mut audio.data {
            *v = v.clamp(-1., 1.);
        }
        Ok(audio)
    }
    pub fn generate(&mut self, request: SongRequest) -> Result<SongResult> {
        let start = Instant::now();
        let cot = request.cot;
        let scale = request.guidance();
        let plan = self.plan(request, None)?;
        let semantic = self.generate_semantic(plan, None)?;
        let nar_start = Instant::now();
        let latents = self.synthesize(&semantic)?;
        let nar_seconds = nar_start.elapsed().as_secs_f64();
        let vae_start = Instant::now();
        let audio = self.decode(&latents, false)?;
        let timing = json::object([
            ("abc", semantic.plan.timing.clone()),
            ("semantic", semantic.timing.clone()),
            ("nar_seconds", nar_seconds.into()),
            ("vae_seconds", vae_start.elapsed().as_secs_f64().into()),
            ("e2e_seconds", start.elapsed().as_secs_f64().into()),
        ]);
        let config = json::object([
            ("generation", self.generation_config.json()),
            ("cot", cot.name().into()),
            ("cfg_scale", (scale as f64).into()),
            ("backend", crate::tensor::backend_name().into()),
            ("model_dtype", "bfloat16".into()),
            ("vae_dtype", "float32".into()),
            (
                "vae_backend",
                self.vae.as_ref().unwrap().backend_name().into(),
            ),
            ("vae_decode", "halo_crop".into()),
            ("vae_core_frames", self.vae_core_frames.into()),
            ("vae_halo_frames", 16usize.into()),
        ]);
        Ok(SongResult {
            audio,
            sample_rate: self.vae.as_ref().unwrap().sample_rate,
            semantic,
            latents,
            config,
            weights: self.weights.clone(),
            timing,
        })
    }
}
