use crate::{
    Result, invalid,
    json::{self, Value},
    tokenizer::Tokenizer,
};
pub const EOD: usize = 151643;
pub const ABC_START: usize = 151847;
pub const ABC_END: usize = 151848;
pub const MUSIC_START: usize = 151851;
pub const MUSIC_END: usize = 151852;
pub const CODEC_OFFSET: usize = 151853;
pub const CODEC_SIZE: usize = 32768;
pub const LATENT_START: usize = 184621;
pub const LATENT_END: usize = 184622;
pub const LATENT_PAD: usize = 184623;
pub const CONTEXT: usize = 24576;
pub const VERSION: &str = "yue2-native-v1";
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cot {
    Off,
    Melody,
    Full,
}
impl Cot {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "off" => Ok(Self::Off),
            "melody" => Ok(Self::Melody),
            "full" => Ok(Self::Full),
            _ => Err(invalid("cot must be off, melody or full")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Melody => "melody",
            Self::Full => "full",
        }
    }
    pub fn instruction(self) -> &'static str {
        match self {
            Self::Off => "Generate music with codec tokens from the given conditions.",
            Self::Melody => {
                "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
            }
            Self::Full => {
                "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
            }
        }
    }
}
#[derive(Clone, Debug)]
pub struct SongRequest {
    pub style: String,
    pub lyrics: String,
    pub cot: Cot,
    pub seed: u64,
    pub abc: Option<String>,
    pub cfg_scale: Option<f32>,
    pub id: String,
}
impl SongRequest {
    pub fn from_json(v: &Value) -> Result<Self> {
        let cfg = match v.optional("cfg_scale") {
            None | Some(Value::Null) => None,
            Some(x) => Some(x.f64()? as f32),
        };
        let abc = match v.optional("abc") {
            None | Some(Value::Null) => None,
            Some(x) => Some(x.string()?.into()),
        };
        let r = Self {
            style: v.get("style")?.string()?.into(),
            lyrics: v.get("lyrics")?.string()?.into(),
            cot: Cot::parse(v.text_or("cot", "full")?)?,
            seed: v.optional("seed").map_or(Ok(831001), Value::u64)?,
            abc,
            cfg_scale: cfg,
            id: v.text_or("id", "song")?.into(),
        };
        r.validate()?;
        Ok(r)
    }
    pub fn validate(&self) -> Result<()> {
        if self.seed >= 1u64 << 63 {
            return Err(invalid("seed must be in [0, 2**63)"));
        }
        let b = self.id.as_bytes();
        if b.is_empty()
            || b.len() > 180
            || !b[0].is_ascii_alphanumeric()
            || !b
                .iter()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-'))
        {
            return Err(invalid("id must be a filename-safe identifier"));
        }
        if self
            .abc
            .as_ref()
            .is_some_and(|s| s.trim().is_empty() || self.cot == Cot::Off)
        {
            return Err(invalid(
                "External ABC requires nonempty text and cot=melody/full",
            ));
        }
        if self
            .cfg_scale
            .is_some_and(|v| !v.is_finite() || !(0.0..=20.0).contains(&v))
        {
            return Err(invalid("cfg_scale must be finite in [0,20]"));
        }
        Ok(())
    }
    pub fn guidance(&self) -> f32 {
        self.cfg_scale
            .unwrap_or(if self.cot == Cot::Off { 1.01 } else { 1.0 })
    }
    pub fn text(&self) -> String {
        format!(
            "{}\n[Tags]\n{}\n[Lyrics]\n{}\n",
            self.cot.instruction(),
            self.style,
            self.lyrics
        )
    }
    pub fn json(&self) -> Value {
        json::object([
            ("style", self.style.clone().into()),
            ("lyrics", self.lyrics.clone().into()),
            ("cot", self.cot.name().into()),
            ("seed", self.seed.into()),
            ("id", self.id.clone().into()),
            ("abc", self.abc.clone().map_or(Value::Null, Value::from)),
            (
                "cfg_scale",
                self.cfg_scale
                    .map_or(Value::Null, |x| Value::from(x as f64)),
            ),
        ])
    }
}
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub penalty_window: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
            penalty_window: 50,
            min_tokens: 200,
            max_tokens: 9000,
        }
    }
}
impl Sampling {
    pub fn abc() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.005,
            penalty_window: 100,
            min_tokens: 32,
            max_tokens: 4096,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || !(0.0..=5.0).contains(&self.temperature)
            || !self.top_p.is_finite()
            || self.top_p <= 0.
            || self.top_p > 1.
            || self.top_k < 1
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.
            || !(1..=100).contains(&self.penalty_window)
            || self.min_tokens > self.max_tokens
            || self.max_tokens < 1
        {
            return Err(invalid("Invalid sampling configuration"));
        }
        Ok(())
    }
    pub fn update(&mut self, v: &Value) -> Result<()> {
        self.temperature = v.float_or("temperature", self.temperature as f64)? as f32;
        self.top_p = v.float_or("top_p", self.top_p as f64)? as f32;
        self.top_k = v.size_or("top_k", self.top_k)?;
        self.repetition_penalty =
            v.float_or("repetition_penalty", self.repetition_penalty as f64)? as f32;
        self.penalty_window = v.size_or("penalty_window", self.penalty_window)?;
        self.min_tokens = v.size_or("min_tokens", self.min_tokens)?;
        self.max_tokens = v.size_or("max_tokens", self.max_tokens)?;
        self.validate()
    }
    pub fn json(&self) -> Value {
        json::object([
            ("temperature", (self.temperature as f64).into()),
            ("top_p", (self.top_p as f64).into()),
            ("top_k", self.top_k.into()),
            (
                "repetition_penalty",
                (self.repetition_penalty as f64).into(),
            ),
            ("penalty_window", self.penalty_window.into()),
            ("min_tokens", self.min_tokens.into()),
            ("max_tokens", self.max_tokens.into()),
        ])
    }
}
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub abc: Sampling,
    pub semantic: Sampling,
    pub ode_steps: usize,
}
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            abc: Sampling::abc(),
            semantic: Sampling::default(),
            ode_steps: 32,
        }
    }
}
impl GenerationConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        if v.size_or("context", CONTEXT)? != CONTEXT
            || v.text_or("ode_method", "midpoint")? != "midpoint"
            || v.text_or("version", VERSION)? != VERSION
        {
            return Err(invalid("Unsupported generation protocol"));
        }
        let mut c = Self::default();
        if let Some(a) = v.optional("abc") {
            c.abc.update(a)?;
        }
        if let Some(s) = v.optional("semantic") {
            c.semantic.update(s)?;
        }
        c.ode_steps = v.size_or("ode_steps", 32)?;
        if c.ode_steps == 0 {
            return Err(invalid("ODE steps must be positive"));
        }
        Ok(c)
    }
    pub fn json(&self) -> Value {
        json::object([
            ("abc", self.abc.json()),
            ("semantic", self.semantic.json()),
            ("ode_steps", self.ode_steps.into()),
            ("ode_method", "midpoint".into()),
            ("context", CONTEXT.into()),
            ("version", VERSION.into()),
        ])
    }
}
pub fn prefix(
    request: &SongRequest,
    tokenizer: &Tokenizer,
    abc: Option<&[usize]>,
) -> Result<Vec<usize>> {
    let mut ids = vec![EOD];
    ids.extend(tokenizer.encode(&request.text())?);
    ids.push(ABC_START);
    if request.cot == Cot::Off {
        ids.extend([ABC_END, MUSIC_START]);
        return Ok(ids);
    }
    let owned;
    let abc = match abc {
        Some(a) => Some(a),
        None => {
            owned = request
                .abc
                .as_ref()
                .map(|s| tokenizer.encode(s))
                .transpose()?;
            owned.as_deref()
        }
    };
    if let Some(abc) = abc {
        if abc.iter().any(|&i| i >= EOD) {
            return Err(invalid("ABC IDs outside text vocabulary"));
        }
        ids.extend_from_slice(abc);
        ids.extend([ABC_END, MUSIC_START]);
    }
    Ok(ids)
}
pub fn negative_prefix(
    request: &SongRequest,
    tokenizer: &Tokenizer,
    abc: &[usize],
) -> Result<Vec<usize>> {
    let mut ids = vec![EOD];
    ids.extend(tokenizer.encode(request.cot.instruction())?);
    if request.cot != Cot::Off {
        if abc.iter().any(|&i| i >= EOD) {
            return Err(invalid("ABC IDs outside text vocabulary"));
        }
        ids.push(ABC_START);
        ids.extend_from_slice(abc);
        ids.push(ABC_END);
    }
    ids.push(MUSIC_START);
    Ok(ids)
}
