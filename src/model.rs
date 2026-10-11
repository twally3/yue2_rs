//! Checkpoint-compatible AR/NAR Mixture-of-Transformers and acoustic flow matching.
use crate::{
    Result,
    checkpoint::{Weight, Weights},
    invalid, json,
    protocol::{self, CONTEXT},
    tensor::{self, Linear, Norm, Tensor, round_bf16, silu, silu_slice},
};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub latent_dim: usize,
    pub max_latent_frames: usize,
    pub timestep_shift: f32,
    pub tie_word_embeddings: bool,
}
impl Config {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let v = json::read(path)?;
        if v.text_or("model_type", "yue2")? != "yue2" || v.text_or("latent_type", "vae")? != "vae" {
            return Err(invalid("Expected YuE2 VAE-latent model"));
        }
        let c = Self {
            hidden_size: v.size_or("hidden_size", 2048)?,
            num_hidden_layers: v.size_or("num_hidden_layers", 28)?,
            num_attention_heads: v.size_or("num_attention_heads", 16)?,
            num_key_value_heads: v.size_or("num_key_value_heads", 8)?,
            head_dim: v.size_or("head_dim", 128)?,
            intermediate_size: v.size_or("intermediate_size", 6144)?,
            vocab_size: v.size_or("vocab_size", 184704)?,
            rms_norm_eps: v.float_or("rms_norm_eps", 1e-6)? as f32,
            rope_theta: v.float_or("rope_theta", 1e6)? as f32,
            max_position_embeddings: v.size_or("max_position_embeddings", CONTEXT)?,
            latent_dim: v.size_or("latent_dim", 64)?,
            max_latent_frames: v.size_or("max_latent_frames", CONTEXT)?,
            timestep_shift: v.float_or("timestep_shift", 1.)? as f32,
            tie_word_embeddings: v.bool_or("tie_word_embeddings", false)?,
        };
        if [
            c.hidden_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.intermediate_size,
            c.vocab_size,
            c.latent_dim,
            c.max_latent_frames,
        ]
        .contains(&0)
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
            || !c.head_dim.is_multiple_of(2)
            || c.rope_theta <= 0.
            || !c.rope_theta.is_finite()
            || c.rms_norm_eps <= 0.
            || !c.rms_norm_eps.is_finite()
            || c.timestep_shift <= 0.
            || !c.timestep_shift.is_finite()
        {
            return Err(invalid("Invalid transformer configuration"));
        }
        Ok(c)
    }
}
fn linear(w: &Weights, name: &str, out: usize, input: usize, bias: bool) -> Result<Linear> {
    Linear::new(
        w.checked(&format!("{name}.weight"), &[out, input])?,
        if bias {
            Some(w.checked(&format!("{name}.bias"), &[out])?)
        } else {
            None
        },
        true,
    )
}
fn norm(w: &Weights, name: &str, dim: usize, c: &Config) -> Result<Norm> {
    Ok(Norm::new(
        w.checked(&format!("{name}.weight"), &[dim])?,
        c.rms_norm_eps,
        true,
    ))
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Norm,
    k_norm: Norm,
}
impl Attention {
    fn load(w: &Weights, name: &str, c: &Config) -> Result<Self> {
        let qdim = c.num_attention_heads * c.head_dim;
        let kdim = c.num_key_value_heads * c.head_dim;
        Ok(Self {
            q: linear(w, &format!("{name}.q_proj"), qdim, c.hidden_size, false)?,
            k: linear(w, &format!("{name}.k_proj"), kdim, c.hidden_size, false)?,
            v: linear(w, &format!("{name}.v_proj"), kdim, c.hidden_size, false)?,
            o: linear(w, &format!("{name}.o_proj"), c.hidden_size, qdim, false)?,
            q_norm: norm(w, &format!("{name}.q_norm"), c.head_dim, c)?,
            k_norm: norm(w, &format!("{name}.k_norm"), c.head_dim, c)?,
        })
    }
    fn enable_vectors(&mut self) {
        self.q.vector_backend = 1;
        self.k.vector_backend = 1;
        self.v.vector_backend = 1;
        self.o.vector_backend = 1;
    }
    fn qkv(&self, x: &Tensor, rotary: &Rotary, c: &Config) -> (Tensor, Tensor, Tensor) {
        let mut q = self.q.apply(x);
        let mut k = self.k.apply(x);
        let v = self.v.apply(x);
        self.q_norm.heads(&mut q, c.head_dim);
        self.k_norm.heads(&mut k, c.head_dim);
        rotary.apply(&mut q);
        rotary.apply(&mut k);
        (q, k, v)
    }
}
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}
impl Rotary {
    fn new(start: usize, rows: usize, c: &Config) -> Self {
        let half = c.head_dim / 2;
        let mut cos = Tensor::zeros(rows, half);
        let mut sin = Tensor::zeros(rows, half);
        for i in 0..half {
            let freq = 1.0 / c.rope_theta.powf((2 * i) as f32 / c.head_dim as f32);
            for r in 0..rows {
                let angle = (start + r) as f32 * freq;
                cos.data[r * half + i] = round_bf16(angle.cos());
                sin.data[r * half + i] = round_bf16(angle.sin());
            }
        }
        Self { cos, sin }
    }
    fn apply(&self, x: &mut Tensor) {
        let half = self.cos.cols;
        assert_eq!(x.rows, self.cos.rows);
        for r in 0..x.rows {
            for i in 0..half {
                let cos = self.cos.data[r * half + i];
                let sin = self.sin.data[r * half + i];
                for h in 0..x.cols / (2 * half) {
                    let a = r * x.cols + h * (2 * half) + i;
                    let b = a + half;
                    let left = x.data[a];
                    let right = x.data[b];
                    x.data[a] = round_bf16(round_bf16(left * cos) - round_bf16(right * sin));
                    x.data[b] = round_bf16(round_bf16(right * cos) + round_bf16(left * sin));
                }
            }
        }
    }
}
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Mlp {
    fn load(w: &Weights, name: &str, c: &Config) -> Result<Self> {
        Ok(Self {
            gate: linear(
                w,
                &format!("{name}.gate_proj"),
                c.intermediate_size,
                c.hidden_size,
                false,
            )?,
            up: linear(
                w,
                &format!("{name}.up_proj"),
                c.intermediate_size,
                c.hidden_size,
                false,
            )?,
            down: linear(
                w,
                &format!("{name}.down_proj"),
                c.hidden_size,
                c.intermediate_size,
                false,
            )?,
        })
    }
    fn enable_vectors(&mut self) {
        self.gate.vector_backend = 1;
        self.up.vector_backend = 1;
        self.down.vector_backend = 1;
    }
    fn apply(&self, x: &Tensor) -> Tensor {
        let mut gate = self.gate.apply(x);
        let up = self.up.apply(x);
        silu_slice(&mut gate.data, true);
        for (g, u) in gate.data.iter_mut().zip(&up.data) {
            *g = round_bf16(*g * u);
        }
        self.down.apply(&gate)
    }
}
struct Pathway {
    input_norm: Norm,
    attention: Attention,
    mlp_norm: Norm,
    mlp: Mlp,
}
impl Pathway {
    fn gpu_layer(&self, kv: &Kv) -> Option<tensor::GpuLayer> {
        Some(tensor::GpuLayer {
            weights: [
                self.attention.q.gpu_weight()?,
                self.attention.k.gpu_weight()?,
                self.attention.v.gpu_weight()?,
                self.attention.o.gpu_weight()?,
                self.mlp.gate.gpu_weight()?,
                self.mlp.up.gpu_weight()?,
                self.mlp.down.gpu_weight()?,
            ],
            norms: [
                self.input_norm.gpu_values(),
                self.attention.q_norm.gpu_values(),
                self.attention.k_norm.gpu_values(),
                self.mlp_norm.gpu_values(),
            ],
            key: kv.key.data.as_ptr(),
            value: kv.value.data.as_ptr(),
        })
    }
}
struct Layer {
    ar: Pathway,
    nar: Pathway,
}
impl Layer {
    fn load(w: &Weights, i: usize, c: &Config) -> Result<Self> {
        let p = format!("model.layers.{i}");
        let mut layer = Self {
            ar: Pathway {
                input_norm: norm(w, &format!("{p}.input_layernorm"), c.hidden_size, c)?,
                attention: Attention::load(w, &format!("{p}.self_attn"), c)?,
                mlp_norm: norm(
                    w,
                    &format!("{p}.post_attention_layernorm"),
                    c.hidden_size,
                    c,
                )?,
                mlp: Mlp::load(w, &format!("{p}.mlp"), c)?,
            },
            nar: Pathway {
                input_norm: norm(w, &format!("{p}.nar_input_layernorm"), c.hidden_size, c)?,
                attention: Attention::load(w, &format!("{p}.nar_self_attn"), c)?,
                mlp_norm: norm(w, &format!("{p}.nar_pre_mlp_layernorm"), c.hidden_size, c)?,
                mlp: Mlp::load(w, &format!("{p}.nar_mlp"), c)?,
            },
        };
        layer.ar.attention.enable_vectors();
        layer.ar.mlp.enable_vectors();
        Ok(layer)
    }
}
pub struct Kv {
    pub key: Tensor,
    pub value: Tensor,
}
pub struct Cache {
    id: usize,
    pub layers: Vec<Kv>,
    pub length: usize,
    pub capacity: usize,
}
impl Drop for Cache {
    fn drop(&mut self) {
        tensor::release_gpu_cache(self.id);
    }
}
impl Cache {
    pub fn new(c: &Config, capacity: usize) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
        Self {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            layers: (0..c.num_hidden_layers)
                .map(|_| Kv {
                    key: Tensor::zeros(0, c.num_key_value_heads * c.head_dim),
                    value: Tensor::zeros(0, c.num_key_value_heads * c.head_dim),
                })
                .collect(),
            length: 0,
            capacity,
        }
    }
}

thread_local! {
    static ATTENTION_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> = const {
        std::cell::RefCell::new((Vec::new(), Vec::new()))
    };
}
/// Query tiles preserve the full visible key set, including grouped-query heads.
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    c: &Config,
    causal_start: Option<usize>,
    cache: Option<(usize, usize)>,
) -> Tensor {
    if let Some((cache, layer)) = cache {
        if let Some(out) = tensor::gpu_cached_attention(
            q,
            k,
            v,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            cache,
            layer,
        ) {
            return out;
        }
    }
    if let Some(out) = tensor::gpu_attention(
        q,
        k,
        v,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.head_dim,
        causal_start,
    ) {
        return out;
    }
    let mut out = Tensor::zeros(q.rows, q.cols);
    let scale = (c.head_dim as f32).sqrt().recip();
    let groups = c.num_attention_heads / c.num_key_value_heads;
    ATTENTION_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let (scores, exponential) = &mut *scratch;
        for start in (0..q.rows).step_by(256) {
            let rows = (q.rows - start).min(256);
            let keys = causal_start.map_or(k.rows, |base| (base + start + rows).min(k.rows));
            scores.resize(rows * keys, 0.);
            exponential.resize(keys, 0.);
            for head in 0..c.num_attention_heads {
                let kh = head / groups;
                let qoffset = start * q.cols + head * c.head_dim;
                let koff = kh * c.head_dim;
                tensor::gemm_cpu(
                    &q.data[qoffset..],
                    &k.data[koff..],
                    scores,
                    rows,
                    keys,
                    c.head_dim,
                    true,
                    q.cols,
                    k.cols,
                    keys,
                    0.,
                );
                for r in 0..rows {
                    let row = &mut scores[r * keys..(r + 1) * keys];
                    let visible = causal_start.map_or(keys, |base| base + start + r + 1);
                    tensor::softmax_row(row, exponential, visible, scale);
                }
                tensor::gemm_cpu(
                    scores,
                    &v.data[koff..],
                    &mut out.data[qoffset..],
                    rows,
                    c.head_dim,
                    keys,
                    false,
                    keys,
                    v.cols,
                    out.cols,
                    0.,
                );
            }
        }
    });
    out.rounded()
}
pub struct Model {
    pub config: Config,
    embedding: Weight,
    latent_position: Weight,
    layers: Vec<Layer>,
    final_norm: Norm,
    lm_head: Linear,
    head_phase: std::sync::atomic::AtomicU8,
    vae2llm: Linear,
    llm2vae: Linear,
    time_in: Linear,
    time_out: Linear,
}
impl Drop for Model {
    fn drop(&mut self) {
        tensor::clear_weight_cache();
        tensor::release_gpu_model();
    }
}
impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let c = Config::read(path.join("config.json"))?;
        let w = Weights::open(path)?;
        let embedding = w.checked("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])?;
        let mut lm_head = if c.tie_word_embeddings {
            Linear::new(embedding.clone(), None, true)?
        } else {
            linear(&w, "lm_head", c.vocab_size, c.hidden_size, false)?
        };
        lm_head.vector_backend = 2;
        Ok(Self {
            latent_position: w
                .checked("latent_pos_embed.pe", &[c.max_latent_frames, c.hidden_size])?,
            layers: (0..c.num_hidden_layers)
                .map(|i| Layer::load(&w, i, &c))
                .collect::<Result<_>>()?,
            final_norm: norm(&w, "model.norm", c.hidden_size, &c)?,
            vae2llm: linear(&w, "vae2llm", c.hidden_size, c.latent_dim, true)?,
            llm2vae: linear(&w, "llm2vae", c.latent_dim, c.hidden_size, true)?,
            time_in: linear(&w, "time_embedder.mlp.0", c.hidden_size, 256, true)?,
            time_out: linear(
                &w,
                "time_embedder.mlp.2",
                c.hidden_size,
                c.hidden_size,
                true,
            )?,
            config: c,
            embedding,
            lm_head,
            head_phase: std::sync::atomic::AtomicU8::new(0),
        })
    }
    fn embed(&self, ids: &[usize]) -> Result<Tensor> {
        let c = &self.config;
        let mut out = Tensor::zeros(ids.len(), c.hidden_size);
        for (r, &id) in ids.iter().enumerate() {
            if id >= c.vocab_size {
                return Err(invalid("Token outside model vocabulary"));
            }
            for (i, v) in out.row_mut(r).iter_mut().enumerate() {
                *v = round_bf16(self.embedding.get(id * c.hidden_size + i));
            }
        }
        Ok(out)
    }
    pub fn forward(&self, ids: &[usize], cache: &mut Cache) -> Result<Tensor> {
        let x = self.hidden(ids, cache)?;
        Ok(self.lm_head.apply(&x.slice(x.rows - 1, x.rows)))
    }
    pub fn forward_phase(
        &self,
        ids: &[usize],
        cache: &mut Cache,
        phase: crate::sampling::Phase,
    ) -> Result<Tensor> {
        let x = self.hidden(ids, cache)?;
        Ok(self.project_phase(&x.slice(x.rows - 1, x.rows), phase))
    }
    fn project_phase(&self, x: &Tensor, phase: crate::sampling::Phase) -> Tensor {
        let code = match phase {
            crate::sampling::Phase::Abc => 1,
            crate::sampling::Phase::Semantic => 2,
        };
        if self
            .head_phase
            .swap(code, std::sync::atomic::Ordering::Relaxed)
            != code
        {
            self.lm_head.clear_cached();
        }
        let (start, end, stop) = match phase {
            crate::sampling::Phase::Abc => (0, protocol::EOD, protocol::ABC_END),
            crate::sampling::Phase::Semantic => (
                protocol::CODEC_OFFSET,
                protocol::CODEC_OFFSET + protocol::CODEC_SIZE,
                protocol::MUSIC_END,
            ),
        };
        let values = self.lm_head.apply_range(x, start, end);
        let stop_value = self.lm_head.apply_range(x, stop, stop + 1);
        let mut full = Tensor::zeros(x.rows, self.config.vocab_size);
        full.data.fill(f32::NEG_INFINITY);
        for r in 0..x.rows {
            full.row_mut(r)[start..end].copy_from_slice(values.row(r));
            full.row_mut(r)[stop] = stop_value.data[r];
        }
        full
    }
    pub fn prefill_phase(
        &self,
        ids: &[usize],
        capacity: usize,
        phase: crate::sampling::Phase,
    ) -> Result<(Cache, Tensor)> {
        let mut cache = Cache::new(&self.config, capacity);
        let mut last = None;
        for chunk in ids.chunks(256) {
            let x = self.hidden(chunk, &mut cache)?;
            last = Some(x.slice(x.rows - 1, x.rows));
        }
        let x = last.ok_or_else(|| invalid("Empty prefill"))?;
        Ok((cache, self.project_phase(&x, phase)))
    }
    fn hidden(&self, ids: &[usize], cache: &mut Cache) -> Result<Tensor> {
        if ids.is_empty()
            || cache.length + ids.len() > cache.capacity
            || cache.length + ids.len() > self.config.max_position_embeddings
            || cache.layers.len() != self.layers.len()
        {
            return Err(invalid("AR input exceeds cache/model context"));
        }
        let mut x = self.embed(ids)?;
        let rotary = Rotary::new(cache.length, ids.len(), &self.config);
        if ids.len() == 1 && cache.length >= 255 {
            let layers: Option<Vec<_>> = self
                .layers
                .iter()
                .zip(&cache.layers)
                .map(|(layer, kv)| layer.ar.gpu_layer(kv))
                .collect();
            if let Some(layers) = layers {
                let c = &self.config;
                if let Some((hidden, shadow)) = tensor::gpu_ar_stack(
                    &x,
                    &layers,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.intermediate_size,
                    cache.length,
                    cache.capacity.min(c.max_position_embeddings),
                    cache.id,
                    c.rms_norm_eps,
                    &rotary.cos,
                    &rotary.sin,
                ) {
                    let width = c.num_key_value_heads * c.head_dim;
                    for (index, kv) in cache.layers.iter_mut().enumerate() {
                        let start = index * 2 * width;
                        kv.key.data.extend_from_slice(&shadow[start..start + width]);
                        kv.key.rows += 1;
                        kv.value
                            .data
                            .extend_from_slice(&shadow[start + width..start + 2 * width]);
                        kv.value.rows += 1;
                    }
                    cache.length += 1;
                    return Ok(self.final_norm.apply(&hidden));
                }
            }
        }
        for (index, (layer, kv)) in self.layers.iter().zip(&mut cache.layers).enumerate() {
            let p = &layer.ar;
            let (q, k, v) = p
                .attention
                .qkv(&p.input_norm.apply(&x), &rotary, &self.config);
            kv.key.append(&k);
            kv.value.append(&v);
            let h = attention(
                &q,
                &kv.key,
                &kv.value,
                &self.config,
                Some(cache.length),
                Some((cache.id, index)),
            );
            x.add(&p.attention.o.apply(&h), true);
            x.add(&p.mlp.apply(&p.mlp_norm.apply(&x)), true);
        }
        cache.length += ids.len();
        Ok(self.final_norm.apply(&x))
    }
    pub fn prefill(
        &self,
        ids: &[usize],
        capacity: usize,
        logits: bool,
    ) -> Result<(Cache, Option<Tensor>)> {
        if ids.is_empty() || capacity < ids.len() {
            return Err(invalid("Invalid prefill/cache capacity"));
        }
        let mut cache = Cache::new(&self.config, capacity);
        let mut last = None;
        for chunk in ids.chunks(256) {
            let x = self.hidden(chunk, &mut cache)?;
            if logits {
                last = Some(x.slice(x.rows - 1, x.rows));
            }
        }
        Ok((cache, last.map(|x| self.lm_head.apply(&x))))
    }
    fn velocity(
        &self,
        state: &Tensor,
        cache: &Cache,
        rotary: &Rotary,
        raw_t: f64,
        nar_cond_end: usize,
    ) -> Tensor {
        let c = &self.config;
        let n = state.rows + 2;
        let mut padded = Tensor::zeros(n, c.latent_dim);
        padded.data[c.latent_dim..(state.rows + 1) * c.latent_dim].copy_from_slice(&state.data);
        let mut x = self.vae2llm.apply(&padded);
        let raw = round_bf16(raw_t as f32);
        let sig = round_bf16(1.0 / (1.0 + (-raw).exp()));
        let top = round_bf16(c.timestep_shift * sig);
        let bottom = round_bf16(1. + round_bf16((c.timestep_shift - 1.) * sig));
        let t = round_bf16(top / bottom);
        let mut sinus = Tensor::zeros(1, 256);
        for i in 0..128 {
            let f = (-10000f32.ln() * i as f32 / 128.).exp();
            sinus.data[i] = round_bf16((t * f).cos());
            sinus.data[128 + i] = round_bf16((t * f).sin());
        }
        let mut time = self.time_in.apply(&sinus);
        for v in &mut time.data {
            *v = round_bf16(silu(*v));
        }
        let time = self.time_out.apply(&time);
        for r in 0..n {
            for i in 0..c.hidden_size {
                let pe = round_bf16(
                    self.latent_position
                        .get(r.min(c.max_latent_frames - 1) * c.hidden_size + i),
                );
                let v = &mut x.data[r * c.hidden_size + i];
                *v = round_bf16(round_bf16(*v + time.data[i]) + pe);
            }
        }
        let visible = if nar_cond_end > 0 {
            nar_cond_end.min(cache.length)
        } else {
            cache.length
        };
        let gpu_layers: Option<Vec<_>> = self
            .layers
            .iter()
            .zip(&cache.layers)
            .map(|(layer, kv)| layer.nar.gpu_layer(kv))
            .collect();
        if let Some(layers) = gpu_layers {
            if let Some(hidden) = tensor::gpu_stack(
                &x,
                &layers,
                c.num_attention_heads,
                c.num_key_value_heads,
                c.head_dim,
                c.intermediate_size,
                visible,
                cache.id,
                c.rms_norm_eps,
                &rotary.cos,
                &rotary.sin,
            ) {
                return self
                    .llm2vae
                    .apply(&self.final_norm.apply(&hidden).slice(1, n - 1));
            }
        }
        for (layer, kv) in self.layers.iter().zip(&cache.layers) {
            let p = &layer.nar;
            let (q, k, v) = p.attention.qkv(&p.input_norm.apply(&x), rotary, c);
            let visible = if nar_cond_end > 0 {
                nar_cond_end.min(cache.length)
            } else {
                cache.length
            };
            let mut key = kv.key.slice(0, visible);
            let mut value = kv.value.slice(0, visible);
            key.append(&k);
            value.append(&v);
            let h = attention(&q, &key, &value, c, None, None);
            x.add(&p.attention.o.apply(&h), true);
            x.add(&p.mlp.apply(&p.mlp_norm.apply(&x)), true);
        }
        self.llm2vae
            .apply(&self.final_norm.apply(&x).slice(1, n - 1))
    }
    pub fn synthesize(
        &self,
        prefix: &[usize],
        codec: &[usize],
        seed: u64,
        steps: usize,
    ) -> Result<Tensor> {
        // Sequential AR matrices are no longer needed in the acoustic stage.
        tensor::clear_weight_cache();
        tensor::clear_gpu_weight_cache();
        if self.config.latent_dim != 64
            || codec.is_empty()
            || codec.iter().any(|&t| t >= protocol::CODEC_SIZE)
            || steps == 0
        {
            return Err(invalid(
                "Expected nonempty codec tokens and a 64-dimensional latent model",
            ));
        }
        let capacity = CONTEXT.min(self.config.max_position_embeddings);
        let size = capacity
            .checked_sub(prefix.len() + 3)
            .map(|n| n / 2)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("Prefix leaves no acoustic context"))?;
        // Generate the full-song CPU noise once, then slice at original context cuts.
        let mut rng = crate::sampling::Generator::new(seed);
        let noise = rng.normal_tensor(codec.len(), 64);
        let mut out = Tensor::zeros(0, 64);
        let total = codec.len().div_ceil(size);
        let dt = 1. / steps as f32;
        for (chunk_index, start) in (0..codec.len()).step_by(size).enumerate() {
            let end = (start + size).min(codec.len());
            let mut ids = prefix.to_vec();
            ids.extend(
                codec[start..end]
                    .iter()
                    .map(|&t| t + protocol::CODEC_OFFSET),
            );
            ids.push(protocol::MUSIC_END);
            eprintln!(
                "Acoustic chunk {}/{}: {} frames",
                chunk_index + 1,
                total,
                end - start
            );
            let (cache, _) = self.prefill(&ids, ids.len(), false)?;
            tensor::clear_gpu_weight_cache();
            let mut state = noise.slice(start, end).rounded();
            let rotary = Rotary::new(cache.length, state.rows + 2, &self.config);
            for step in 0..steps {
                let t = 1. - step as f64 / steps as f64;
                let raw = logit(t);
                let first = self.velocity(&state, &cache, &rotary, raw, 0);
                let mut mid = state.clone();
                for (i, v) in mid.data.iter_mut().enumerate() {
                    *v = round_bf16(*v - round_bf16(first.data[i] * (dt / 2.)));
                }
                let velocity =
                    self.velocity(&mid, &cache, &rotary, logit(t - 0.5 / steps as f64), 0);
                for (i, v) in state.data.iter_mut().enumerate() {
                    *v = round_bf16(*v - round_bf16(velocity.data[i] * dt));
                }
                eprintln!(
                    "Acoustic chunk {}/{}: step {}/{}",
                    chunk_index + 1,
                    total,
                    step + 1,
                    steps
                );
            }
            if !state.finite() {
                return Err(invalid("Flow matching produced nonfinite latents"));
            }
            out.append(&state);
        }
        Ok(out)
    }
}
fn logit(t: f64) -> f64 {
    (t / (1. - t)).ln().clamp(-20., 20.)
}
