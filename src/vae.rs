//! FP32 Oobleck VAE: weight normalization, SnakeBeta, and exact-core tiled decode.
use crate::{
    Result,
    checkpoint::{DType, Weights},
    invalid,
    json::{self, Value},
    tensor::{self, Tensor},
};
use std::path::Path;

struct ConvPhase {
    taps: Vec<usize>,
    weight: Vec<f32>,
    id: usize,
}
struct Conv {
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    transpose: bool,
    weight: Vec<f32>,
    weight_id: usize,
    phases: Vec<ConvPhase>,
    bias: Vec<f32>,
}
impl Conv {
    fn load(
        w: &Weights,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        transpose: bool,
        bias: bool,
    ) -> Result<Self> {
        let shape = if transpose {
            [input, output, kernel]
        } else {
            [output, input, kernel]
        };
        let v = w.checked(&format!("{name}.weight_v"), &shape)?;
        let g = w.checked(&format!("{name}.weight_g"), &[shape[0], 1, 1])?;
        if v.dtype != DType::F32 || g.dtype != DType::F32 {
            return Err(invalid("VAE weights must remain FP32"));
        }
        let mut weight = v.values();
        let gain = g.values();
        for (i, row) in weight.chunks_mut(shape[1] * kernel).enumerate() {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if !norm.is_finite() || norm == 0. {
                return Err(invalid("Invalid VAE weight normalization"));
            }
            for x in row {
                *x *= gain[i] / norm;
            }
        }
        let bias = if bias {
            let b = w.checked(&format!("{name}.bias"), &[output])?;
            if b.dtype != DType::F32 {
                return Err(invalid("VAE bias must be FP32"));
            }
            b.values()
        } else {
            vec![0.; output]
        };
        let mut phases = Vec::new();
        if transpose {
            for phase in 0..stride {
                let taps: Vec<_> = (0..kernel)
                    .filter(|&j| j % stride == (phase + padding) % stride)
                    .collect();
                let cols = input * taps.len();
                let mut packed = vec![0.; output * cols];
                for o in 0..output {
                    for i in 0..input {
                        for (t, &j) in taps.iter().enumerate() {
                            packed[o * cols + i * taps.len() + t] =
                                weight[(i * output + o) * kernel + j];
                        }
                    }
                }
                phases.push(ConvPhase {
                    taps,
                    weight: packed,
                    id: crate::checkpoint::resource_id(),
                });
            }
            weight = Vec::new();
        }
        Ok(Self {
            input,
            output,
            kernel,
            stride,
            padding,
            dilation,
            transpose,
            weight,
            weight_id: crate::checkpoint::resource_id(),
            phases,
            bias,
        })
    }
    fn length(&self, n: usize) -> usize {
        if self.transpose {
            (n - 1) * self.stride + self.dilation * (self.kernel - 1) + 1 - 2 * self.padding
        } else {
            (n + 2 * self.padding - self.dilation * (self.kernel - 1) - 1) / self.stride + 1
        }
    }
    fn support(&self, low: i64, high: i64) -> (i64, i64) {
        let s = self.stride as i64;
        let p = self.padding as i64;
        let d = self.dilation as i64;
        let k = self.kernel as i64;
        if self.transpose {
            (div_ceil(low + p - d * (k - 1), s), (high + p).div_euclid(s))
        } else {
            (low * s - p, high * s - p + d * (k - 1))
        }
    }
    fn apply(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.cols, self.input);
        if self.transpose {
            return self.transposed(x);
        }
        let length = self.length(x.rows);
        let cols = self.input * self.kernel;
        let mut y = Tensor::zeros(length, self.output);
        let mut patch = Vec::new();
        for start in (0..length).step_by(512) {
            let rows = (length - start).min(512);
            patch.resize(rows * cols, 0.);
            patch.fill(0.);
            for r in 0..rows {
                for j in 0..self.kernel {
                    let pos = ((start + r) * self.stride + j * self.dilation) as isize
                        - self.padding as isize;
                    if pos >= 0 && (pos as usize) < x.rows {
                        for c in 0..self.input {
                            patch[r * cols + c * self.kernel + j] =
                                x.data[pos as usize * x.cols + c];
                        }
                    }
                }
            }
            tensor::cached_linear_f32(
                &patch,
                &self.weight,
                &mut y.data[start * self.output..],
                rows,
                self.output,
                cols,
                self.weight_id,
            );
        }
        for row in y.data.chunks_mut(self.output) {
            for (v, b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        }
        y
    }
    fn transposed(&self, x: &Tensor) -> Tensor {
        assert_eq!(self.dilation, 1);
        let length = self.length(x.rows);
        let mut y = Tensor::zeros(length, self.output);
        for phase in 0..self.stride {
            let packed = &self.phases[phase];
            let taps = &packed.taps;
            let cols = self.input * taps.len();
            let rows = (length.saturating_sub(phase)).div_ceil(self.stride);
            if rows == 0 {
                continue;
            }
            let mut patch = Vec::new();
            let mut result = Vec::new();
            for start in (0..rows).step_by(512) {
                let count = (rows - start).min(512);
                patch.resize(count * cols, 0.);
                patch.fill(0.);
                for r in 0..count {
                    let pos = (phase + (start + r) * self.stride) as i64;
                    for (t, &j) in taps.iter().enumerate() {
                        let source = (pos + self.padding as i64 - j as i64) / self.stride as i64;
                        if source >= 0 && (source as usize) < x.rows {
                            for i in 0..self.input {
                                patch[r * cols + i * taps.len() + t] =
                                    x.data[source as usize * x.cols + i];
                            }
                        }
                    }
                }
                result.resize(count * self.output, 0.);
                tensor::cached_linear_f32(
                    &patch,
                    &packed.weight,
                    &mut result,
                    count,
                    self.output,
                    cols,
                    packed.id,
                );
                for r in 0..count {
                    let dst = y.row_mut(phase + (start + r) * self.stride);
                    for o in 0..self.output {
                        dst[o] = result[r * self.output + o] + self.bias[o];
                    }
                }
            }
        }
        y
    }
}
fn div_ceil(a: i64, b: i64) -> i64 {
    -(-a).div_euclid(b)
}
enum Activation {
    Elu,
    Snake { alpha: Vec<f32>, beta: Vec<f32> },
    Tanh,
    Identity,
}
impl Activation {
    fn load(w: &Weights, name: &str, channels: usize, snake: bool) -> Result<Self> {
        if !snake {
            return Ok(Self::Elu);
        }
        let a = w.checked(&format!("{name}.alpha"), &[channels])?;
        let b = w.checked(&format!("{name}.beta"), &[channels])?;
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(invalid("VAE activation must be FP32"));
        }
        Ok(Self::Snake {
            alpha: a.values().into_iter().map(f32::exp).collect(),
            beta: b.values().into_iter().map(f32::exp).collect(),
        })
    }
    fn apply(&self, x: &mut Tensor) {
        match self {
            Self::Elu => {
                for v in &mut x.data {
                    if *v < 0. {
                        *v = v.exp() - 1.;
                    }
                }
            }
            Self::Snake { alpha, beta } => {
                let mut sine: Vec<f32> = x
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v * alpha[i % x.cols])
                    .collect();
                tensor::sin_slice(&mut sine);
                let inv_beta: Vec<f32> = beta.iter().map(|v| 1. / (v + 1e-9)).collect();
                let mut offset = 0;
                for row in x.data.chunks_mut(x.cols) {
                    for (i, v) in row.iter_mut().enumerate() {
                        let s = sine[offset + i];
                        *v += inv_beta[i] * (s * s);
                    }
                    offset += row.len();
                }
            }
            Self::Tanh => {
                for v in &mut x.data {
                    *v = v.tanh();
                }
            }
            Self::Identity => {}
        }
    }
}
enum Op {
    Conv(Conv),
    Activation(Activation),
    Sequence(Vec<Op>),
    Residual(Box<Op>),
}
impl Op {
    fn apply(&self, mut x: Tensor) -> Tensor {
        match self {
            Self::Conv(c) => c.apply(&x),
            Self::Activation(a) => {
                a.apply(&mut x);
                x
            }
            Self::Sequence(s) => {
                for child in s {
                    x = child.apply(x);
                }
                x
            }
            Self::Residual(child) => {
                let mut h = child.apply(x.clone());
                h.add(&x, false);
                h
            }
        }
    }
    fn length(&self, mut n: usize) -> usize {
        match self {
            Self::Conv(c) => c.length(n),
            Self::Sequence(s) => {
                for child in s {
                    n = child.length(n);
                }
                n
            }
            _ => n,
        }
    }
    fn support(&self, mut low: i64, mut high: i64) -> (i64, i64) {
        match self {
            Self::Conv(c) => c.support(low, high),
            Self::Sequence(s) => {
                for child in s.iter().rev() {
                    (low, high) = child.support(low, high);
                }
                (low, high)
            }
            Self::Residual(child) => {
                let (a, b) = child.support(low, high);
                (a.min(low), b.max(high))
            }
            _ => (low, high),
        }
    }
}
fn residual(w: &Weights, name: &str, channels: usize, dilation: usize, snake: bool) -> Result<Op> {
    Ok(Op::Residual(Box::new(Op::Sequence(vec![
        Op::Activation(Activation::load(
            w,
            &format!("{name}.layers.0"),
            channels,
            snake,
        )?),
        Op::Conv(Conv::load(
            w,
            &format!("{name}.layers.1"),
            channels,
            channels,
            7,
            1,
            3 * dilation,
            dilation,
            false,
            true,
        )?),
        Op::Activation(Activation::load(
            w,
            &format!("{name}.layers.2"),
            channels,
            snake,
        )?),
        Op::Conv(Conv::load(
            w,
            &format!("{name}.layers.3"),
            channels,
            channels,
            1,
            1,
            0,
            1,
            false,
            true,
        )?),
    ]))))
}
fn sizes(v: &Value, key: &str) -> Result<Vec<usize>> {
    v.get(key)?.array()?.iter().map(Value::usize).collect()
}
fn network(w: &Weights, name: &str, v: &Value, decoder: bool) -> Result<Op> {
    let base = v.size_or("channels", 128)?;
    let mut mults = vec![1];
    mults.extend(sizes(v, "c_mults")?);
    let strides = sizes(v, "strides")?;
    if strides.len() + 1 != mults.len() || strides.contains(&0) || mults.contains(&0) || base == 0 {
        return Err(invalid("Invalid Oobleck channel/stride configuration"));
    }
    for k in ["antialias_activation", "use_filter", "use_nearest_upsample"] {
        if v.bool_or(k, false)? {
            return Err(invalid(format!("Unsupported VAE option {k}")));
        }
    }
    let snake = v.bool_or("use_snake", false)?;
    if snake && v.text_or("snake_type", "vanilla")? != "vanilla" {
        return Err(invalid("Expected vanilla SnakeBeta"));
    }
    let latent = v.size_or("latent_dim", 32)?;
    let mut ops = Vec::new();
    if decoder {
        ops.push(Op::Conv(Conv::load(
            w,
            &format!("{name}.layers.0"),
            latent,
            base * mults[mults.len() - 1],
            7,
            1,
            3,
            1,
            false,
            true,
        )?));
        let mut index = 1;
        for i in (1..mults.len()).rev() {
            let input = base * mults[i];
            let output = base * mults[i - 1];
            let stride = strides[i - 1];
            let p = format!("{name}.layers.{index}.layers");
            ops.push(Op::Sequence(vec![
                Op::Activation(Activation::load(w, &format!("{p}.0"), input, snake)?),
                Op::Conv(Conv::load(
                    w,
                    &format!("{p}.1"),
                    input,
                    output,
                    2 * stride,
                    stride,
                    stride.div_ceil(2),
                    1,
                    true,
                    true,
                )?),
                residual(w, &format!("{p}.2"), output, 1, snake)?,
                residual(w, &format!("{p}.3"), output, 3, snake)?,
                residual(w, &format!("{p}.4"), output, 9, snake)?,
            ]));
            index += 1;
        }
        ops.push(Op::Activation(Activation::load(
            w,
            &format!("{name}.layers.{index}"),
            base,
            snake,
        )?));
        ops.push(Op::Conv(Conv::load(
            w,
            &format!("{name}.layers.{}", index + 1),
            base,
            v.size_or("out_channels", 2)?,
            7,
            1,
            3,
            1,
            false,
            false,
        )?));
        ops.push(Op::Activation(if v.bool_or("final_tanh", true)? {
            Activation::Tanh
        } else {
            Activation::Identity
        }));
    } else {
        ops.push(Op::Conv(Conv::load(
            w,
            &format!("{name}.layers.0"),
            v.size_or("in_channels", 2)?,
            base,
            7,
            1,
            3,
            1,
            false,
            true,
        )?));
        for i in 0..strides.len() {
            let input = base * mults[i];
            let output = base * mults[i + 1];
            let stride = strides[i];
            let p = format!("{name}.layers.{}.layers", i + 1);
            ops.push(Op::Sequence(vec![
                residual(w, &format!("{p}.0"), input, 1, snake)?,
                residual(w, &format!("{p}.1"), input, 3, snake)?,
                residual(w, &format!("{p}.2"), input, 9, snake)?,
                Op::Activation(Activation::load(w, &format!("{p}.3"), input, snake)?),
                Op::Conv(Conv::load(
                    w,
                    &format!("{p}.4"),
                    input,
                    output,
                    2 * stride,
                    stride,
                    stride.div_ceil(2),
                    1,
                    false,
                    true,
                )?),
            ]));
        }
        let index = strides.len() + 1;
        let input = base * mults[mults.len() - 1];
        ops.push(Op::Activation(Activation::load(
            w,
            &format!("{name}.layers.{index}"),
            input,
            snake,
        )?));
        ops.push(Op::Conv(Conv::load(
            w,
            &format!("{name}.layers.{}", index + 1),
            input,
            latent,
            3,
            1,
            1,
            1,
            false,
            true,
        )?));
    }
    Ok(Op::Sequence(ops))
}
pub struct Vae {
    decoder: Op,
    encoder: Option<Op>,
    pub sample_rate: usize,
    pub latent_dim: usize,
    pub audio_channels: usize,
    pub downsampling_ratio: usize,
    pub core_frames: usize,
    pub halo_frames: usize,
}
impl Drop for Vae {
    fn drop(&mut self) {
        tensor::clear_gpu_weight_cache();
    }
}
impl Vae {
    pub fn open(path: impl AsRef<Path>, decoder_only: bool) -> Result<Self> {
        let path = path.as_ref();
        let v = json::read(path.join("config.json"))?;
        if v.text_or("model_type", "yue2_vae")? != "yue2_vae" {
            return Err(invalid("Expected YuE2 VAE checkpoint"));
        }
        let w = Weights::open(path)?;
        let decoder_config = v.get("decoder_config")?;
        let downsampling_ratio = v.size_or("downsampling_ratio", 1920)?;
        let latent_dim = v.size_or("latent_dim", 64)?;
        let sample_rate = v.size_or("sample_rate", 48000)?;
        let audio_channels = v.size_or("audio_channels", 2)?;
        if sizes(decoder_config, "strides")?.iter().product::<usize>() != downsampling_ratio
            || decoder_config.size_or("latent_dim", 32)? != latent_dim
            || decoder_config.size_or("out_channels", 2)? != audio_channels
            || sample_rate == 0
        {
            return Err(invalid("VAE configuration mismatch"));
        }
        Ok(Self {
            decoder: network(&w, "decoder", decoder_config, true)?,
            encoder: if decoder_only {
                None
            } else {
                Some(network(&w, "encoder", v.get("encoder_config")?, false)?)
            },
            sample_rate,
            latent_dim,
            audio_channels,
            downsampling_ratio,
            core_frames: v.size_or("decode_core_frames", 1024)?,
            halo_frames: v.size_or("decode_halo_frames", 16)?,
        })
    }
    pub fn natural_output_length(&self, frames: usize) -> Result<usize> {
        if frames == 0 {
            return Err(invalid("Empty VAE latent"));
        }
        Ok(self.decoder.length(frames))
    }
    pub fn required_halo(&self, core: usize) -> usize {
        let (low, high) = self
            .decoder
            .support(0, (core * self.downsampling_ratio - 1) as i64);
        0i64.max(-low).max(high - core as i64 + 1) as usize
    }
    fn validate_latent(&self, z: &Tensor) -> Result<()> {
        if z.rows == 0 || z.cols != self.latent_dim || !z.finite() {
            return Err(invalid("Expected finite nonempty [frames,64] latents"));
        }
        Ok(())
    }
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        self.validate_latent(z)?;
        let out = self.decoder.apply(z.clone());
        if !out.finite() {
            return Err(invalid("VAE produced nonfinite audio"));
        }
        Ok(out)
    }
    pub fn decode_tiled(
        &self,
        z: &Tensor,
        core_frames: usize,
        halo_frames: usize,
    ) -> Result<Tensor> {
        self.validate_latent(z)?;
        if core_frames == 0 || halo_frames < self.required_halo(core_frames) {
            return Err(invalid(format!(
                "VAE halo must be at least {}",
                self.required_halo(core_frames.max(1))
            )));
        }
        let length = self.natural_output_length(z.rows)?;
        let mut out = Tensor::zeros(length, self.audio_channels);
        let total = z.rows.div_ceil(core_frames);
        for (index, start) in (0..z.rows).step_by(core_frames).enumerate() {
            let end = (start + core_frames).min(z.rows);
            let left = start.saturating_sub(halo_frames);
            let right = (end + halo_frames).min(z.rows);
            eprintln!("VAE decoding chunk {}/{}", index + 1, total);
            let tile = self.decode(&z.slice(left, right))?;
            let a = start * self.downsampling_ratio;
            let b = (end * self.downsampling_ratio).min(length);
            let crop = (start - left) * self.downsampling_ratio;
            if crop + b - a > tile.rows {
                return Err(invalid("VAE tile did not cover output core"));
            }
            out.data[a * self.audio_channels..b * self.audio_channels].copy_from_slice(
                &tile.data[crop * self.audio_channels..(crop + b - a) * self.audio_channels],
            );
        }
        Ok(out)
    }
    pub fn encode(&self, audio: &Tensor, sample: bool, seed: u64) -> Result<Tensor> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| invalid("Encoder was not loaded"))?;
        if audio.cols != self.audio_channels
            || audio.rows < self.downsampling_ratio
            || !audio.finite()
        {
            return Err(invalid(
                "Expected finite stereo audio with at least one latent frame",
            ));
        }
        let pre = encoder.apply(audio.clone());
        if pre.cols != 2 * self.latent_dim {
            return Err(invalid("VAE posterior dimension mismatch"));
        }
        let noise = if sample {
            Some(crate::sampling::Generator::new(seed).normal_tensor(pre.rows, self.latent_dim))
        } else {
            None
        };
        let mut z = Tensor::zeros(pre.rows, self.latent_dim);
        for r in 0..pre.rows {
            for c in 0..self.latent_dim {
                let mean = pre.data[r * pre.cols + c];
                z.data[r * z.cols + c] = if let Some(n) = &noise {
                    let scale = pre.data[r * pre.cols + self.latent_dim + c];
                    let sd = if scale > 20. {
                        scale
                    } else {
                        scale.exp().ln_1p()
                    };
                    mean + n.data[r * z.cols + c] * (sd + 1e-4)
                } else {
                    mean
                };
            }
        }
        Ok(z)
    }
}
