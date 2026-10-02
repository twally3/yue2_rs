//! Row-major FP32 compute with checkpoint-native BF16 rounding and Apple BLAS.
use crate::{Result, checkpoint::Weight, invalid};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, c_char},
    sync::{Arc, Mutex, OnceLock},
};
unsafe extern "C" {
    fn vDSP_vsmul(
        a: *const f32,
        stride_a: isize,
        b: *const f32,
        c: *mut f32,
        stride_c: isize,
        n: usize,
    );
    fn vDSP_vsadd(
        a: *const f32,
        stride_a: isize,
        b: *const f32,
        c: *mut f32,
        stride_c: isize,
        n: usize,
    );
    fn vDSP_vsdiv(
        a: *const f32,
        stride_a: isize,
        b: *const f32,
        c: *mut f32,
        stride_c: isize,
        n: usize,
    );
    fn vDSP_maxv(a: *const f32, stride: isize, c: *mut f32, n: usize);
    fn vDSP_sve(a: *const f32, stride: isize, c: *mut f32, n: usize);
    fn yue2_metal_available() -> i32;
    fn yue2_metal_vector_default() -> i32;
    fn yue2_metal_vector(
        input: *const f32,
        raw: *const u8,
        mapping: usize,
        dtype: i32,
        rounded: i32,
        priority: i32,
        out: *mut f32,
        rows: usize,
        cols: usize,
    ) -> i32;
    fn yue2_physical_memory() -> u64;
    fn yue2_metal_clear_weights();
    fn yue2_metal_release_model();
    fn yue2_metal_linear(
        a: *const f32,
        weight: *const u8,
        mapping: usize,
        dtype: i32,
        round: i32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> i32;
    fn yue2_metal_ar_stack(
        input: *const f32,
        out: *mut f32,
        shadow: *mut f32,
        layers: *const GpuLayer,
        count: usize,
        hidden: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        intermediate: usize,
        prefix: usize,
        cache: usize,
        eps: f32,
        cos: *const f32,
        sin: *const f32,
    ) -> i32;
    fn yue2_metal_stack(
        x: *const f32,
        out: *mut f32,
        layers: *const GpuLayer,
        count: usize,
        rows: usize,
        hidden: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        intermediate: usize,
        prefix: usize,
        cache: usize,
        eps: f32,
        cos: *const f32,
        sin: *const f32,
    ) -> i32;
    fn yue2_metal_cached_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        keys: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        cache: usize,
        layer: usize,
    ) -> i32;
    fn yue2_metal_name() -> *const c_char;
    fn yue2_metal_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        rows: usize,
        keys: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        causal: i64,
        scale: f32,
    ) -> i32;
    fn yue2_metal_gemm(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: i32,
        lda: usize,
        ldb: usize,
        ldc: usize,
        beta: f32,
    ) -> i32;
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GpuWeight {
    raw: *const u8,
    mapping: usize,
    rows: usize,
    cols: usize,
    dtype: i32,
    rounded: i32,
}
#[repr(C)]
pub(crate) struct GpuLayer {
    pub weights: [GpuWeight; 7],
    pub norms: [*const f32; 4],
    pub key: *const f32,
    pub value: *const f32,
}
pub(crate) fn gpu_stack(
    x: &Tensor,
    layers: &[GpuLayer],
    heads: usize,
    kv: usize,
    dim: usize,
    intermediate: usize,
    prefix: usize,
    cache: usize,
    eps: f32,
    cos: &Tensor,
    sin: &Tensor,
) -> Option<Tensor> {
    if x.rows < 64
        || !metal_available()
        || std::env::var("YUE2_NAR").as_deref() == Ok("cpu")
        || std::env::var("YUE2_ATTENTION").as_deref() == Ok("cpu")
        || !dim.is_multiple_of(32)
        || !x.cols.is_multiple_of(16)
        || !intermediate.is_multiple_of(16)
    {
        return None;
    }
    for width in [
        x.cols,
        heads.checked_mul(dim)?,
        kv.checked_mul(dim)?,
        intermediate,
    ] {
        if x.rows.checked_mul(width)? > u32::MAX as usize {
            return None;
        }
    }
    if prefix.checked_add(x.rows)? > i32::MAX as usize {
        return None;
    }
    let mut out = Tensor::zeros(x.rows, x.cols);
    let ok = unsafe {
        yue2_metal_stack(
            x.data.as_ptr(),
            out.data.as_mut_ptr(),
            layers.as_ptr(),
            layers.len(),
            x.rows,
            x.cols,
            heads,
            kv,
            dim,
            intermediate,
            prefix,
            cache,
            eps,
            cos.data.as_ptr(),
            sin.data.as_ptr(),
        )
    };
    (ok != 0).then_some(out)
}
pub(crate) fn gpu_ar_stack(
    x: &Tensor,
    layers: &[GpuLayer],
    heads: usize,
    kv: usize,
    dim: usize,
    intermediate: usize,
    prefix: usize,
    cache: usize,
    eps: f32,
    cos: &Tensor,
    sin: &Tensor,
) -> Option<(Tensor, Vec<f32>)> {
    if x.rows != 1
        || !(255..4096).contains(&prefix)
        || dim > 256
        || !dim.is_multiple_of(32)
        || !metal_available()
        || !vector_backend_enabled()
        || std::env::var("YUE2_ATTENTION").as_deref() == Ok("cpu")
    {
        return None;
    }
    let mut out = Tensor::zeros(1, x.cols);
    let mut shadow = vec![0.; layers.len() * 2 * kv * dim];
    let ok = unsafe {
        yue2_metal_ar_stack(
            x.data.as_ptr(),
            out.data.as_mut_ptr(),
            shadow.as_mut_ptr(),
            layers.as_ptr(),
            layers.len(),
            x.cols,
            heads,
            kv,
            dim,
            intermediate,
            prefix,
            cache,
            eps,
            cos.data.as_ptr(),
            sin.data.as_ptr(),
        )
    };
    (ok != 0).then_some((out, shadow))
}
type WeightKey = (usize, usize, usize, bool);
struct ConvertedWeight {
    // Retain the mapping so an address cannot be reused for a different file.
    _owner: Weight,
    values: Arc<Vec<f32>>,
}
struct WeightCache {
    values: HashMap<WeightKey, ConvertedWeight>,
    bytes: usize,
    budget: usize,
}
fn weight_cache() -> &'static Mutex<WeightCache> {
    static CACHE: OnceLock<Mutex<WeightCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let physical = unsafe { yue2_physical_memory() } as usize;
        let gib = 1usize << 30;
        let automatic = if physical < 12 * gib {
            physical / 4
        } else {
            physical.saturating_sub(2 * gib) / 2
        }
        .min(8 * gib);
        let budget = std::env::var("YUE2_CPU_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .and_then(|v| v.checked_mul(1 << 20))
            .unwrap_or(automatic)
            .min(physical.saturating_mul(2) / 3);
        Mutex::new(WeightCache {
            values: HashMap::new(),
            bytes: 0,
            budget,
        })
    })
}
pub fn clear_weight_cache() {
    let mut cache = weight_cache().lock().unwrap_or_else(|e| e.into_inner());
    cache.values.clear();
    cache.bytes = 0;
}
pub fn clear_gpu_weight_cache() {
    if metal_available() {
        unsafe {
            yue2_metal_clear_weights();
        }
    }
}
pub(crate) fn release_gpu_model() {
    if metal_available() {
        unsafe {
            yue2_metal_release_model();
        }
    }
}
pub fn weight_cache_bytes() -> usize {
    weight_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .bytes
}
fn converted_weight(
    weight: &Weight,
    bf16: bool,
    begin: usize,
    end: usize,
) -> Option<Arc<Vec<f32>>> {
    let (map, offset, _) = weight.cache_identity();
    let count = end - begin;
    let width = if weight.dtype == crate::checkpoint::DType::F32 {
        4
    } else {
        2
    };
    let key = (map, offset + begin * width, count, bf16);
    let mut cache = weight_cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(value) = cache.values.get(&key) {
        return Some(value.values.clone());
    }
    let bytes = count.checked_mul(4)?;
    // Keep admitted matrices rather than cycling an LRU through every layer.
    if bytes > cache.budget.saturating_sub(cache.bytes) {
        return None;
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).ok()?;
    values.resize(count, 0.);
    weight.copy_range(begin, end, &mut values);
    if bf16 && weight.dtype != crate::checkpoint::DType::BF16 {
        round_slice(&mut values);
    }
    let values = Arc::new(values);
    cache.values.insert(
        key,
        ConvertedWeight {
            _owner: weight.clone(),
            values: values.clone(),
        },
    );
    cache.bytes += bytes;
    Some(values)
}
thread_local! {
    static WEIGHT_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static ACTIVATION_SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}
pub fn metal_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::env::var("YUE2_BACKEND").as_deref() != Ok("cpu")
            && unsafe { yue2_metal_available() != 0 }
    })
}
pub fn backend_name() -> String {
    if metal_available() {
        let name = unsafe { CStr::from_ptr(yue2_metal_name()) }.to_string_lossy();
        format!("Metal MPS ({name}) + Accelerate")
    } else {
        "Accelerate CPU".into()
    }
}
fn vector_backend_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("YUE2_AR").as_deref() {
        Ok("metal") => true,
        Ok("cpu") => false,
        _ => unsafe { yue2_metal_vector_default() != 0 },
    })
}
pub(crate) fn gpu_cached_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    kv: usize,
    dim: usize,
    cache: usize,
    layer: usize,
) -> Option<Tensor> {
    if q.rows != 1
        || !(256..=4096).contains(&k.rows)
        || dim > 256
        || !metal_available()
        || !vector_backend_enabled()
        || std::env::var("YUE2_ATTENTION").as_deref() == Ok("cpu")
    {
        return None;
    }
    let mut out = Tensor::zeros(1, q.cols);
    let ok = unsafe {
        yue2_metal_cached_attention(
            q.data.as_ptr(),
            k.data.as_ptr(),
            v.data.as_ptr(),
            out.data.as_mut_ptr(),
            k.rows,
            heads,
            kv,
            dim,
            cache,
            layer,
        )
    };
    (ok != 0).then_some(out)
}
pub fn gpu_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    kv: usize,
    dim: usize,
    causal: Option<usize>,
) -> Option<Tensor> {
    if q.rows < 64 || !metal_available() || std::env::var("YUE2_ATTENTION").as_deref() == Ok("cpu")
    {
        return None;
    }
    assert!(heads > 0 && kv > 0 && heads.is_multiple_of(kv));
    if !dim.is_multiple_of(16) {
        return None;
    }
    assert_eq!(q.cols, heads * dim);
    assert_eq!((k.cols, v.rows, v.cols), (kv * dim, k.rows, k.cols));
    assert_eq!(q.data.len(), q.rows * q.cols);
    assert_eq!(k.data.len(), k.rows * k.cols);
    assert_eq!(v.data.len(), v.rows * v.cols);
    let mut out = Tensor::zeros(q.rows, q.cols);
    let ok = unsafe {
        yue2_metal_attention(
            q.data.as_ptr(),
            k.data.as_ptr(),
            v.data.as_ptr(),
            out.data.as_mut_ptr(),
            q.rows,
            k.rows,
            heads,
            kv,
            dim,
            causal.map_or(-1, |c| c as i64),
            (dim as f32).sqrt().recip(),
        )
    };
    (ok != 0).then_some(out)
}

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvexpf(out: *mut f32, input: *const f32, count: *const i32);
    fn vvsinf(out: *mut f32, input: *const f32, count: *const i32);
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        inc_x: i32,
        beta: f32,
        y: *mut f32,
        inc_y: i32,
    );
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}
#[derive(Clone, Debug)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}
impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.; rows.checked_mul(cols).expect("Tensor size overflow")],
        }
    }
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(invalid("Tensor shape mismatch"));
        }
        Ok(Self { rows, cols, data })
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.rows);
        Self {
            rows: end - start,
            cols: self.cols,
            data: self.data[start * self.cols..end * self.cols].to_vec(),
        }
    }
    pub fn append(&mut self, other: &Self) {
        assert_eq!(self.cols, other.cols);
        self.data.extend_from_slice(&other.data);
        self.rows += other.rows;
    }
    pub fn rounded(mut self) -> Self {
        round_slice(&mut self.data);
        self
    }
    pub fn add(&mut self, other: &Self, bf16: bool) {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
            if bf16 {
                *a = round_bf16(*a);
            }
        }
    }
    pub fn finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}
#[inline]
pub fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    if b & 0x7fffffff > 0x7f800000 {
        return f32::from_bits((b & 0xffff0000) | 0x00400000);
    }
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
pub fn round_slice(v: &mut [f32]) {
    for x in v {
        *x = round_bf16(*x);
    }
}
pub fn silu(v: f32) -> f32 {
    v / (1. + (-v).exp())
}
pub fn silu_slice(values: &mut [f32], bf16: bool) {
    ACTIVATION_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let (input, exponential) = &mut *scratch;
        for chunk in values.chunks_mut(i32::MAX as usize) {
            input.resize(chunk.len(), 0.);
            exponential.resize(chunk.len(), 0.);
            for (dst, src) in input.iter_mut().zip(chunk.iter()) {
                *dst = -*src;
            }
            unsafe {
                vvexpf(
                    exponential.as_mut_ptr(),
                    input.as_ptr(),
                    &(chunk.len() as i32),
                );
            }
            for (value, e) in chunk.iter_mut().zip(exponential.iter()) {
                *value /= 1. + e;
                if bf16 {
                    *value = round_bf16(*value);
                }
            }
        }
    });
}
pub fn sin_slice(values: &mut [f32]) {
    for chunk in values.chunks_mut(i32::MAX as usize) {
        let input = chunk.to_vec();
        unsafe {
            vvsinf(chunk.as_mut_ptr(), input.as_ptr(), &(chunk.len() as i32));
        }
    }
}
pub fn exp_to(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (src, dst) in input
        .chunks(i32::MAX as usize)
        .zip(output.chunks_mut(i32::MAX as usize))
    {
        unsafe {
            vvexpf(dst.as_mut_ptr(), src.as_ptr(), &(src.len() as i32));
        }
    }
}
pub fn softmax_row(row: &mut [f32], scratch: &mut [f32], visible: usize, scale: f32) {
    assert!(visible > 0 && visible <= row.len() && visible <= scratch.len());
    row[visible..].fill(0.);
    let row = &mut row[..visible];
    let scratch = &mut scratch[..visible];
    let p = row.as_mut_ptr();
    let mut max = 0.;
    let mut sum = 0.;
    unsafe {
        vDSP_vsmul(p, 1, &scale, p, 1, visible);
        vDSP_maxv(p, 1, &mut max, visible);
        vDSP_vsadd(p, 1, &-max, p, 1, visible);
    }
    exp_to(row, scratch);
    unsafe {
        vDSP_sve(scratch.as_ptr(), 1, &mut sum, visible);
        vDSP_vsdiv(scratch.as_ptr(), 1, &sum, p, 1, visible);
    }
}
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
    beta: f32,
) {
    gemm_impl(a, b, c, m, n, k, transpose_b, lda, ldb, ldc, beta, true);
}
pub(crate) fn cached_linear_f32(
    input: &[f32],
    weight: &[f32],
    out: &mut [f32],
    rows: usize,
    outputs: usize,
    cols: usize,
    id: usize,
) {
    assert_eq!(input.len(), rows * cols);
    assert_eq!(weight.len(), outputs * cols);
    assert!(out.len() >= rows * outputs);
    if rows >= 32
        && rows.saturating_mul(outputs).saturating_mul(cols) >= 32_000_000
        && metal_available()
    {
        let ok = unsafe {
            yue2_metal_linear(
                input.as_ptr(),
                weight.as_ptr().cast(),
                id,
                1,
                0,
                out.as_mut_ptr(),
                rows,
                outputs,
                cols,
            )
        };
        if ok != 0 {
            return;
        }
    }
    gemm(
        input, weight, out, rows, outputs, cols, true, cols, cols, outputs, 0.,
    );
}
pub fn gemm_cpu(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
    beta: f32,
) {
    gemm_impl(a, b, c, m, n, k, transpose_b, lda, ldb, ldc, beta, false);
}
fn gemm_impl(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
    beta: f32,
    use_metal: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize);
    assert!(k > 0);
    assert!(lda <= i32::MAX as usize && ldb <= i32::MAX as usize && ldc <= i32::MAX as usize);
    assert!(lda >= k && ldb >= if transpose_b { k } else { n } && ldc >= n);
    assert!(a.len() >= (m - 1) * lda + k);
    assert!(
        b.len()
            >= if transpose_b {
                (n - 1) * ldb + k
            } else {
                (k - 1) * ldb + n
            }
    );
    assert!(c.len() >= (m - 1) * ldc + n);
    if use_metal
        && m >= 32
        && m.saturating_mul(n).saturating_mul(k) >= 32_000_000
        && metal_available()
    {
        let ok = unsafe {
            yue2_metal_gemm(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                i32::from(transpose_b),
                lda,
                ldb,
                ldc,
                beta,
            )
        };
        if ok != 0 {
            return;
        }
    }
    if m == 1 {
        unsafe {
            cblas_sgemv(
                101,
                if transpose_b { 111 } else { 112 },
                if transpose_b { n } else { k } as i32,
                if transpose_b { k } else { n } as i32,
                1.,
                b.as_ptr(),
                ldb as i32,
                a.as_ptr(),
                1,
                beta,
                c.as_mut_ptr(),
                1,
            );
        }
        return;
    }
    unsafe {
        cblas_sgemm(
            101,
            111,
            if transpose_b { 112 } else { 111 },
            m as i32,
            n as i32,
            k as i32,
            1.,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            beta,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
}
#[derive(Clone)]
pub struct Linear {
    pub weight: Weight,
    pub bias: Option<Vec<f32>>,
    pub bf16: bool,
    pub(crate) vector_backend: u8,
}
impl Linear {
    pub fn new(weight: Weight, bias: Option<Weight>, bf16: bool) -> Result<Self> {
        if weight.shape.len() != 2 {
            return Err(invalid("Expected matrix weight"));
        }
        if let Some(ref b) = bias {
            b.expect(&[weight.shape[0]], "linear bias")?;
        }
        Ok(Self {
            weight,
            bias: bias.map(|b| b.values()),
            bf16,
            vector_backend: 0,
        })
    }
    pub(crate) fn gpu_weight(&self) -> Option<GpuWeight> {
        if self.bias.is_some() || !self.bf16 {
            return None;
        }
        Some(GpuWeight {
            raw: self.weight.raw_bytes().as_ptr(),
            mapping: self.weight.cache_identity().0,
            rows: self.weight.shape[0],
            cols: self.weight.shape[1],
            dtype: match self.weight.dtype {
                crate::checkpoint::DType::BF16 => 0,
                crate::checkpoint::DType::F32 => 1,
                crate::checkpoint::DType::F16 => 2,
            },
            rounded: 1,
        })
    }
    pub(crate) fn clear_cached(&self) {
        let (map, offset, count) = self.weight.cache_identity();
        let width = if self.weight.dtype == crate::checkpoint::DType::F32 {
            4
        } else {
            2
        };
        let end = offset + count * width;
        let mut cache = weight_cache().lock().unwrap_or_else(|e| e.into_inner());
        let mut removed = 0;
        cache.values.retain(|key, value| {
            let keep = key.0 != map || key.1 < offset || key.1 >= end;
            if !keep {
                removed += value.values.len() * 4;
            }
            keep
        });
        cache.bytes -= removed;
    }
    pub fn apply(&self, x: &Tensor) -> Tensor {
        self.apply_range(x, 0, self.weight.shape[0])
    }
    pub fn apply_range(&self, x: &Tensor, begin: usize, end: usize) -> Tensor {
        assert!(begin < end && end <= self.weight.shape[0]);
        let out = end - begin;
        let input = self.weight.shape[1];
        assert!(
            out <= i32::MAX as usize && input <= i32::MAX as usize && x.rows <= i32::MAX as usize
        );
        assert_eq!(x.cols, input);
        assert_eq!(
            x.data.len(),
            x.rows.checked_mul(x.cols).expect("Tensor size overflow")
        );
        let mut y = Tensor::zeros(x.rows, out);
        if self.vector_backend != 0
            && x.rows == 1
            && out.saturating_mul(input) >= 1_000_000
            && metal_available()
        {
            if vector_backend_enabled() {
                let dtype = match self.weight.dtype {
                    crate::checkpoint::DType::BF16 => 0,
                    crate::checkpoint::DType::F32 => 1,
                    crate::checkpoint::DType::F16 => 2,
                };
                let width = if dtype == 1 { 4 } else { 2 };
                let raw = &self.weight.raw_bytes()[begin * input * width..end * input * width];
                let ok = unsafe {
                    yue2_metal_vector(
                        x.data.as_ptr(),
                        raw.as_ptr(),
                        self.weight.cache_identity().0,
                        dtype,
                        i32::from(self.bf16),
                        self.vector_backend as i32,
                        y.data.as_mut_ptr(),
                        out,
                        input,
                    )
                };
                if ok != 0 {
                    self.finish(&mut y, begin);
                    return y;
                }
            }
        }
        if x.rows >= 32
            && x.rows.saturating_mul(out).saturating_mul(input) >= 32_000_000
            && metal_available()
        {
            let dtype = match self.weight.dtype {
                crate::checkpoint::DType::BF16 => 0,
                crate::checkpoint::DType::F32 => 1,
                crate::checkpoint::DType::F16 => 2,
            };
            let width = if dtype == 1 { 4 } else { 2 };
            let raw = &self.weight.raw_bytes()[begin * input * width..end * input * width];
            let ok = unsafe {
                yue2_metal_linear(
                    x.data.as_ptr(),
                    raw.as_ptr(),
                    self.weight.cache_identity().0,
                    dtype,
                    i32::from(self.bf16),
                    y.data.as_mut_ptr(),
                    x.rows,
                    out,
                    input,
                )
            };
            if ok != 0 {
                self.finish(&mut y, begin);
                return y;
            }
        }
        let cached = if x.rows == 1 {
            converted_weight(&self.weight, self.bf16, begin * input, end * input)
        } else {
            None
        };
        if let Some(ref weight) = cached {
            unsafe {
                cblas_sgemv(
                    101,
                    111,
                    out as i32,
                    input as i32,
                    1.,
                    weight.as_ptr(),
                    input as i32,
                    x.data.as_ptr(),
                    1,
                    0.,
                    y.data.as_mut_ptr(),
                    1,
                );
            }
            self.finish(&mut y, begin);
            return y;
        }
        // Conversion panels bound RAM even for the 184704 x 2048 vocabulary.
        let panel = if x.rows >= 32 && metal_available() {
            1024
        } else {
            256
        };
        WEIGHT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            for start in (0..out).step_by(panel) {
                let end = (start + panel).min(out);
                scratch.resize((end - start) * input, 0.);
                self.weight.copy_range(
                    (begin + start) * input,
                    (begin + end) * input,
                    &mut scratch,
                );
                if self.bf16 && self.weight.dtype != crate::checkpoint::DType::BF16 {
                    round_slice(&mut scratch);
                }
                let w = &scratch[..];
                if x.rows == 1 {
                    unsafe {
                        cblas_sgemv(
                            101,
                            111,
                            (end - start) as i32,
                            input as i32,
                            1.,
                            w.as_ptr(),
                            input as i32,
                            x.data.as_ptr(),
                            1,
                            0.,
                            y.data[start..].as_mut_ptr(),
                            1,
                        );
                    }
                } else {
                    gemm(
                        &x.data,
                        w,
                        &mut y.data[start..],
                        x.rows,
                        end - start,
                        input,
                        true,
                        input,
                        input,
                        out,
                        0.,
                    );
                }
            }
        });
        self.finish(&mut y, begin);
        y
    }
    fn finish(&self, y: &mut Tensor, begin: usize) {
        let out = y.cols;
        for row in y.data.chunks_mut(out) {
            for (i, v) in row.iter_mut().enumerate() {
                if let Some(b) = &self.bias {
                    *v += if self.bf16 {
                        round_bf16(b[begin + i])
                    } else {
                        b[begin + i]
                    };
                }
                if self.bf16 {
                    *v = round_bf16(*v);
                }
            }
        }
    }
}
pub struct Norm {
    weight: Vec<f32>,
    eps: f32,
    pub bf16: bool,
}
impl Norm {
    pub fn new(weight: Weight, eps: f32, bf16: bool) -> Self {
        let mut values = weight.values();
        if bf16 {
            round_slice(&mut values);
        }
        Self {
            weight: values,
            eps,
            bf16,
        }
    }
    pub(crate) fn gpu_values(&self) -> *const f32 {
        self.weight.as_ptr()
    }
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let mut y = x.clone();
        for row in y.data.chunks_mut(x.cols) {
            assert_eq!(row.len(), self.weight.len());
            let scale = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + self.eps)
                .sqrt()
                .recip();
            let scale = if self.bf16 { round_bf16(scale) } else { scale };
            for (v, w) in row.iter_mut().zip(&self.weight) {
                if self.bf16 {
                    *v = round_bf16(round_bf16(*v * scale) * *w);
                } else {
                    *v *= scale * w;
                }
            }
        }
        y
    }
    pub fn heads(&self, x: &mut Tensor, dim: usize) {
        for head in x.data.chunks_mut(dim) {
            let scale = (head.iter().map(|v| v * v).sum::<f32>() / dim as f32 + self.eps)
                .sqrt()
                .recip();
            let scale = if self.bf16 { round_bf16(scale) } else { scale };
            for (v, w) in head.iter_mut().zip(&self.weight) {
                *v = if self.bf16 {
                    round_bf16(round_bf16(*v * scale) * *w)
                } else {
                    *v * scale * w
                };
            }
        }
    }
}
