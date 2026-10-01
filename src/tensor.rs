//! Row-major FP32 compute with checkpoint-native BF16 rounding and Apple BLAS.
use crate::{Result, checkpoint::Weight, invalid};
use std::{
    ffi::{CStr, c_char},
    sync::OnceLock,
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
    fn yue2_metal_name() -> *const c_char;
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
    for chunk in values.chunks_mut(i32::MAX as usize) {
        let input: Vec<f32> = chunk.iter().map(|v| -*v).collect();
        let mut exp = vec![0.; chunk.len()];
        unsafe {
            vvexpf(exp.as_mut_ptr(), input.as_ptr(), &(chunk.len() as i32));
        }
        for (value, e) in chunk.iter_mut().zip(exp) {
            *value /= 1. + e;
            if bf16 {
                *value = round_bf16(*value);
            }
        }
    }
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
        })
    }
    pub fn apply(&self, x: &Tensor) -> Tensor {
        self.apply_range(x, 0, self.weight.shape[0])
    }
    pub fn apply_range(&self, x: &Tensor, begin: usize, end: usize) -> Tensor {
        assert!(begin < end && end <= self.weight.shape[0]);
        let out = end - begin;
        let input = self.weight.shape[1];
        assert_eq!(x.cols, input);
        assert_eq!(
            x.data.len(),
            x.rows.checked_mul(x.cols).expect("Tensor size overflow")
        );
        let mut y = Tensor::zeros(x.rows, out);
        // Conversion panels bound RAM even for the 184704 x 2048 vocabulary.
        let panel = if x.rows >= 32 && metal_available() {
            1024
        } else {
            256
        };
        for start in (0..out).step_by(panel) {
            let end = (start + panel).min(out);
            let mut w = vec![0.; (end - start) * input];
            self.weight
                .copy_range((begin + start) * input, (begin + end) * input, &mut w);
            if self.bf16 && self.weight.dtype != crate::checkpoint::DType::BF16 {
                round_slice(&mut w);
            }
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
                    &w,
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
        y
    }
}
pub struct Norm {
    weight: Vec<f32>,
    eps: f32,
    pub bf16: bool,
}
impl Norm {
    pub fn new(weight: Weight, eps: f32, bf16: bool) -> Self {
        Self {
            weight: weight.values(),
            eps,
            bf16,
        }
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
                    *v = round_bf16(round_bf16(*v * scale) * round_bf16(*w));
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
                    round_bf16(round_bf16(*v * scale) * round_bf16(*w))
                } else {
                    *v * scale * w
                };
            }
        }
    }
}
