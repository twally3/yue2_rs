//! Memory-mapped safetensors; no model code is executed when loading a checkpoint.
use crate::{Result, invalid, json};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    fs::File,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Resolve a local export or download checkpoint data using macOS's system curl.
pub fn resolve_model(model: &Path, revision: Option<&str>) -> Result<PathBuf> {
    if model.is_dir() {
        return Ok(model.canonicalize()?);
    }
    let id = model
        .to_str()
        .ok_or_else(|| invalid("Model identifier is not UTF-8"))?;
    if model.is_absolute() || id.starts_with('.') {
        return Err(invalid(format!(
            "Missing model directory {}",
            model.display()
        )));
    }
    let parts: Vec<_> = id.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|s| {
            s.is_empty()
                || *s == "."
                || *s == ".."
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        })
    {
        return Err(invalid(
            "Use a local checkpoint directory or owner/model Hub identifier",
        ));
    }
    let short = PathBuf::from("models").join(parts[1]);
    if parts[0] == "m-a-p"
        && short.join("config.json").is_file()
        && short.join("model.safetensors").is_file()
    {
        let saved = std::fs::read_to_string(short.join("revision.txt")).unwrap_or_default();
        if revision.is_none() || revision.is_some_and(|r| r == saved.trim()) {
            return Ok(short.canonicalize()?);
        }
    }
    fn encode(s: &str) -> String {
        s.bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') {
                    (b as char).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect()
    }
    let rev = encode(revision.unwrap_or("main"));
    let cache = PathBuf::from("models").join(format!("{}--{}", parts[0], parts[1]));
    let reference = cache.join("refs").join(&rev);
    let is_commit = |s: &str| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit());
    let saved = if revision.is_some_and(is_commit) {
        revision.unwrap().to_owned()
    } else {
        std::fs::read_to_string(&reference)
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    if is_commit(&saved) {
        let local = cache.join(&saved);
        if local.join("config.json").is_file()
            && local.join("revision.txt").is_file()
            && (local.join("model.safetensors").is_file()
                || local.join("model.safetensors.index.json").is_file())
        {
            return Ok(local.canonicalize()?);
        }
    }
    let url = format!("https://huggingface.co/api/models/{id}/revision/{rev}");
    let output = std::process::Command::new("/usr/bin/curl")
        .args(["--fail", "--location", "--silent", "--show-error", &url])
        .output()?;
    if !output.status.success() {
        return Err(invalid(format!(
            "Unable to resolve {id}: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let metadata = json::parse(std::str::from_utf8(&output.stdout)?)?;
    let commit = metadata.get("sha")?.string()?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("Invalid model commit identity"));
    }
    let directory = cache.join(commit);
    std::fs::create_dir_all(&directory)?;
    for value in metadata.get("siblings")?.array()? {
        let name = value.get("rfilename")?.string()?;
        let component = Path::new(name);
        let ordinary = component.components().count() == 1
            && ([
                "config.json",
                "qwen.tiktoken",
                "yue2_generation_config.json",
                "weights_manifest.json",
                "model.safetensors.index.json",
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
            ]
            .contains(&name)
                || name.ends_with(".safetensors"));
        let license = [
            "licenses/SnakeBeta-NVIDIA-MIT.txt",
            "licenses/stable-audio-tools-MIT.txt",
        ]
        .contains(&name);
        if !ordinary && !license {
            continue;
        }
        let target = directory.join(name);
        if target.exists() {
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = target.with_extension("part");
        let url = format!("https://huggingface.co/{id}/resolve/{commit}/{name}");
        eprintln!("Downloading {id}/{name}");
        let status = std::process::Command::new("/usr/bin/curl")
            .args([
                "--fail",
                "--location",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
            ])
            .arg(&temporary)
            .arg(url)
            .status()?;
        if !status.success() {
            return Err(invalid(format!(
                "Checkpoint download failed for {name}; the partial file is retained for resuming"
            )));
        }
        std::fs::rename(temporary, target)?;
    }
    std::fs::write(directory.join("revision.txt"), format!("{commit}\n"))?;
    std::fs::create_dir_all(reference.parent().unwrap())?;
    std::fs::write(reference, format!("{commit}\n"))?;
    Ok(directory.canonicalize()?)
}

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> i32;
}
pub(crate) fn resource_id() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}
struct Mapping {
    ptr: *mut c_void,
    len: usize,
    id: usize,
}
// Mappings are immutable and remain alive until every tensor view has dropped.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}
impl Mapping {
    fn open(path: &Path) -> Result<Self> {
        let f = File::open(path)?;
        let len = usize::try_from(f.metadata()?.len())?;
        if len < 8 {
            return Err(invalid("Short safetensors file"));
        }
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, f.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self {
            ptr,
            len,
            id: resource_id(),
        })
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr, self.len);
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    BF16,
    F16,
}
impl DType {
    fn width(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }
}
#[derive(Clone)]
pub struct Weight {
    map: Arc<Mapping>,
    offset: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}
impl Weight {
    pub(crate) fn cache_identity(&self) -> (usize, usize, usize) {
        (self.map.id, self.offset, self.len())
    }
    pub(crate) fn raw_bytes(&self) -> &[u8] {
        &self.map.bytes()[self.offset..self.offset + self.len() * self.dtype.width()]
    }
    pub fn copy_range(&self, start: usize, end: usize, out: &mut [f32]) {
        assert!(start <= end && end <= self.len() && out.len() == end - start);
        let bytes = &self.map.bytes()
            [self.offset + start * self.dtype.width()..self.offset + end * self.dtype.width()];
        match self.dtype {
            DType::BF16 => {
                for (v, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *v = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
                }
            }
            DType::F32 => {
                for (v, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *v = f32::from_le_bytes(b.try_into().unwrap());
                }
            }
            DType::F16 => {
                for (v, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *v = half_to_float(u16::from_le_bytes([b[0], b[1]]));
                }
            }
        }
    }
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, index: usize) -> f32 {
        assert!(index < self.len());
        let start = self.offset + index * self.dtype.width();
        let b = &self.map.bytes()[start..];
        match self.dtype {
            DType::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()),
            DType::BF16 => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
            DType::F16 => half_to_float(u16::from_le_bytes([b[0], b[1]])),
        }
    }
    pub fn values(&self) -> Vec<f32> {
        let mut values = vec![0.; self.len()];
        self.copy_range(0, self.len(), &mut values);
        values
    }
    pub fn expect(&self, shape: &[usize], name: &str) -> Result<()> {
        if self.shape != shape {
            return Err(invalid(format!(
                "Tensor {name}: expected {shape:?}, found {:?}",
                self.shape
            )));
        }
        Ok(())
    }
}
fn half_to_float(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 31) as i32;
    let m = (h & 1023) as u32;
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            let shift = m.leading_zeros() - 21;
            sign | ((127 - 14 - shift) << 23) | ((m << shift & 1023) << 13)
        }
    } else if e == 31 {
        sign | 0x7f800000 | (m << 13)
    } else {
        sign | (((e + 112) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
pub struct Weights {
    tensors: BTreeMap<String, Weight>,
    pub directory: PathBuf,
}
impl Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let index = path.join("model.safetensors.index.json");
        let files: Vec<String> = if index.exists() {
            let v = json::read(index)?;
            v.get("weight_map")?
                .object()?
                .values()
                .map(|v| v.string().map(str::to_owned))
                .collect::<Result<std::collections::BTreeSet<_>>>()?
                .into_iter()
                .collect()
        } else {
            vec!["model.safetensors".into()]
        };
        let mut tensors = BTreeMap::new();
        for file in files {
            if Path::new(&file).components().count() != 1 {
                return Err(invalid("Invalid shard path"));
            }
            let map = Arc::new(Mapping::open(&path.join(file))?);
            let bytes = map.bytes();
            let header = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
            if header > 100_000_000 || header > bytes.len() - 8 {
                return Err(invalid("Invalid safetensors header size"));
            }
            let data = header + 8;
            let root = json::parse(std::str::from_utf8(&bytes[8..data])?)?;
            let mut ranges = Vec::new();
            for (name, v) in root.object()? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match v.get("dtype")?.string()? {
                    "F32" => DType::F32,
                    "BF16" => DType::BF16,
                    "F16" => DType::F16,
                    d => return Err(invalid(format!("Unsupported dtype {d} for {name}"))),
                };
                let shape = v
                    .get("shape")?
                    .array()?
                    .iter()
                    .map(json::Value::usize)
                    .collect::<Result<Vec<_>>>()?;
                let offsets = v.get("data_offsets")?.array()?;
                if offsets.len() != 2 {
                    return Err(invalid("Invalid tensor offsets"));
                }
                let a = offsets[0].usize()?;
                let b = offsets[1].usize()?;
                let count = shape
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    .ok_or_else(|| invalid("Tensor size overflow"))?;
                let size = count
                    .checked_mul(dtype.width())
                    .ok_or_else(|| invalid("Tensor bytes overflow"))?;
                if b < a || b > bytes.len() - data || b - a != size {
                    return Err(invalid(format!("Invalid tensor extent {name}")));
                }
                ranges.push((a, b));
                let w = Weight {
                    map: map.clone(),
                    offset: data + a,
                    shape,
                    dtype,
                };
                if tensors.insert(name.clone(), w).is_some() {
                    return Err(invalid(format!("Duplicate tensor {name}")));
                }
            }
            ranges.sort_unstable();
            if ranges.windows(2).any(|r| r[0].1 > r[1].0) {
                return Err(invalid("Overlapping safetensors tensors"));
            }
        }
        Ok(Self {
            tensors,
            directory: path.to_owned(),
        })
    }
    pub fn tensor(&self, name: &str) -> Result<Weight> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| invalid(format!("Missing checkpoint tensor {name}")))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn checked(&self, name: &str, shape: &[usize]) -> Result<Weight> {
        let w = self.tensor(name)?;
        w.expect(shape, name)?;
        Ok(w)
    }
}
