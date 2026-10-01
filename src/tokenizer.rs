//! Frozen Qwen byte-pair tokenizer, using macOS ICU for Unicode and pretokenization.
use crate::{Result, invalid, protocol::EOD};
use std::{collections::HashMap, ffi::c_void, path::Path};

#[link(name = "icucore")]
unsafe extern "C" {
    fn unorm2_getNFCInstance(status: *mut i32) -> *const c_void;
    fn unorm2_normalize(
        norm: *const c_void,
        source: *const u16,
        length: i32,
        dest: *mut u16,
        capacity: i32,
        status: *mut i32,
    ) -> i32;
    fn uregex_open(
        pattern: *const u16,
        length: i32,
        flags: u32,
        parse_error: *mut c_void,
        status: *mut i32,
    ) -> *mut c_void;
    fn uregex_setText(regex: *mut c_void, text: *const u16, len: i32, status: *mut i32);
    fn uregex_findNext(regex: *mut c_void, status: *mut i32) -> i8;
    fn uregex_start(regex: *mut c_void, group: i32, status: *mut i32) -> i32;
    fn uregex_end(regex: *mut c_void, group: i32, status: *mut i32) -> i32;
    fn uregex_close(regex: *mut c_void);
}
struct Regex(*mut c_void);
impl Drop for Regex {
    fn drop(&mut self) {
        unsafe {
            uregex_close(self.0);
        }
    }
}
pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, usize>,
    bytes: Vec<Vec<u8>>,
}
impl Tokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut ranks = HashMap::new();
        let mut bytes = vec![Vec::new(); EOD];
        for line in text.lines().filter(|l| !l.is_empty()) {
            let mut parts = line.split_whitespace();
            let b = decode_base64(parts.next().ok_or_else(|| invalid("Invalid merge line"))?)?;
            let rank: usize = parts
                .next()
                .ok_or_else(|| invalid("Missing BPE rank"))?
                .parse()?;
            if parts.next().is_some()
                || rank >= EOD
                || !bytes[rank].is_empty()
                || ranks.contains_key(&b)
            {
                return Err(invalid("Invalid or duplicate BPE rank"));
            }
            bytes[rank] = b.clone();
            ranks.insert(b, rank);
        }
        if ranks.len() != EOD || bytes.iter().any(Vec::is_empty) {
            return Err(invalid("Expected 151643 ordinary tokenizer entries"));
        }
        for b in 0..=255u8 {
            if !ranks.contains_key(&vec![b]) {
                return Err(invalid("Missing byte fallback token"));
            }
        }
        Ok(Self { ranks, bytes })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let input: Vec<u16> = text.encode_utf16().collect();
        if input.len() > i32::MAX as usize {
            return Err(invalid("Text exceeds ICU length limit"));
        }
        let mut status = 0;
        let norm = unsafe { unorm2_getNFCInstance(&mut status) };
        check(status)?;
        let mut normalized = vec![0; input.len() * 2 + 16];
        let needed = unsafe {
            unorm2_normalize(
                norm,
                input.as_ptr(),
                input.len() as i32,
                normalized.as_mut_ptr(),
                normalized.len() as i32,
                &mut status,
            )
        };
        if status == 15 {
            normalized.resize(needed as usize, 0);
            status = 0;
            unsafe {
                unorm2_normalize(
                    norm,
                    input.as_ptr(),
                    input.len() as i32,
                    normalized.as_mut_ptr(),
                    normalized.len() as i32,
                    &mut status,
                );
            }
        }
        check(status)?;
        normalized.truncate(needed as usize);
        let pattern:Vec<u16>=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".encode_utf16().collect();
        status = 0;
        let regex = Regex(unsafe {
            uregex_open(
                pattern.as_ptr(),
                pattern.len() as i32,
                0,
                std::ptr::null_mut(),
                &mut status,
            )
        });
        check(status)?;
        unsafe {
            uregex_setText(
                regex.0,
                normalized.as_ptr(),
                normalized.len() as i32,
                &mut status,
            );
        }
        check(status)?;
        let mut result = Vec::new();
        let mut previous = 0;
        loop {
            let found = unsafe { uregex_findNext(regex.0, &mut status) };
            check(status)?;
            if found == 0 {
                break;
            }
            let a = unsafe { uregex_start(regex.0, 0, &mut status) } as usize;
            let b = unsafe { uregex_end(regex.0, 0, &mut status) } as usize;
            check(status)?;
            if a != previous || b <= a {
                return Err(invalid("Tokenizer pattern did not cover input"));
            }
            let s = String::from_utf16(&normalized[a..b])?;
            self.bpe(s.as_bytes(), &mut result);
            previous = b;
        }
        if previous != normalized.len() {
            return Err(invalid("Tokenizer left unmatched text"));
        }
        Ok(result)
    }
    fn bpe(&self, bytes: &[u8], out: &mut Vec<usize>) {
        if let Some(&id) = self.ranks.get(bytes) {
            out.push(id);
            return;
        }
        let mut boundaries: Vec<usize> = (0..=bytes.len()).collect();
        while boundaries.len() > 2 {
            let mut best = usize::MAX;
            let mut selected = 0;
            for i in 0..boundaries.len() - 2 {
                if let Some(&rank) = self.ranks.get(&bytes[boundaries[i]..boundaries[i + 2]]) {
                    if rank < best {
                        best = rank;
                        selected = i;
                    }
                }
            }
            if best == usize::MAX {
                break;
            }
            boundaries.remove(selected + 1);
        }
        for pair in boundaries.windows(2) {
            out.push(self.ranks[&bytes[pair[0]..pair[1]]]);
        }
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < EOD {
                bytes.extend_from_slice(&self.bytes[id]);
            } else if let Some(s) = special(id) {
                bytes.extend_from_slice(s.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
fn check(status: i32) -> Result<()> {
    if status > 0 {
        Err(invalid(format!("macOS ICU error {status}")))
    } else {
        Ok(())
    }
}
fn special(id: usize) -> Option<String> {
    let i = id.checked_sub(EOD)?;
    Some(match i {
        0 => "<|endoftext|>".into(),
        1 => "<|im_start|>".into(),
        2 => "<|im_end|>".into(),
        3 => "<R>".into(),
        4 => "<S>".into(),
        5 => "<X>".into(),
        6 => "<mask>".into(),
        7 => "<sep>".into(),
        204 => "<abc>".into(),
        205 => "</abc>".into(),
        8..=207 => format!("<extra_{}>", i - 8),
        _ => return None,
    })
}
fn decode_base64(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(4) {
        return Err(invalid("Invalid base64 length"));
    }
    let mut out = Vec::new();
    for chunk in s.as_bytes().chunks(4) {
        let mut n = 0u32;
        let mut padding = 0;
        for &b in chunk {
            let v = match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => {
                    padding += 1;
                    0
                }
                _ => return Err(invalid("Invalid base64 token")),
            };
            n = (n << 6) | v as u32;
        }
        if padding > 2 {
            return Err(invalid("Invalid base64 padding"));
        }
        out.push((n >> 16) as u8);
        if padding < 2 {
            out.push((n >> 8) as u8);
        }
        if padding < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}
