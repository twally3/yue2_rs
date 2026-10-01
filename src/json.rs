//! A small, strict JSON reader for checkpoint headers and generation requests.
use crate::{Result, invalid};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}
impl Value {
    pub fn get(&self, key: &str) -> Result<&Value> {
        self.optional(key)
            .ok_or_else(|| invalid(format!("Missing JSON field {key}")))
    }
    pub fn optional(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(o) => o.get(key),
            _ => None,
        }
    }
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err(invalid("Expected JSON string")),
        }
    }
    pub fn array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(a) => Ok(a),
            _ => Err(invalid("Expected JSON array")),
        }
    }
    pub fn object(&self) -> Result<&BTreeMap<String, Value>> {
        match self {
            Self::Object(o) => Ok(o),
            _ => Err(invalid("Expected JSON object")),
        }
    }
    pub fn usize(&self) -> Result<usize> {
        match self {
            Self::Number(s) => Ok(s.parse()?),
            _ => Err(invalid("Expected unsigned JSON integer")),
        }
    }
    pub fn u64(&self) -> Result<u64> {
        match self {
            Self::Number(s) => Ok(s.parse()?),
            _ => Err(invalid("Expected unsigned JSON integer")),
        }
    }
    pub fn f64(&self) -> Result<f64> {
        match self {
            Self::Number(s) => Ok(s.parse()?),
            _ => Err(invalid("Expected JSON number")),
        }
    }
    pub fn boolean(&self) -> Result<bool> {
        match self {
            Self::Bool(b) => Ok(*b),
            _ => Err(invalid("Expected JSON boolean")),
        }
    }
    pub fn size_or(&self, k: &str, d: usize) -> Result<usize> {
        self.optional(k).map_or(Ok(d), Self::usize)
    }
    pub fn float_or(&self, k: &str, d: f64) -> Result<f64> {
        self.optional(k).map_or(Ok(d), Self::f64)
    }
    pub fn bool_or(&self, k: &str, d: bool) -> Result<bool> {
        self.optional(k).map_or(Ok(d), Self::boolean)
    }
    pub fn text_or<'a>(&'a self, k: &str, d: &'a str) -> Result<&'a str> {
        self.optional(k).map_or(Ok(d), Self::string)
    }
}
pub fn object(items: impl IntoIterator<Item = (impl Into<String>, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Self::Number(v.to_string())
    }
}
impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Self::Number(v.to_string())
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        assert!(v.is_finite());
        Self::Number(v.to_string())
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
pub fn ids(v: &[usize]) -> Value {
    Value::Array(v.iter().copied().map(Value::from).collect())
}
pub fn read(path: impl AsRef<Path>) -> Result<Value> {
    let path = path.as_ref();
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let text = std::fs::read_to_string(path).map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!("Could not read JSON file {}: {error}", resolved.display()),
        )
    })?;
    parse(&text)
        .map_err(|error| invalid(format!("Invalid JSON in {}: {error}", resolved.display())))
}
pub fn write(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    std::fs::write(path, stringify(value) + "\n")?;
    Ok(())
}
pub fn stringify(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.clone(),
        Value::String(s) => {
            let mut r = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => r.push_str("\\\""),
                    '\\' => r.push_str("\\\\"),
                    '\n' => r.push_str("\\n"),
                    '\r' => r.push_str("\\r"),
                    '\t' => r.push_str("\\t"),
                    c if c < ' ' => r.push_str(&format!("\\u{:04x}", c as u32)),
                    c => r.push(c),
                }
            }
            r.push('"');
            r
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(stringify).collect::<Vec<_>>().join(",")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}:{}", stringify(&Value::from(k.clone())), stringify(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}
pub fn parse(text: &str) -> Result<Value> {
    let mut p = Parser {
        data: text.as_bytes(),
        pos: 0,
    };
    let v = p.value(0)?;
    p.space();
    if p.pos != p.data.len() {
        return Err(invalid("Trailing JSON content"));
    }
    Ok(v)
}
struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Parser<'_> {
    fn space(&mut self) {
        while self.pos < self.data.len()
            && matches!(self.data[self.pos], b' ' | b'\t' | b'\r' | b'\n')
        {
            self.pos += 1;
        }
    }
    fn eat(&mut self, b: u8) -> bool {
        self.space();
        if self.data.get(self.pos) == Some(&b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, b: u8) -> Result<()> {
        if self.eat(b) {
            Ok(())
        } else {
            Err(invalid(format!(
                "Expected JSON '{}' at {}",
                b as char, self.pos
            )))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 128 {
            return Err(invalid("JSON nesting limit"));
        }
        self.space();
        match self.data.get(self.pos).copied() {
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'{') => {
                self.pos += 1;
                let mut o = BTreeMap::new();
                if !self.eat(b'}') {
                    loop {
                        self.space();
                        let k = self.string()?;
                        self.expect(b':')?;
                        let v = self.value(depth + 1)?;
                        if o.insert(k, v).is_some() {
                            return Err(invalid("Duplicate JSON key"));
                        }
                        if self.eat(b'}') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Ok(Value::Object(o))
            }
            Some(b'[') => {
                self.pos += 1;
                let mut a = Vec::new();
                if !self.eat(b']') {
                    loop {
                        a.push(self.value(depth + 1)?);
                        if self.eat(b']') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Ok(Value::Array(a))
            }
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'-' | b'0'..=b'9') => {
                let start = self.pos;
                if self.data[self.pos] == b'-' {
                    self.pos += 1;
                }
                match self.data.get(self.pos) {
                    Some(b'0') => self.pos += 1,
                    Some(b'1'..=b'9') => {
                        self.pos += 1;
                        while self.data.get(self.pos).is_some_and(u8::is_ascii_digit) {
                            self.pos += 1;
                        }
                    }
                    _ => return Err(invalid("Invalid JSON number")),
                };
                if self.data.get(self.pos) == Some(&b'.') {
                    self.pos += 1;
                    self.digits()?;
                }
                if self
                    .data
                    .get(self.pos)
                    .is_some_and(|b| matches!(b, b'e' | b'E'))
                {
                    self.pos += 1;
                    if self
                        .data
                        .get(self.pos)
                        .is_some_and(|b| matches!(b, b'+' | b'-'))
                    {
                        self.pos += 1;
                    }
                    self.digits()?;
                }
                let n = std::str::from_utf8(&self.data[start..self.pos])?;
                if !n.parse::<f64>()?.is_finite() {
                    return Err(invalid("Nonfinite JSON number"));
                }
                Ok(Value::Number(n.into()))
            }
            _ => Err(invalid(format!("Invalid JSON value at {}", self.pos))),
        }
    }
    fn digits(&mut self) -> Result<()> {
        let start = self.pos;
        while self.data.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        if start == self.pos {
            Err(invalid("Expected JSON digits"))
        } else {
            Ok(())
        }
    }
    fn literal(&mut self, s: &[u8], v: Value) -> Result<Value> {
        if self.data.get(self.pos..self.pos + s.len()) == Some(s) {
            self.pos += s.len();
            Ok(v)
        } else {
            Err(invalid("Invalid JSON literal"))
        }
    }
    fn hex(&mut self) -> Result<u32> {
        let end = self.pos + 4;
        let s = std::str::from_utf8(
            self.data
                .get(self.pos..end)
                .ok_or_else(|| invalid("Short Unicode escape"))?,
        )?;
        let v = u32::from_str_radix(s, 16)?;
        self.pos = end;
        Ok(v)
    }
    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let b = *self
                .data
                .get(self.pos)
                .ok_or_else(|| invalid("Unterminated JSON string"))?;
            self.pos += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let e = *self
                        .data
                        .get(self.pos)
                        .ok_or_else(|| invalid("Short escape"))?;
                    self.pos += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(10),
                        b'r' => out.push(13),
                        b't' => out.push(9),
                        b'u' => {
                            let mut cp = self.hex()?;
                            if (0xd800..=0xdbff).contains(&cp) {
                                if self.data.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(invalid("Unpaired surrogate"));
                                }
                                self.pos += 2;
                                let low = self.hex()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err(invalid("Invalid low surrogate"));
                                }
                                cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                            }
                            let c = char::from_u32(cp)
                                .ok_or_else(|| invalid("Invalid Unicode escape"))?;
                            out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                        }
                        _ => return Err(invalid("Invalid JSON escape")),
                    }
                }
                0..=31 => return Err(invalid("Control byte in JSON string")),
                _ => out.push(b),
            }
        }
        Ok(String::from_utf8(out)?)
    }
}
