//! Native artifacts: NumPy arrays, float WAV, and CoreAudio 24-bit FLAC.
use crate::{Result, invalid, tensor::Tensor};
use std::{ffi::c_void, os::unix::ffi::OsStrExt};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

#[repr(C)]
struct StreamDescription {
    rate: f64,
    format: u32,
    flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels: u32,
    bits: u32,
    reserved: u32,
}
#[repr(C)]
struct AudioBuffer {
    channels: u32,
    bytes: u32,
    data: *mut c_void,
}
#[repr(C)]
struct AudioBufferList {
    count: u32,
    buffer: AudioBuffer,
}
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFURLCreateFromFileSystemRepresentation(
        allocator: *const c_void,
        bytes: *const u8,
        len: isize,
        is_directory: u8,
    ) -> *const c_void;
    fn CFRelease(value: *const c_void);
}
#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioFormatGetProperty(
        property: u32,
        specifier_size: u32,
        specifier: *const c_void,
        data_size: *mut u32,
        data: *mut c_void,
    ) -> i32;
    fn ExtAudioFileCreateWithURL(
        url: *const c_void,
        file_type: u32,
        format: *const StreamDescription,
        layout: *const c_void,
        flags: u32,
        file: *mut *mut c_void,
    ) -> i32;
    fn ExtAudioFileSetProperty(
        file: *mut c_void,
        property: u32,
        size: u32,
        data: *const c_void,
    ) -> i32;
    fn ExtAudioFileWrite(file: *mut c_void, frames: u32, buffers: *const AudioBufferList) -> i32;
    fn ExtAudioFileDispose(file: *mut c_void) -> i32;
}
fn fourcc(v: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*v)
}
fn audio_status(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "CoreAudio {operation} failed ({status}); the macOS audio codec service must be available"
        )))
    }
}
pub fn save_flac(path: impl AsRef<Path>, audio: &Tensor, sample_rate: usize) -> Result<()> {
    if audio.rows == 0
        || audio.cols == 0
        || audio.cols > 8
        || sample_rate == 0
        || sample_rate > 0xfffff
        || !audio.finite()
        || audio.rows.checked_mul(audio.cols) != Some(audio.data.len())
    {
        return Err(invalid("Invalid FLAC audio shape"));
    }
    let path = std::path::absolute(path)?;
    let name = path.as_os_str().as_bytes();
    let url = unsafe {
        CFURLCreateFromFileSystemRepresentation(
            std::ptr::null(),
            name.as_ptr(),
            name.len() as isize,
            0,
        )
    };
    if url.is_null() {
        return Err(invalid("Unable to create CoreAudio file URL"));
    }
    let mut output = StreamDescription {
        rate: sample_rate as f64,
        format: fourcc(b"flac"),
        flags: 3,
        bytes_per_packet: 0,
        frames_per_packet: 0,
        bytes_per_frame: 0,
        channels: audio.cols as u32,
        bits: 24,
        reserved: 0,
    };
    let mut size = std::mem::size_of::<StreamDescription>() as u32;
    let status = unsafe {
        AudioFormatGetProperty(
            fourcc(b"fmti"),
            0,
            std::ptr::null(),
            &mut size,
            (&mut output as *mut StreamDescription).cast(),
        )
    };
    if status != 0 {
        unsafe {
            CFRelease(url);
        }
        return audio_status(status, "FLAC format");
    }
    let mut file = std::ptr::null_mut();
    let status = unsafe {
        ExtAudioFileCreateWithURL(
            url,
            fourcc(b"flac"),
            &output,
            std::ptr::null(),
            1,
            &mut file,
        )
    };
    unsafe {
        CFRelease(url);
    }
    audio_status(status, "file creation")?;
    let client = StreamDescription {
        rate: sample_rate as f64,
        format: fourcc(b"lpcm"),
        flags: 9,
        bytes_per_packet: audio.cols as u32 * 4,
        frames_per_packet: 1,
        bytes_per_frame: audio.cols as u32 * 4,
        channels: audio.cols as u32,
        bits: 32,
        reserved: 0,
    };
    let result = (|| -> Result<()> {
        audio_status(
            unsafe {
                ExtAudioFileSetProperty(
                    file,
                    fourcc(b"cfmt"),
                    std::mem::size_of::<StreamDescription>() as u32,
                    (&client as *const StreamDescription).cast(),
                )
            },
            "client format",
        )?;
        for start in (0..audio.rows).step_by(65536) {
            let count = (audio.rows - start).min(65536);
            let buffers = AudioBufferList {
                count: 1,
                buffer: AudioBuffer {
                    channels: audio.cols as u32,
                    bytes: (count * audio.cols * 4) as u32,
                    data: audio.data[start * audio.cols..].as_ptr() as *mut c_void,
                },
            };
            audio_status(
                unsafe { ExtAudioFileWrite(file, count as u32, &buffers) },
                "FLAC encode",
            )?;
        }
        Ok(())
    })();
    let closed = unsafe { ExtAudioFileDispose(file) };
    result?;
    audio_status(closed, "file close")
}

unsafe extern "C" {
    fn CC_SHA256_Init(context: *mut u32) -> i32;
    fn CC_SHA256_Update(context: *mut u32, data: *const u8, len: u32) -> i32;
    fn CC_SHA256_Final(digest: *mut u8, context: *mut u32) -> i32;
}
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = BufReader::new(File::open(path)?);
    let mut context = [0u32; 28];
    unsafe {
        CC_SHA256_Init(context.as_mut_ptr());
    }
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        unsafe {
            CC_SHA256_Update(context.as_mut_ptr(), buffer.as_ptr(), n as u32);
        }
    }
    let mut digest = [0u8; 32];
    unsafe {
        CC_SHA256_Final(digest.as_mut_ptr(), context.as_mut_ptr());
    }
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}
fn npy_header(writer: &mut impl Write, descr: &str, shape: &[usize]) -> Result<()> {
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let shape = if shape.len() == 1 {
        format!("({dimensions},)")
    } else {
        format!("({dimensions})")
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    writer.write_all(b"\x93NUMPY\x01\x00")?;
    writer.write_all(&(header.len() as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    Ok(())
}
pub fn save_ids(path: impl AsRef<Path>, ids: &[usize]) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    npy_header(&mut f, "<i4", &[ids.len()])?;
    for &id in ids {
        f.write_all(&i32::try_from(id)?.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}
pub fn save_tensor(path: impl AsRef<Path>, x: &Tensor) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    npy_header(&mut f, "<f4", &[x.rows, x.cols])?;
    for v in &x.data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}
pub fn load_ids(path: impl AsRef<Path>) -> Result<Vec<usize>> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY\x01\x00" {
        return Err(invalid("Expected NumPy v1.0 token array"));
    }
    let mut length = [0; 2];
    f.read_exact(&mut length)?;
    let mut header = vec![0; u16::from_le_bytes(length) as usize];
    f.read_exact(&mut header)?;
    let header = String::from_utf8(header)?;
    if !header.contains("'<i4'") || !header.contains("'fortran_order': False") {
        return Err(invalid("Expected little-endian INT32 token array"));
    }
    let a = header
        .find('(')
        .ok_or_else(|| invalid("Missing NumPy shape"))?;
    let b = header[a..]
        .find(')')
        .ok_or_else(|| invalid("Invalid NumPy shape"))?
        + a;
    let shape: Vec<usize> = header[a + 1..b]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    if shape.len() != 1 {
        return Err(invalid("Expected a 1D token array"));
    }
    let mut ids = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        let mut bytes = [0; 4];
        f.read_exact(&mut bytes)?;
        ids.push(usize::try_from(i32::from_le_bytes(bytes))?);
    }
    if f.read(&mut [0; 1])? != 0 {
        return Err(invalid("Trailing token array data"));
    }
    Ok(ids)
}
pub fn load_tensor(path: impl AsRef<Path>) -> Result<Tensor> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY\x01\x00" {
        return Err(invalid("Expected NumPy v1.0 array"));
    }
    let mut len = [0; 2];
    f.read_exact(&mut len)?;
    let mut header = vec![0; u16::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let header = String::from_utf8(header)?;
    if !header.contains("'<f4'") || !header.contains("'fortran_order': False") {
        return Err(invalid("Expected little-endian FP32 row-major NumPy array"));
    }
    let a = header
        .find('(')
        .ok_or_else(|| invalid("Missing NumPy shape"))?;
    let b = header[a..]
        .find(')')
        .ok_or_else(|| invalid("Invalid NumPy shape"))?
        + a;
    let shape: Vec<usize> = header[a + 1..b]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    if shape.len() != 2 {
        return Err(invalid("Expected a 2D latent array"));
    }
    let n = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| invalid("NumPy size overflow"))?;
    let bytes = n
        .checked_mul(4)
        .ok_or_else(|| invalid("NumPy bytes overflow"))?;
    let mut data = vec![0; bytes];
    f.read_exact(&mut data)?;
    if f.read(&mut [0; 1])? != 0 {
        return Err(invalid("Trailing NumPy data"));
    }
    Tensor::from_vec(
        shape[0],
        shape[1],
        data.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}
pub fn save_wav(path: impl AsRef<Path>, audio: &Tensor, sample_rate: usize) -> Result<()> {
    let bytes = audio
        .data
        .len()
        .checked_mul(4)
        .ok_or_else(|| invalid("WAV size overflow"))?;
    let size = u32::try_from(bytes)?;
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(size + 48).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?;
    f.write_all(&(audio.cols as u16).to_le_bytes())?;
    f.write_all(&(sample_rate as u32).to_le_bytes())?;
    f.write_all(&((sample_rate * audio.cols * 4) as u32).to_le_bytes())?;
    f.write_all(&((audio.cols * 4) as u16).to_le_bytes())?;
    f.write_all(&32u16.to_le_bytes())?;
    f.write_all(b"fact")?;
    f.write_all(&4u32.to_le_bytes())?;
    f.write_all(&(audio.rows as u32).to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&size.to_le_bytes())?;
    for &v in &audio.data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}
