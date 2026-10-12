//! Native two-voice ABC checks and instrumental conversion.
//! Ported from YuE2's Apache-2.0 ABC tools and MIT instrumental helpers.
use std::collections::BTreeMap;
use yue2_rs::{
    Result,
    json::{self, Value},
};

const VOICES: [&str; 2] = ["Vocal", "Ins"];
const HEADERS: [&str; 2] = [
    "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
    "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"",
];
const LENGTHS: [usize; 11] = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1];
// The native power-of-two grid has at most 1,024 whole-note divisions.
// Integer ticks avoid approximating quarter-note fractions during comparisons.
const QUARTER: usize = 256;

fn check(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}
fn positive(text: &str) -> Result<usize> {
    let n: usize = text.parse()?;
    check(
        n > 0 && n <= 1_000_000,
        "Expected a bounded positive integer",
    )?;
    Ok(n)
}
fn meter(text: &str) -> Result<(usize, usize)> {
    let (n, d) = text
        .split_once('/')
        .ok_or("Meter must be an explicit fraction")?;
    let (n, d) = (positive(n)?, positive(d)?);
    check(
        d <= 1024 && d.is_power_of_two(),
        "Unsupported meter denominator",
    )?;
    Ok((n, d))
}
fn signature(key: &str) -> Result<[i32; 7]> {
    let major = [
        "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
    ];
    let minor = [
        "Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m",
        "A#m",
    ];
    let count = major
        .iter()
        .position(|&k| k == key)
        .or_else(|| minor.iter().position(|&k| k == key))
        .ok_or_else(|| format!("Unsupported major/minor key: {key}"))? as i32
        - 7;
    let mut result = [0; 7];
    let order = if count > 0 {
        [3, 0, 4, 1, 5, 2, 6]
    } else {
        [6, 2, 5, 1, 4, 0, 3]
    };
    for &letter in &order[..count.unsigned_abs() as usize] {
        result[letter] = count.signum();
    }
    Ok(result)
}
fn pitch_name(text: &str) -> Option<usize> {
    let b = text.as_bytes();
    if !b.first().is_some_and(|b| (b'A'..=b'G').contains(b)) {
        return None;
    }
    let mut n = 1;
    if b.get(n).is_some_and(|b| *b == b'b' || *b == b'#') {
        let accidental = b[n];
        n += 1;
        if b.get(n) == Some(&accidental) {
            n += 1;
        }
    }
    Some(n)
}
fn chord(text: &str) -> bool {
    let (root, bass) = text
        .split_once('/')
        .map_or((text, None), |(a, b)| (a, Some(b)));
    let Some(n) = pitch_name(root) else {
        return false;
    };
    [
        "", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2", "6", "m6",
        "7sus4", "m(maj7)",
    ]
    .contains(&&root[n..])
        && bass.is_none_or(|b| pitch_name(b) == Some(b.len()))
}
fn fraction(ticks: usize) -> String {
    let mut a = ticks;
    let mut b = QUARTER;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let (n, d) = (ticks / a, QUARTER / a);
    if d == 1 {
        n.to_string()
    } else {
        format!("{n}/{d}")
    }
}
fn fraction_value(ticks: usize) -> Value {
    fraction(ticks).into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    pub start: usize,
    pub pitch: usize,
    pub duration: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Bar {
    start: usize,
    duration: usize,
    meter: (usize, usize),
}
#[derive(Clone, Debug)]
struct Voice {
    meter: (usize, usize),
    key: String,
    time: usize,
    notes: Vec<Note>,
    bars: Vec<Bar>,
    chords: Vec<(usize, String)>,
    keys: Vec<(usize, String)>,
    pending: Option<(usize, i32)>,
}
#[derive(Clone)]
pub struct Score {
    text: String,
    denominator: usize,
    bpm: usize,
    voices: [Voice; 2],
    music_lines: BTreeMap<usize, usize>,
    sections: BTreeMap<usize, String>,
}
enum Token {
    Chord(String),
    Key(String),
    Note {
        letter: u8,
        accidental: Option<i32>,
        octave: i32,
        units: usize,
        tie: bool,
    },
}
fn token(text: &str, cursor: &mut usize) -> Result<Token> {
    let b = text.as_bytes();
    let start = *cursor;
    if b.get(start) == Some(&b'"') {
        let end = text[start + 1..].find('"').ok_or("Unclosed chord symbol")? + start + 1;
        let value = &text[start + 1..end];
        check(chord(value), format!("Unsupported chord: {value}"))?;
        *cursor = end + 1;
        return Ok(Token::Chord(value.into()));
    }
    if text[start..].starts_with("[K:") {
        let end = text[start + 3..].find(']').ok_or("Unclosed inline key")? + start + 3;
        let value = &text[start + 3..end];
        signature(value)?;
        *cursor = end + 1;
        return Ok(Token::Key(value.into()));
    }
    let mut i = start;
    let accidental = match b.get(i) {
        Some(b'^') | Some(b'_') => {
            let mark = b[i];
            i += 1;
            let mut count = 1;
            if b.get(i) == Some(&mark) {
                i += 1;
                count += 1;
            }
            Some(if mark == b'^' { count } else { -count })
        }
        Some(b'=') => {
            i += 1;
            Some(0)
        }
        _ => None,
    };
    let letter = *b.get(i).ok_or("Missing ABC note")?;
    i += 1;
    check(
        matches!(letter, b'A'..=b'G' | b'a'..=b'g' | b'z'),
        format!("Unsupported ABC token at byte {start}"),
    )?;
    let mut octave = 0;
    let mut up = false;
    let mut down = false;
    while matches!(b.get(i), Some(b'\'') | Some(b',')) {
        if b[i] == b'\'' {
            up = true;
            octave += 1;
        } else {
            down = true;
            octave -= 1;
        }
        i += 1;
        check((-16..=16).contains(&octave), "Octave outside MIDI range")?;
    }
    check(!(up && down), "Mixed octave marks")?;
    let number = i;
    while b.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    let units = if i == number {
        1
    } else {
        positive(&text[number..i])?
    };
    check(
        LENGTHS.contains(&units),
        "Unsupported duration; split into tied supported lengths",
    )?;
    let tie = b.get(i) == Some(&b'-');
    i += usize::from(tie);
    check(
        letter != b'z' || (accidental.is_none() && octave == 0 && !tie),
        "Rest with accidental, octave or tie",
    )?;
    *cursor = i;
    Ok(Token::Note {
        letter,
        accidental,
        octave,
        units,
        tie,
    })
}
fn parse_bar(body: &str, v: &mut Voice, unit: usize) -> Result<()> {
    let length = 1024 * v.meter.0 / v.meter.1;
    let start = v.time;
    let mut offset = 0;
    let mut local = [None; 7];
    let mut cursor = 0;
    if body == "Z" {
        check(v.pending.is_none(), "Tie enters a full-measure rest")?;
        offset = length;
    } else {
        while cursor < body.len() {
            if body.as_bytes()[cursor].is_ascii_whitespace() {
                cursor += 1;
                continue;
            }
            check(offset < length, "Event after measure end")?;
            match token(body, &mut cursor)? {
                Token::Chord(c) => v.chords.push((start + offset, c)),
                Token::Key(k) => {
                    v.key = k.clone();
                    v.keys.push((start + offset, k));
                    local = [None; 7];
                }
                Token::Note {
                    letter,
                    accidental,
                    octave,
                    units,
                    tie,
                } => {
                    let duration = unit * units;
                    check(offset + duration <= length, "Note/rest exceeds meter")?;
                    if letter == b'z' {
                        check(v.pending.is_none(), "Tie enters a rest")?;
                    } else {
                        let index = b"CDEFGAB"
                            .iter()
                            .position(|&l| l == letter.to_ascii_uppercase())
                            .unwrap();
                        let written = 60
                            + [0, 2, 4, 5, 7, 9, 11][index]
                            + 12 * (i32::from(letter.is_ascii_lowercase()) + octave);
                        let alteration =
                            accidental.unwrap_or(local[index].unwrap_or(signature(&v.key)?[index]));
                        if let Some(a) = accidental {
                            local[index] = Some(a);
                        }
                        let mut pitch = written + alteration;
                        if let Some((old, old_written)) = v.pending {
                            if accidental.is_none() && written == old_written {
                                pitch = old as i32;
                            }
                            check(pitch == old as i32, "Tie changes sounding pitch")?;
                            v.notes.last_mut().unwrap().duration += duration;
                        } else {
                            check((0..=127).contains(&pitch), "Pitch outside MIDI range")?;
                            v.notes.push(Note {
                                start: start + offset,
                                pitch: pitch as usize,
                                duration,
                            });
                        }
                        v.pending = tie.then_some((pitch as usize, written));
                    }
                    offset += duration;
                }
            }
        }
    }
    check(
        offset == length,
        format!(
            "Measure duration {} != {} quarter notes",
            fraction(offset),
            fraction(length)
        ),
    )?;
    v.bars.push(Bar {
        start,
        duration: length,
        meter: v.meter,
    });
    v.time = v
        .time
        .checked_add(length)
        .ok_or("Score duration overflow")?;
    Ok(())
}
impl Score {
    pub fn parse(text: &str) -> Result<Self> {
        let lines: Vec<_> = text.lines().collect();
        check(lines.len() >= 12, "Incomplete native two-voice ABC")?;
        check(
            lines[0..2] == ["X:1", "T:"],
            "Expected X:1 and blank T: header",
        )?;
        let m = meter(lines[2].strip_prefix("M:").ok_or("Missing M: header")?)?;
        let denominator = positive(
            lines[3]
                .strip_prefix("L:1/")
                .ok_or("Expected L:1/<power of two>")?,
        )?;
        check(
            denominator <= 1024 && denominator.is_power_of_two(),
            "Unsupported L: denominator",
        )?;
        let bpm = positive(
            lines[4]
                .strip_prefix("Q:1/4=")
                .ok_or("Expected quarter-note BPM")?,
        )?;
        check(
            lines[5..7] == HEADERS,
            "Preserve native Vocal and Ins definitions",
        )?;
        let key = lines[7].strip_prefix("K:").ok_or("Missing K: header")?;
        signature(key)?;
        let voice = Voice {
            meter: m,
            key: key.into(),
            time: 0,
            notes: vec![],
            bars: vec![],
            chords: vec![],
            keys: vec![(0, key.into())],
            pending: None,
        };
        let mut voices = [voice.clone(), voice];
        let mut music_lines = BTreeMap::new();
        let mut sections = BTreeMap::new();
        let mut cursor = 8;
        while cursor < lines.len() {
            while cursor < lines.len() && lines[cursor].starts_with("% ") {
                sections.insert(voices[0].time, lines[cursor][2..].into());
                cursor += 1;
            }
            check(cursor < lines.len(), "Dangling section comment")?;
            let mut counts = [0; 2];
            for index in 0..2 {
                check(
                    lines.get(cursor) == Some(&format!("V: {}", VOICES[index]).as_str()),
                    format!("Expected V: {}", VOICES[index]),
                )?;
                cursor += 1;
                let v = &mut voices[index];
                let mut fields = [false; 2];
                while let Some(line) = lines.get(cursor) {
                    if let Some(value) = line.strip_prefix("M:") {
                        check(!fields[0], "Duplicate M: field")?;
                        fields[0] = true;
                        v.meter = meter(value)?;
                    } else if let Some(value) = line.strip_prefix("K:") {
                        check(!fields[1], "Duplicate K: field")?;
                        fields[1] = true;
                        signature(value)?;
                        v.key = value.into();
                        v.keys.push((v.time, value.into()));
                    } else {
                        break;
                    }
                    cursor += 1;
                }
                let body = lines
                    .get(cursor)
                    .ok_or("Missing voice music line")?
                    .strip_suffix('|')
                    .ok_or("Music line must end with a plain barline")?;
                music_lines.insert(cursor, index);
                cursor += 1;
                for raw in body.split('|') {
                    let bar = raw.trim();
                    check(
                        !bar.is_empty(),
                        "Empty/repeat/double barline is unsupported",
                    )?;
                    if bar == "Z" || matches!(bar, "Z2" | "Z3" | "Z4") {
                        let n = if bar == "Z" { 1 } else { bar[1..].parse()? };
                        for _ in 0..n {
                            parse_bar("Z", v, 1024 / denominator)?;
                            counts[index] += 1;
                        }
                    } else {
                        parse_bar(bar, v, 1024 / denominator)?;
                        counts[index] += 1;
                    }
                }
                check(
                    (1..=4).contains(&counts[index]),
                    "Expected 1–4 bars per voice group",
                )?;
            }
            check(
                counts[0] == counts[1],
                "Voices have different measure counts",
            )?;
        }
        check(
            voices.iter().all(|v| v.pending.is_none()),
            "Unresolved tie at end of score",
        )?;
        check(
            voices[1].chords.is_empty(),
            "Native chord symbols belong in Vocal",
        )?;
        check(
            voices[0].bars == voices[1].bars,
            "Voice meter/time grids differ",
        )?;
        check(
            voices[0].keys == voices[1].keys,
            "Voice key timelines differ",
        )?;
        Ok(Self {
            text: text.into(),
            denominator,
            bpm,
            voices,
            music_lines,
            sections,
        })
    }
    pub fn report(&self) -> Value {
        json::object([
            (
                "scope",
                "native-dialect symbolic check; not audio verification".into(),
            ),
            ("bpm", self.bpm.into()),
            ("unit_length", format!("1/{}", self.denominator).into()),
            ("duration_quarters", fraction_value(self.voices[0].time)),
            (
                "nominal_duration_seconds",
                (self.voices[0].time as f64 / QUARTER as f64 * 60. / self.bpm as f64).into(),
            ),
            (
                "voices",
                json::object(VOICES.iter().zip(&self.voices).map(|(name, v)| {
                    (
                        *name,
                        json::object([
                            ("sounding_notes", v.notes.len().into()),
                            ("measures", v.bars.len().into()),
                            (
                                "notes",
                                Value::Array(
                                    v.notes
                                        .iter()
                                        .map(|n| {
                                            json::object([
                                                ("onset_quarters", fraction_value(n.start)),
                                                ("midi_pitch", n.pitch.into()),
                                                ("duration_quarters", fraction_value(n.duration)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            (
                                "chords",
                                Value::Array(
                                    v.chords
                                        .iter()
                                        .map(|(t, c)| {
                                            Value::Array(vec![fraction_value(*t), c.clone().into()])
                                        })
                                        .collect(),
                                ),
                            ),
                            (
                                "keys",
                                Value::Array(
                                    v.keys
                                        .iter()
                                        .map(|(t, k)| {
                                            Value::Array(vec![fraction_value(*t), k.clone().into()])
                                        })
                                        .collect(),
                                ),
                            ),
                            (
                                "bars",
                                Value::Array(
                                    v.bars
                                        .iter()
                                        .map(|b| {
                                            Value::Array(vec![
                                                fraction_value(b.start),
                                                fraction_value(b.duration),
                                                Value::Array(vec![
                                                    b.meter.0.into(),
                                                    b.meter.1.into(),
                                                ]),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                        ]),
                    )
                })),
            ),
        ])
    }
    pub fn compare(&self, after: &Self, voice: Option<usize>, allow_tempo: bool) -> Value {
        let mut differences = Vec::new();
        let mut names = Vec::new();
        if self.bpm != after.bpm && !allow_tempo {
            differences.push(Value::from("Quarter-note tempo differs"));
        }
        for (i, name) in VOICES.iter().enumerate() {
            if voice.is_some_and(|v| v != i) {
                continue;
            }
            names.push(Value::from(*name));
            if self.voices[i].bars != after.voices[i].bars {
                differences.push(format!("{name}: meter/time grid differs").into());
            }
            if self.voices[i].notes != after.voices[i].notes {
                let a = &self.voices[i].notes;
                let b = &after.voices[i].notes;
                let first = a
                    .iter()
                    .zip(b)
                    .position(|(a, b)| a != b)
                    .unwrap_or(a.len().min(b.len()));
                differences.push(
                    format!(
                        "{name}: pitch, onset or duration differs at note {}",
                        first + 1
                    )
                    .into(),
                );
            }
        }
        json::object([
            ("match", differences.is_empty().into()),
            ("differences", Value::Array(differences)),
            ("compared_voices", Value::Array(names)),
            ("tempo_change_allowed", allow_tempo.into()),
            (
                "scope",
                "exact symbolic notes and meter; no audio or harmony guarantee".into(),
            ),
        ])
    }
    pub fn strip_chords(&self, keep: Option<usize>) -> Result<String> {
        let mut lines: Vec<String> = self.text.lines().map(str::to_owned).collect();
        for (&line, &voice) in &self.music_lines {
            let mut out = String::new();
            let mut cursor = 0;
            let body = &lines[line];
            while cursor < body.len() {
                let byte = body.as_bytes()[cursor];
                if byte == b'|'
                    || byte == b'Z'
                    || byte.is_ascii_whitespace()
                    || (byte.is_ascii_digit() && out.ends_with('Z'))
                {
                    out.push(byte as char);
                    cursor += 1;
                    continue;
                }
                let start = cursor;
                match token(body, &mut cursor)? {
                    Token::Chord(_) => {}
                    Token::Note { units, .. } if keep.is_some_and(|k| k != voice) => {
                        out.push('z');
                        if units != 1 {
                            out.push_str(&units.to_string());
                        }
                    }
                    _ => out.push_str(&body[start..cursor]),
                }
            }
            lines[line] = out;
        }
        let text = lines.join("\n") + "\n";
        let parsed = Self::parse(&text)?;
        check(
            self.compare(&parsed, keep, false).get("match")?.boolean()?,
            "Chord stripping changed retained melody",
        )?;
        check(
            parsed.voices.iter().all(|v| v.chords.is_empty()),
            "Chord symbols remain",
        )?;
        Ok(text)
    }
    pub fn instrumental(&self, reject_overlap: bool, keep_chords: bool) -> Result<(String, Value)> {
        let vocal = &self.voices[0];
        let ins = &self.voices[1];
        check(
            !vocal.notes.is_empty() || !ins.notes.is_empty(),
            "Score has no sounding notes",
        )?;
        let mut merged = vocal.notes.clone();
        let mut affected = Vec::new();
        for (index, n) in ins.notes.iter().enumerate() {
            let mut pieces = vec![(n.start, n.start + n.duration)];
            for v in &vocal.notes {
                let (left, right) = (v.start, v.start + v.duration);
                let mut next = Vec::new();
                for (a, b) in pieces {
                    if right <= a || left >= b {
                        next.push((a, b));
                    } else {
                        if a < left {
                            next.push((a, left));
                        }
                        if right < b {
                            next.push((right, b));
                        }
                    }
                }
                pieces = next;
            }
            if pieces != [(n.start, n.start + n.duration)] {
                check(
                    !reject_overlap,
                    "Vocal and Ins overlap; choose vocal priority or edit the arrangement",
                )?;
                affected.push(json::object([
                    ("ins_note", index.into()),
                    ("onset_quarters", fraction_value(n.start)),
                    ("midi_pitch", n.pitch.into()),
                    ("duration_quarters", fraction_value(n.duration)),
                    (
                        "retained_segments",
                        Value::Array(
                            pieces
                                .iter()
                                .map(|(a, b)| {
                                    Value::Array(vec![fraction_value(*a), fraction_value(b - a)])
                                })
                                .collect(),
                        ),
                    ),
                ]));
            }
            merged.extend(pieces.into_iter().map(|(a, b)| Note {
                start: a,
                pitch: n.pitch,
                duration: b - a,
            }));
        }
        merged.sort_by_key(|n| (n.start, n.pitch, n.duration));
        check(
            vocal
                .keys
                .iter()
                .all(|(t, _)| vocal.bars.iter().any(|b| b.start == *t)),
            "Inline key changes inside bars require explicit normalization before conversion",
        )?;
        let chords = if keep_chords {
            vocal.chords.clone()
        } else {
            vec![]
        };
        let output = compile(
            self.bpm,
            &vocal.bars,
            &vocal.keys,
            &self.sections,
            &merged,
            &chords,
        )?;
        let parsed = Self::parse(&output)?;
        check(
            parsed.voices[1].notes == merged && parsed.voices[0].notes.is_empty(),
            "Instrumental transfer failed exact note roundtrip",
        )?;
        check(
            parsed.voices[1].bars == ins.bars && parsed.bpm == self.bpm,
            "Transfer changed meter or tempo",
        )?;
        let report = json::object([
            ("vocal_notes_before", vocal.notes.len().into()),
            ("vocal_notes_after", 0usize.into()),
            ("original_ins_notes", ins.notes.len().into()),
            ("output_ins_notes", merged.len().into()),
            ("vocal_pitch_onset_duration_preserved", true.into()),
            ("meter_and_tempo_preserved", true.into()),
            (
                "chords",
                if keep_chords { "preserved" } else { "removed" }.into(),
            ),
            (
                "overlap_policy",
                if reject_overlap { "error" } else { "vocal" }.into(),
            ),
            ("affected_ins_notes", Value::Array(affected)),
        ]);
        Ok((output, report))
    }
}

fn parts(mut ticks: usize, unit: usize) -> Result<Vec<usize>> {
    check(
        ticks.is_multiple_of(unit),
        "Duration is outside the selected native grid",
    )?;
    ticks /= unit;
    let mut parts = Vec::new();
    for n in LENGTHS {
        while ticks >= n {
            parts.push(n);
            ticks -= n;
        }
    }
    Ok(parts)
}
fn spelling(pitch: usize, key: &str, active: &mut [Option<i32>; 7]) -> Result<String> {
    let sig = signature(key)?;
    let sharp = sig.iter().sum::<i32>() >= 0;
    let mut candidates = Vec::new();
    for (i, pc) in [0, 2, 4, 5, 7, 9, 11].into_iter().enumerate() {
        for a in -1i32..=1 {
            let base = pitch as i32 - a - 60 - pc;
            if base.rem_euclid(12) != 0 {
                continue;
            }
            let letter = b"CDEFGAB"[i];
            candidates.push((
                (
                    a != sig[i],
                    a.abs(),
                    if sharp { a < 0 } else { a > 0 },
                    letter,
                ),
                i,
                a,
                base.div_euclid(12),
            ));
        }
    }
    candidates.sort();
    let (_, i, a, octave) = candidates[0];
    let mut out = String::new();
    if a != active[i].unwrap_or(sig[i]) {
        out.push(match a {
            -1 => '_',
            0 => '=',
            _ => '^',
        });
        active[i] = Some(a);
    }
    let letter = b"CDEFGAB"[i];
    out.push(if octave > 0 {
        letter.to_ascii_lowercase()
    } else {
        letter
    } as char);
    if octave < 0 {
        out.push_str(&",".repeat((-octave) as usize));
    }
    if octave > 1 {
        out.push_str(&"'".repeat((octave - 1) as usize));
    }
    Ok(out)
}
fn timed_rest(out: &mut String, ticks: usize, unit: usize) -> Result<()> {
    for n in parts(ticks, unit)? {
        out.push('z');
        if n != 1 {
            out.push_str(&n.to_string());
        }
    }
    Ok(())
}
fn active_chords(chords: &[(usize, String)]) -> Vec<(usize, String)> {
    let mut result: Vec<(usize, String)> = Vec::new();
    for c in chords {
        if result.last().is_none_or(|last| last.1 != c.1) {
            result.push(c.clone());
        }
    }
    result
}
fn compile(
    bpm: usize,
    bars: &[Bar],
    keys: &[(usize, String)],
    sections: &BTreeMap<usize, String>,
    notes: &[Note],
    chords: &[(usize, String)],
) -> Result<String> {
    for pair in notes.windows(2) {
        check(
            pair[0].start + pair[0].duration <= pair[1].start,
            "Overlapping lead melody notes",
        )?;
    }
    for pair in chords.windows(2) {
        check(
            pair[0].0 != pair[1].0,
            "Conflicting/duplicate chord events at one onset",
        )?;
    }
    let unit = bars
        .iter()
        .map(|b| b.duration)
        .chain(notes.iter().flat_map(|n| [n.start, n.duration]))
        .chain(chords.iter().map(|c| c.0))
        .fold(32, |mut a, mut b| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        });
    let denominator = 1024 / unit;
    let first = &bars[0];
    let mut out = format!(
        "X:1\nT:\nM:{}/{}\nL:1/{denominator}\nQ:1/4={bpm}\n{}\n{}\nK:{}\n",
        first.meter.0, first.meter.1, HEADERS[0], HEADERS[1], keys[0].1
    );
    let mut previous_meter = first.meter;
    let mut previous_key = keys[0].1.as_str();
    for bar in bars {
        if let Some(section) = sections.get(&bar.start) {
            out.push_str(&format!("% {section}\n"));
        }
        let key = keys
            .iter()
            .rev()
            .find(|(t, _)| *t <= bar.start)
            .unwrap()
            .1
            .as_str();
        let end = bar.start + bar.duration;
        for (voice, name) in VOICES.iter().enumerate() {
            out.push_str(&format!("V: {name}\n"));
            if bar.meter != previous_meter {
                out.push_str(&format!("M:{}/{}\n", bar.meter.0, bar.meter.1));
            }
            if key != previous_key {
                out.push_str(&format!("K:{key}\n"));
            }
            let mut body = String::new();
            if voice == 0 {
                let mut changes = vec![(
                    bar.start,
                    chords
                        .iter()
                        .rev()
                        .find(|(t, _)| *t <= bar.start)
                        .map(|(_, s)| s.as_str()),
                )];
                changes.extend(
                    chords
                        .iter()
                        .filter(|(t, _)| *t > bar.start && *t < end)
                        .map(|(t, s)| (*t, Some(s.as_str()))),
                );
                for (i, (t, symbol)) in changes.iter().enumerate() {
                    if let Some(s) = symbol {
                        body.push_str(&format!("\"{s}\""));
                    }
                    timed_rest(&mut body, changes.get(i + 1).map_or(end, |c| c.0) - t, unit)?;
                }
                if changes.iter().all(|(_, s)| s.is_none()) {
                    body = "Z".into();
                }
            } else {
                let mut cursor = bar.start;
                let mut active = [None; 7];
                for n in notes
                    .iter()
                    .filter(|n| n.start < end && n.start + n.duration > bar.start)
                {
                    let a = n.start.max(bar.start);
                    let b = (n.start + n.duration).min(end);
                    timed_rest(&mut body, a - cursor, unit)?;
                    let lengths = parts(b - a, unit)?;
                    for (i, count) in lengths.iter().enumerate() {
                        body.push_str(&spelling(n.pitch, key, &mut active)?);
                        if *count != 1 {
                            body.push_str(&count.to_string());
                        }
                        if i + 1 < lengths.len() || b < n.start + n.duration {
                            body.push('-');
                        }
                    }
                    cursor = b;
                }
                timed_rest(&mut body, end - cursor, unit)?;
                if cursor == bar.start {
                    body = "Z".into();
                }
            }
            out.push_str(&body);
            out.push_str("|\n");
        }
        previous_meter = bar.meter;
        previous_key = key;
    }
    let parsed = Score::parse(&out)?;
    check(
        parsed.voices[1].notes == notes,
        "Compiled notes changed pitch, onset or duration",
    )?;
    check(
        active_chords(&parsed.voices[0].chords) == active_chords(chords),
        "Compiled harmony changed",
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn score(vocal: &str, ins: &str, key: &str) -> String {
        format!(
            "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n{}\n{}\nK:{key}\nV: Vocal\n{vocal}\nV: Ins\n{ins}\n",
            HEADERS[0], HEADERS[1]
        )
    }
    #[test]
    fn ties_accidentals_and_rests_compare_by_sounding_events() {
        let a = Score::parse(&score("^F32-|F16z16|", "Z2|", "C")).unwrap();
        let b = Score::parse(&score("^F48|", "Z|", "C"));
        assert!(b.is_err());
        assert_eq!(
            a.voices[0].notes,
            vec![Note {
                start: 0,
                pitch: 66,
                duration: 1536
            }]
        );
        let equivalent = Score::parse(&score("^F16-^F16-|F16z16|", "Z|Z|", "C")).unwrap();
        assert!(
            a.compare(&equivalent, None, false)
                .get("match")
                .unwrap()
                .boolean()
                .unwrap()
        );
        assert!(Score::parse(&score("C32-|D32|", "Z2|", "C")).is_err());
    }
    #[test]
    fn chord_removal_preserves_header_quotes_and_both_melodies() {
        let a = Score::parse(&score("\"Dm7\"F16A16|", "C16z16|", "D")).unwrap();
        let text = a.strip_chords(None).unwrap();
        assert!(text.contains(HEADERS[0]));
        let b = Score::parse(&text).unwrap();
        assert!(b.voices[0].chords.is_empty());
        assert_eq!(a.voices[0].notes, b.voices[0].notes);
        assert_eq!(a.voices[1].notes, b.voices[1].notes);
        let selected = Score::parse(&a.strip_chords(Some(0)).unwrap()).unwrap();
        assert!(selected.voices[1].notes.is_empty());
    }
    #[test]
    fn instrumental_transfer_preserves_vocal_and_nonoverlapping_fills() {
        let a = Score::parse(&score("\"D\"F16z16|", "C32|", "D")).unwrap();
        assert!(a.instrumental(true, true).is_err());
        let (text, report) = a.instrumental(false, true).unwrap();
        let b = Score::parse(&text).unwrap();
        assert!(b.voices[0].notes.is_empty());
        assert_eq!(
            b.voices[1].notes,
            vec![
                Note {
                    start: 0,
                    pitch: 66,
                    duration: 512
                },
                Note {
                    start: 512,
                    pitch: 61,
                    duration: 512
                }
            ]
        );
        assert_eq!(
            report
                .get("affected_ins_notes")
                .unwrap()
                .array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            active_chords(&a.voices[0].chords),
            active_chords(&b.voices[0].chords)
        );
        let (text, _) = a.instrumental(false, false).unwrap();
        assert!(Score::parse(&text).unwrap().voices[0].chords.is_empty());
    }
    #[test]
    fn rejects_unsupported_notation_and_mismatched_voice_grids() {
        for notation in [
            "C/2z31|",
            "[CE]32|",
            "(3CDEz29|",
            "C32||",
            "C31|",
            "\"Cgarbage\"C32|",
        ] {
            assert!(
                Score::parse(&score(notation, "Z|", "C")).is_err(),
                "{notation}"
            );
        }
        assert!(Score::parse(&score("C32|", "Z2|", "C")).is_err());
    }
    #[test]
    fn transfer_retains_fine_grid_meter_changes_sections_and_tied_accidentals() {
        let text = format!(
            "X:1\nT:\nM:4/4\nL:1/64\nQ:1/4=90\n{}\n{}\nK:C\n% verse\nV: Vocal\n^F32-^F32-|\nV: Ins\nZ|\n% chorus\nV: Vocal\nM:3/4\nK:G\nF16z32|\nV: Ins\nM:3/4\nK:G\nZ|\n",
            HEADERS[0], HEADERS[1]
        );
        let source = Score::parse(&text).unwrap();
        let (out, _) = source.instrumental(false, true).unwrap();
        let target = Score::parse(&out).unwrap();
        assert_eq!(source.voices[0].notes, target.voices[1].notes);
        assert_eq!(source.voices[0].bars, target.voices[1].bars);
        assert_eq!(source.sections, target.sections);
        let fine =
            Score::parse(&score("Cz3D12z48|", "Z|", "C").replace("L:1/32", "L:1/64")).unwrap();
        let (out, _) = fine.instrumental(false, true).unwrap();
        assert!(out.contains("L:1/64"));
        assert_eq!(
            fine.voices[0].notes,
            Score::parse(&out).unwrap().voices[1].notes
        );
    }
}
