# Native score editing

The `music_tools` helper ports the upstream structural/event checks to Rust.
It accepts the limited two-voice YuE2/SheetSage2 dialect, not arbitrary ABC notation.
Times are exact quarter-note fractions on a power-of-two grid. Comparisons resolve
ties and accidentals into sounding MIDI pitches, onsets and durations.

Preserve this header layout:

```abc
X:1
T:
M:4/4
L:1/32
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
```

The body alternates `V: Vocal`, a line containing 1–4 bars, `V: Ins`, and the
corresponding 1–4 bars. `% verse`, `% chorus`, etc. precede a group. Both voices
must have the same meter/time grid and key-change timeline. Full-measure rests
use `Z`, `Z2`, `Z3`, `Z4`; note/rest duration integers must be one of
`1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48`. Split other durations into supported
pieces and tie sounding notes. Ordinary rests use `z`.

Use explicit fraction meters, `L:1/<power of two>` through `L:1/1024`, integer
quarter-note BPM, standard major/minor keys, plain barlines, and single notes.
The helper rejects tuplets, slash durations, repeats, decorations and polyphonic
note brackets. Do not silently discard unsupported notation. Preserve the source
and record any authorized normalization before supplying native ABC.

Accidentals persist by letter within a bar, including across octaves, matching
the native exporters. Unmarked tied continuations retain their accidental across
a barline. Inline `[K:...]` changes reset active accidentals. An unresolved tie,
tie into a rest, mismatched tie pitch or over/underfull bar fails validation.

Chord symbols belong in Vocal. Supported qualities are major, `m`, `dim`, `aug`,
`7`, `maj7`, `m7`, `dim7`, `m7b5`, `sus4`, `sus2`, `6`, `m6`, `7sus4`, and
`m(maj7)`, with optional slash bass and sharp/flat/double-sharp/double-flat roots.

## Inspect, compare and strip chords

```sh
./target/release/examples/music_tools inspect edits/score.abc --output edits/check.json
./target/release/examples/music_tools compare original.abc edits/score.abc \
  --voices Vocal --output edits/comparison.json
./target/release/examples/music_tools strip-chords original.abc edits/melody.abc
```

Reports include exact notes, meter grids, chords, key timelines and nominal score
duration. `compare` checks both voices by default; `--voices Vocal` or `Ins` selects
one. Tempo changes fail unless `--allow-tempo-change` is supplied. It does not
compare harmony or claim audio fidelity. Exit 0 means match; 1 means a valid
comparison found differences; 2 means a command or notation error.

`strip-chords` keeps both melodies, timing, rests, section structure and header
quotes. `--keep-voice Vocal` or `Ins` silences the other voice with rests, checking
the retained melody exactly. It writes a new file and refuses existing outputs.
Changing a voice's name is not the instrumental voice-transfer operation.

For lyric adaptation, retain phrase lengths when requested and record syllable,
stress and breath choices. A note/syllable sidecar is editorial evidence; the
native model has no phoneme-alignment argument or hard realization guarantee.
