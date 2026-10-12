# Instrumental generation and score covers

For a text-only instrumental request, let YuE2 plan the score. Do not substitute
an agent-composed example melody unless the user explicitly asks for that.
Write a request with the user's instrumental style and empty lyrics or just
section tags; use `cot: "full"` for planned harmony or `melody` for free accompaniment.
Include the intended instruments and no vocals in the style.

```sh
./target/release/yue2_rs plan --request work/instrumental.json --output outputs/planning
./target/release/examples/music_tools instrumentalize outputs/planning/score.abc \
  --output outputs/conversion
./target/release/yue2_rs generate --request work/instrumental.json \
  --abc-file outputs/conversion/score.abc --output outputs/instrumental
./target/release/examples/music_tools listen outputs/instrumental --output outputs/listen
```

Check planning truncation before converting; do not render a silently truncated
score. If planning or validation fails, retain the failure and diagnose it. Do not
automatically switch to agent composition. An explicitly requested retry may use
a new seed while retaining the failed plan.

For user-provided ABC, begin with `music_tools inspect`, then convert that score;
do not recompose it. The same applies to an externally transcribed audio reference.
If the user explicitly requests agent composition, write native two-voice ABC,
validate it, then use the conversion and generation steps.

## Transfer invariants

`instrumentalize` resolves actual note events, then:

- Transfers every Vocal pitch, onset and duration into Ins; Vocal contains rests
  and retained chord symbols only.
- Retains original Ins passages that do not overlap Vocal. The default `--overlap
  vocal` clips overlapping Ins events to their uncovered intervals and records
  every affected source note in `transfer.json`.
- Supports `--overlap error` when overlap reduction is not allowed.
- Retains meter, tempo and harmony by default. Use `--drop-chords` and render with
  `--cot melody` for a melody-only instrumental cover.
- Re-parses the compiled score and checks exact note and active-harmony roundtrips.

The fresh conversion directory contains `source.abc`, `score.abc` and
`transfer.json`, including source/output hashes. Ins is a single melody line;
preserving both overlapping voices as polyphony is outside this native dialect.
The compiler selects an exact grid through `L:1/1024` and emits one bar per group.
Inline key changes inside bars require explicit normalization before conversion.

The converted score is a **new external ABC input**. Reusing `--plan` would restore
the original prefix and original voice assignment. After rendering, check that
the saved `score.abc` matches the converted score and the saved request has empty
or section-only lyrics. Deliver the real style prompt, transfer report, audio and
recorded duration. Symbolic conversion does not prove the generated recording is
free of vocals; report listening evidence separately.
