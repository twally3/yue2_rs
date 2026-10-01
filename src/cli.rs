use crate::{
    Result, audio, invalid, json,
    pipeline::{Pipeline, SymbolicPlan},
    protocol::{Cot, SongRequest},
};
use std::{collections::BTreeMap, path::PathBuf};
pub fn run(default_command: &str) -> Result<i32> {
    let mut args = std::env::args().skip(1).peekable();
    let command = if args.peek().is_some_and(|s| !s.starts_with('-')) {
        args.next().unwrap()
    } else {
        default_command.into()
    };
    let mut flags = BTreeMap::new();
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "YuE2 native macOS inference\n\nUsage: yue2_rs [generate|plan|decode] --output PATH [OPTIONS]\n\n  --request FILE          JSON song request (default examples/song.json)\n  --abc-file FILE         External ABC score\n  --cot full|melody|off    Symbolic planning mode\n  --model PATH_OR_ID      YuE2-3B checkpoint (default m-a-p/YuE2-3B)\n  --vae PATH_OR_ID        Stereo VAE checkpoint (default m-a-p/YuE2-Vae)\n  --revision COMMIT       Pin the transformer revision\n  --vae-revision COMMIT   Pin the VAE revision\n  --seed INTEGER          Request-local seed\n  --config FILE           Native generation configuration overrides\n  --plan DIRECTORY        Restore an exact saved plan for generation\n  --latent FILE           FP32 latent.npy for decode\n  --vae-core-frames N      Decoder tile size (default 512)\n\nRun cargo run --release --example generate -- --output outputs/create"
            );
            return Ok(0);
        }
        if ![
            "--output",
            "--request",
            "--abc-file",
            "--cot",
            "--model",
            "--vae",
            "--seed",
            "--config",
            "--plan",
            "--latent",
            "--vae-core-frames",
            "--revision",
            "--vae-revision",
        ]
        .contains(&arg.as_str())
        {
            return Err(invalid(format!("Unknown option {arg}")));
        }
        let value = args
            .next()
            .ok_or_else(|| invalid(format!("Missing value for {arg}")))?;
        if flags.insert(arg, value).is_some() {
            return Err(invalid("Duplicate command-line option"));
        }
    }
    let output = PathBuf::from(
        flags
            .get("--output")
            .ok_or_else(|| invalid("--output is required"))?,
    );
    if output.exists() {
        return Err(invalid("Choose a fresh output directory"));
    }
    let model = flags.get("--model").map_or("m-a-p/YuE2-3B", String::as_str);
    let vae = flags.get("--vae").map_or("m-a-p/YuE2-Vae", String::as_str);
    let core_frames = flags
        .get("--vae-core-frames")
        .map_or(Ok(512), |s| s.parse::<usize>())?;
    if core_frames == 0 {
        return Err(invalid("Decoder tile size must be positive"));
    }
    if command == "decode" {
        let latent = flags
            .get("--latent")
            .ok_or_else(|| invalid("decode requires --latent"))?;
        let vae_path = crate::checkpoint::resolve_model(
            std::path::Path::new(vae),
            flags.get("--vae-revision").map(String::as_str),
        )?;
        crate::pipeline::verify_weights(&vae_path)?;
        eprintln!("Matrix backend: {}", crate::tensor::backend_name());
        let decoder = crate::vae::Vae::open(vae_path, true)?;
        let mut audio = decoder.decode_tiled(&audio::load_tensor(latent)?, core_frames, 16)?;
        for v in &mut audio.data {
            *v = v.clamp(-1., 1.);
        }
        audio::save_flac(&output, &audio, decoder.sample_rate)?;
        println!("{}", output.display());
        return Ok(0);
    }
    let input = if let Some(path) = flags.get("--request") {
        json::read(path)?
    } else {
        json::parse(include_str!("../examples/song.json"))?
    };
    let mut request = SongRequest::from_json(&input)?;
    if let Some(abc) = flags.get("--abc-file") {
        request.abc = Some(std::fs::read_to_string(abc)?);
    }
    if let Some(cot) = flags.get("--cot") {
        request.cot = Cot::parse(cot)?;
    }
    if let Some(seed) = flags.get("--seed") {
        request.seed = seed.parse()?;
    }
    request.validate()?;
    let generation_config = flags
        .get("--config")
        .map(|path| crate::protocol::GenerationConfig::from_json(&json::read(path)?))
        .transpose()?;
    let mut pipe = Pipeline::from_pretrained_revision(
        model,
        vae,
        flags.get("--revision").map(String::as_str),
        flags.get("--vae-revision").map(String::as_str),
    )?;
    if let Some(config) = generation_config {
        pipe.generation_config = config;
    }
    pipe.vae_core_frames = core_frames;
    match command.as_str() {
        "plan" => {
            let plan = pipe.plan(request, None)?;
            plan.save(&output)?;
            println!("{}", output.display());
            Ok(i32::from(plan.truncated))
        }
        "generate" => {
            if let Some(saved) = flags.get("--plan") {
                if ["--abc-file", "--request", "--cot", "--seed"]
                    .iter()
                    .any(|&k| flags.contains_key(k))
                {
                    return Err(invalid(
                        "Exact --plan reuse cannot override its request; pass edited ABC using --abc-file",
                    ));
                }
                let plan = SymbolicPlan::load(saved)?;
                let semantic = pipe.generate_semantic(plan, None)?;
                let latent = pipe.synthesize(&semantic)?;
                let audio = pipe.decode(&latent, false)?;
                let result = pipe.result(semantic, latent, audio);
                result.save_artifacts(&output)?;
                println!("{}", output.join("audio.flac").display());
                return Ok(i32::from(result.truncated()));
            }
            let song = pipe.generate(request)?;
            song.save_artifacts(&output)?;
            println!(
                "{}",
                json::stringify(&json::object([
                    (
                        "audio",
                        output
                            .join("audio.flac")
                            .to_string_lossy()
                            .into_owned()
                            .into()
                    ),
                    (
                        "truncated",
                        json::object([
                            ("abc", song.semantic.plan.truncated.into()),
                            ("semantic", song.semantic.truncated.into())
                        ])
                    )
                ]))
            );
            Ok(i32::from(song.truncated()))
        }
        _ => Err(invalid("Command must be generate, plan or decode")),
    }
}
