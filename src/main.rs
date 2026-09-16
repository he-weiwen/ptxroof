//! Thin CLI shell over the `ptxroof` library.
//!
//! One verb: `analyze` — text report by default, `--json` for the
//! serialized result tree, `--dump-ast` for the parsed-module debug
//! view. Further verbs join the `Command` enum additively as their
//! features land.

use anyhow::Context;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

/// Exit code for operational errors (unreadable input, parse error,
/// bad flags). Mirrors the diff(1)/grep(1) convention: 0 = success,
/// 1 = reserved for "ran fine and the answer is negative", 2 = could
/// not do the job. clap itself exits 2 on usage errors, the same
/// class of problem.
const EXIT_ERROR: u8 = 2;

#[derive(Parser)]
#[command(
    name = "ptxroof",
    version,
    about = "Static roofline analysis for PTX kernels"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Analyze a PTX file: per-loop flops/bytes/AI and verdicts
    Analyze {
        input: PathBuf,
        /// Emit the result tree as JSON instead of the text report
        #[arg(long)]
        json: bool,
        /// Bind a kernel parameter for numeric columns:
        /// `name=value` or `idx:name=value` (params are positional)
        #[arg(long)]
        bind: Vec<String>,
        /// Launch block dimensions `x,y,z` for per-CTA totals;
        /// defaults to .reqntid/.maxntid when the kernel carries one
        #[arg(long)]
        launch: Option<String>,
        /// Print the parsed module in canonical PTX form and exit
        #[arg(long)]
        dump_ast: bool,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("ptxroof: {err:#}");
            ExitCode::from(EXIT_ERROR)
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let Command::Analyze {
        input,
        json,
        bind,
        launch,
        dump_ast,
    } = cli.command;
    let source =
        std::fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?;

    if dump_ast {
        let module = ptxroof::ptx::parse::parser::parse(&source)
            .with_context(|| format!("parsing {}", input.display()))?;
        print!("{}", ptxroof::ptx::print::dump(&module));
        return Ok(ExitCode::SUCCESS);
    }

    let bindings = bind
        .iter()
        .map(|b| ptxroof::report::parse_bind(b))
        .collect::<Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?;
    let launch = launch
        .map(|s| parse_launch(&s))
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let opts = ptxroof::report::AnalyzeOptions { bindings, launch };
    let report = ptxroof::report::analyze(&source, &input.display().to_string(), &opts)
        .with_context(|| format!("analyzing {}", input.display()))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", ptxroof::report::text::render(&report));
    }
    Ok(ExitCode::SUCCESS)
}

/// `x,y,z` (y/z default to 1).
fn parse_launch(text: &str) -> Result<[u32; 3], String> {
    let mut dims = [1u32; 3];
    let parts: Vec<&str> = text.split(',').collect();
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("--launch `{text}`: expected x,y,z"));
    }
    for (slot, part) in dims.iter_mut().zip(&parts) {
        *slot = part
            .trim()
            .parse()
            .map_err(|_| format!("--launch `{text}`: `{part}` is not a number"))?;
    }
    Ok(dims)
}
