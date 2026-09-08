//! Corpus-wide lex check: every committed fixture
//! lexes with zero `Error` tokens. This is the transcription-fidelity
//! gate for the lexer — when a new fixture or toolchain introduces a
//! token shape we don't handle, this test names the file and the byte.

use ptxroof::parse::lexer::{TokenKind, tokenize};
use std::fs;
use std::path::{Path, PathBuf};

fn collect_ptx(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("fixture dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_ptx(&path, out);
        } else if path.extension().is_some_and(|e| e == "ptx") {
            out.push(path);
        }
    }
}

#[test]
fn every_fixture_lexes_with_zero_error_tokens() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut files = Vec::new();
    collect_ptx(&fixtures, &mut files);
    files.sort();
    assert!(
        files.len() >= 8,
        "fixture corpus missing? found {} .ptx files",
        files.len()
    );

    for path in files {
        let source = fs::read_to_string(&path).expect("fixture readable");
        let errors: Vec<_> = tokenize(&source)
            .into_iter()
            .filter(|t| t.kind == TokenKind::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "{}: {} error token(s), first at byte {} ({:?})",
            path.display(),
            errors.len(),
            errors[0].offset,
            errors[0].text,
        );
    }
}
