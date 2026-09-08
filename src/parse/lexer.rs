//! Lexer for PTX program text.
//!
//! Transcribed from v1's `lib/PTX/Tokenizer.cpp` (the reference spec,
//! v1 lives only in git history
//! now, last at 690d81d), extended from inline-asm bodies to full
//! programs:
//!
//! - directives lex as `Dot` + `Identifier` (`.visible`, `.reg`, `.loc`);
//! - label definitions need `Colon` (`$L__BB0_4:`), and `$`-prefixed
//!   symbols lex as `Identifier` with the `$` kept in the text;
//! - kernel signatures need parens, counted register declarations
//!   (`.reg .b32 %r<38>;`) need angle brackets, `.file`/`.pragma`/
//!   `.section` need string literals;
//! - special registers carry dotted components (`%tid.x`, `%ctaid.y`);
//! - immediates include float bit-patterns (`0f3F800000`,
//!   `0d4030000000000000`) alongside decimal/hex integers.
//!
//! Dropped from v1: the `%N`/`$N` operand-reference and `%%name` escape
//! forms — those exist only in unsubstituted inline-asm bodies, never in
//! full programs (the corpus-wide lex check is the evidence).
//!
//! Identifiers may contain `::` segments (`shared::cluster`,
//! `mbarrier::complete_tx`) so the parser never reassembles them. PTX
//! modifier names may start with digits (`cp.async.bulk.tensor.5d`); a
//! numeric run that continues with identifier characters is promoted to
//! `Identifier` — except hex and float bit-patterns, which stay `Number`.
//!
//! On an unexpected byte the lexer emits one `Error` token and skips the
//! byte; it never panics (ground rule: all frontend paths recover).

/// A single lexed token. `text` borrows from the source buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub text: &'a str,
    /// Byte offset into the source, for diagnostics.
    pub offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// Mnemonic, modifier, directive name, symbol, or `$`-label; may
    /// contain `::` segments.
    Identifier,
    /// `%p1`, `%rd5`, `%tid.x` — register or special-register reference.
    Register,
    /// Integer, float, or float-bit-pattern immediate (sign included).
    Number,
    /// `"..."` literal, quotes included in the text.
    String,
    Dot,
    Comma,
    Semicolon,
    Colon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    /// `<` / `>` — counted register declarations (`%r<38>`).
    Lt,
    Gt,
    Plus,
    Minus,
    /// `@` — predicate prefix.
    At,
    /// `!` — negated predicate.
    Bang,
    /// `|` between destination registers (`setp ... p|q`).
    Pipe,
    EndOfFile,
    /// Unexpected byte; the lexer skips it and continues.
    Error,
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_cont(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

/// Lex an entire PTX program. Always ends with an `EndOfFile` token.
pub fn tokenize(source: &str) -> Vec<Token<'_>> {
    Lexer {
        src: source.as_bytes(),
        source,
        pos: 0,
    }
    .run()
}

struct Lexer<'a> {
    src: &'a [u8],
    source: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn run(mut self) -> Vec<Token<'a>> {
        let mut out = Vec::with_capacity(self.src.len() / 4);
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.src.len() {
                out.push(self.make(TokenKind::EndOfFile, self.pos));
                return out;
            }
            out.push(self.next_token());
        }
    }

    fn peek(&self, ahead: usize) -> u8 {
        *self.src.get(self.pos + ahead).unwrap_or(&0)
    }

    fn make(&self, kind: TokenKind, start: usize) -> Token<'a> {
        Token {
            kind,
            text: &self.source[start..self.pos],
            offset: start as u32,
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.src.len() {
            match self.peek(0) {
                b' ' | b'\t' | b'\n' | b'\r' | b'\x0c' | b'\x0b' => self.pos += 1,
                b'/' if self.peek(1) == b'/' => {
                    while self.pos < self.src.len() && self.peek(0) != b'\n' {
                        self.pos += 1;
                    }
                }
                b'/' if self.peek(1) == b'*' => {
                    self.pos += 2;
                    while self.pos + 1 < self.src.len()
                        && !(self.peek(0) == b'*' && self.peek(1) == b'/')
                    {
                        self.pos += 1;
                    }
                    self.pos = (self.pos + 2).min(self.src.len());
                }
                _ => return,
            }
        }
    }

    fn next_token(&mut self) -> Token<'a> {
        let start = self.pos;
        let c = self.peek(0);

        let punct = |kind| Some(kind);
        let single = match c {
            b'.' => punct(TokenKind::Dot),
            b',' => punct(TokenKind::Comma),
            b';' => punct(TokenKind::Semicolon),
            b':' => punct(TokenKind::Colon),
            b'(' => punct(TokenKind::LParen),
            b')' => punct(TokenKind::RParen),
            b'[' => punct(TokenKind::LBracket),
            b']' => punct(TokenKind::RBracket),
            b'{' => punct(TokenKind::LBrace),
            b'}' => punct(TokenKind::RBrace),
            b'<' => punct(TokenKind::Lt),
            b'>' => punct(TokenKind::Gt),
            b'+' => punct(TokenKind::Plus),
            b'@' => punct(TokenKind::At),
            b'!' => punct(TokenKind::Bang),
            b'|' => punct(TokenKind::Pipe),
            _ => None,
        };
        if let Some(kind) = single {
            self.pos += 1;
            return self.make(kind, start);
        }

        match c {
            b'-' => {
                if self.peek(1).is_ascii_digit() {
                    self.lex_number()
                } else {
                    self.pos += 1;
                    self.make(TokenKind::Minus, start)
                }
            }
            b'"' => self.lex_string(),
            b'%' => self.lex_register(),
            b'$' => self.lex_dollar_symbol(),
            _ if c.is_ascii_digit() => self.lex_number(),
            _ if is_ident_start(c) => self.lex_identifier(),
            _ => {
                self.pos += 1;
                self.make(TokenKind::Error, start)
            }
        }
    }

    /// Identifier with optional `::` segments. The segment after `::`
    /// may start with a digit — PTX cache hints like `L2::128B` need
    /// this (v1 rule, kept).
    fn lex_identifier(&mut self) -> Token<'a> {
        let start = self.pos;
        while self.pos < self.src.len() {
            if is_ident_cont(self.peek(0)) {
                self.pos += 1;
            } else if self.peek(0) == b':' && self.peek(1) == b':' && is_ident_cont(self.peek(2)) {
                self.pos += 2;
            } else {
                break;
            }
        }
        self.make(TokenKind::Identifier, start)
    }

    /// `$L__BB0_4`, `$L__info_string0` — local labels and debug symbols.
    fn lex_dollar_symbol(&mut self) -> Token<'a> {
        let start = self.pos;
        self.pos += 1; // consume $
        if !is_ident_start(self.peek(0)) {
            return self.make(TokenKind::Error, start);
        }
        while is_ident_cont(self.peek(0)) {
            self.pos += 1;
        }
        self.make(TokenKind::Identifier, start)
    }

    /// `%p1`, `%rd5`, and special registers with dotted components
    /// (`%tid.x`). A dot is consumed only when an identifier start
    /// follows, so `%r2,` or `%r<38>` never over-consume.
    fn lex_register(&mut self) -> Token<'a> {
        let start = self.pos;
        self.pos += 1; // consume %
        if !is_ident_start(self.peek(0)) {
            return self.make(TokenKind::Error, start);
        }
        while is_ident_cont(self.peek(0)) {
            self.pos += 1;
        }
        while self.peek(0) == b'.' && is_ident_start(self.peek(1)) {
            self.pos += 1;
            while is_ident_cont(self.peek(0)) {
                self.pos += 1;
            }
        }
        self.make(TokenKind::Register, start)
    }

    /// `"..."`; PTX string literals have no escape sequences.
    fn lex_string(&mut self) -> Token<'a> {
        let start = self.pos;
        self.pos += 1; // consume opening quote
        while self.pos < self.src.len() && self.peek(0) != b'"' && self.peek(0) != b'\n' {
            self.pos += 1;
        }
        if self.peek(0) != b'"' {
            return self.make(TokenKind::Error, start); // unterminated
        }
        self.pos += 1; // consume closing quote
        self.make(TokenKind::String, start)
    }

    /// Numbers: decimal (optionally signed, with `.` fraction and
    /// exponent), hex (`0x1F`), float bit-patterns (`0f3F800000`,
    /// `0d` + 16 hex digits). A numeric run that continues with
    /// identifier characters is a digit-leading modifier name (`5d`,
    /// `64x4`) and is promoted to `Identifier` (v1 rule, kept) —
    /// except hex and bit-patterns, which stay `Number`.
    fn lex_number(&mut self) -> Token<'a> {
        let start = self.pos;
        if self.peek(0) == b'-' {
            self.pos += 1;
        }
        let is_hex = self.peek(0) == b'0' && matches!(self.peek(1), b'x' | b'X');
        if is_hex {
            self.pos += 2;
            while self.peek(0).is_ascii_hexdigit() {
                self.pos += 1;
            }
        } else {
            while self.peek(0).is_ascii_digit() {
                self.pos += 1;
            }
            if self.peek(0) == b'.' && self.peek(1).is_ascii_digit() {
                self.pos += 1;
                while self.peek(0).is_ascii_digit() {
                    self.pos += 1;
                }
            }
            if matches!(self.peek(0), b'e' | b'E') && {
                let s = self.peek(1);
                s.is_ascii_digit() || (matches!(s, b'+' | b'-') && self.peek(2).is_ascii_digit())
            } {
                self.pos += 1;
                if matches!(self.peek(0), b'+' | b'-') {
                    self.pos += 1;
                }
                while self.peek(0).is_ascii_digit() {
                    self.pos += 1;
                }
            }
        }

        let mut promoted = false;
        while is_ident_cont(self.peek(0)) {
            promoted = true;
            self.pos += 1;
        }
        let tok = self.make(TokenKind::Number, start);
        if promoted && !is_hex && !is_float_bit_pattern(tok.text) {
            return Token {
                kind: TokenKind::Identifier,
                ..tok
            };
        }
        tok
    }
}

/// `0f` + 8 hex digits (f32) or `0d` + 16 hex digits (f64), per the
/// PTX ISA's exact-bit float representation.
fn is_float_bit_pattern(text: &str) -> bool {
    let b = text.as_bytes();
    let hex_rest = |n: usize| b.len() == 2 + n && b[2..].iter().all(u8::is_ascii_hexdigit);
    b.len() > 2
        && b[0] == b'0'
        && match b[1] {
            b'f' | b'F' => hex_rest(8),
            b'd' | b'D' => hex_rest(16),
            _ => false,
        }
}

#[cfg(test)]
mod tests {
    use super::TokenKind::*;
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src).iter().map(|t| t.kind).collect()
    }

    fn texts(src: &str) -> Vec<&str> {
        tokenize(src).iter().map(|t| t.text).collect()
    }

    #[test]
    fn instruction_with_modifiers() {
        assert_eq!(
            kinds("fma.rn.f32 %f1, %f2, %f3;"),
            [
                Identifier, Dot, Identifier, Dot, Identifier, Register, Comma, Register, Comma,
                Register, Semicolon, EndOfFile
            ]
        );
    }

    #[test]
    fn predicated_branch_to_local_label() {
        assert_eq!(
            kinds("@!%p2 bra $L__BB0_4;"),
            [
                At, Bang, Register, Identifier, Identifier, Semicolon, EndOfFile
            ]
        );
        assert_eq!(texts("@%p6 bra $L__BB0_4;")[3], "$L__BB0_4");
    }

    #[test]
    fn label_definition_uses_colon() {
        assert_eq!(kinds("$L__BB0_2:"), [Identifier, Colon, EndOfFile]);
    }

    #[test]
    fn scope_qualified_modifier_is_one_identifier() {
        assert_eq!(texts("ld.shared::cta.b32")[2], "shared::cta");
        assert_eq!(texts("ld.global.L2::128B.f32")[4], "L2::128B");
    }

    #[test]
    fn special_register_keeps_dotted_component() {
        assert_eq!(texts("mov.u32 %r21, %ctaid.x;")[3], "%r21");
        assert_eq!(texts("mov.u32 %r21, %ctaid.x;")[5], "%ctaid.x");
        assert_eq!(kinds("%tid.x"), [Register, EndOfFile]);
    }

    #[test]
    fn counted_register_declaration() {
        assert_eq!(
            kinds(".reg .b32 %r<38>;"),
            [
                Dot, Identifier, Dot, Identifier, Register, Lt, Number, Gt, Semicolon, EndOfFile
            ]
        );
    }

    #[test]
    fn numbers_and_bit_patterns() {
        let toks = tokenize("0f3F800000 0d4030000000000000 0x1F -7 1.5 1e+5 42");
        assert!(toks[..7].iter().all(|t| t.kind == Number), "{toks:?}");
        assert_eq!(toks[3].text, "-7");
    }

    #[test]
    fn digit_leading_modifier_promotes_to_identifier() {
        // cp.async.bulk.tensor.5d — `5d` is a modifier name, not a number.
        assert_eq!(*kinds("tensor.5d").last().unwrap(), EndOfFile);
        assert_eq!(kinds("5d"), [Identifier, EndOfFile]);
        assert_eq!(kinds("64x4"), [Identifier, EndOfFile]);
        // ...but a malformed bit-pattern-looking run stays whatever it is:
        assert_eq!(kinds("0f3F80"), [Identifier, EndOfFile]); // 6 hex digits, not 8
    }

    #[test]
    fn string_literals() {
        assert_eq!(
            kinds(".pragma \"nounroll\";"),
            [Dot, Identifier, String, Semicolon, EndOfFile]
        );
        assert_eq!(texts(".file 1 \"a/b.cu\"")[3], "\"a/b.cu\"");
        assert_eq!(kinds("\"unterminated"), [Error, EndOfFile]);
    }

    #[test]
    fn memory_operand_with_negative_offset() {
        assert_eq!(
            kinds("[%rd1+-4]"),
            [LBracket, Register, Plus, Number, RBracket, EndOfFile]
        );
    }

    #[test]
    fn kernel_signature_punctuation() {
        assert_eq!(
            kinds(".entry k(.param .u32 k_param_0)"),
            [
                Dot, Identifier, Identifier, LParen, Dot, Identifier, Dot, Identifier, Identifier,
                RParen, EndOfFile
            ]
        );
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            kinds("// begin inline asm\nadd /* x */ .f32"),
            [Identifier, Dot, Identifier, EndOfFile]
        );
    }

    #[test]
    fn named_constant_and_sink_operand() {
        assert_eq!(kinds("WARP_SZ"), [Identifier, EndOfFile]);
        assert_eq!(kinds("_"), [Identifier, EndOfFile]);
    }

    #[test]
    fn error_token_skips_byte_and_continues() {
        assert_eq!(kinds("&add"), [Error, Identifier, EndOfFile]);
        assert_eq!(kinds("%7"), [Error, Number, EndOfFile]); // operand refs dropped
        assert_eq!(kinds("$ x"), [Error, Identifier, EndOfFile]);
    }

    #[test]
    fn offsets_are_byte_positions() {
        let toks = tokenize("ab  %r1");
        assert_eq!(toks[0].offset, 0);
        assert_eq!(toks[1].offset, 4);
    }
}
