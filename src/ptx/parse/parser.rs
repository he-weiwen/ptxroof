//! Recursive-descent parser: PTX text → flat module IR.
//!
//! Transcribed from v1's `lib/PTX/Parser.cpp` (v1 lives only in git
//! history now, last at 690d81d; statement/operand grammar,
//! per-statement error recovery)
//! and extended from inline-asm bodies to full programs: module
//! directives, kernel signatures with param tables, `.reg`/`.shared`
//! declarations, labels, extended `.loc` (plain and
//! `function_name/inlined_at` forms — both appear in k2), `.pragma` in
//! body position, `.section .debug_*` data sections (parsed and
//! skipped), `{ ... }` statement blocks (inline-asm expansions —
//! flattened, with the labels, registers and variables they declare
//! renamed per scope), and `.branchtargets`.
//!
//! Malformed instructions in kernel bodies become `Stmt::Unparsed`; recovery
//! skips to a semicolon, closing brace, or EOF. Unsupported top-level
//! constructs and some structural errors return `ParseError` with a line
//! number. Some declarations and directives are skipped. This is not full
//! PTX validation, and lexer panics on arbitrary UTF-8 are not caught.

use crate::ptx::ir::{
    FileDirective, FundamentalType, Instr, Kernel, Module, Opcode, Operand, OperandId, Param,
    Predicate, RegDecl, SharedDecl, SourceLoc, SpecialReg, Stmt,
};
use crate::ptx::literal::parse_int;
use crate::ptx::parse::lexer::{Token, TokenKind, tokenize};
use crate::support::index::{Idx, IdxRange, IndexVec};
use crate::support::intern::{Interner, Symbol};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
#[error("parse error at line {line}: {message}")]
pub struct ParseError {
    pub line: u32,
    pub message: String,
}

pub fn parse(source: &str) -> Result<Module, ParseError> {
    Parser::new(source).run()
}

/// An open `{ }` block: "any variable declared within a scope is not
/// available outside the scope" (PTX ISA §9.7.13.1). The names it
/// declares are renamed `name$scopeN`, a counted register family's
/// members `prefix$scopeN$i`.
struct Scope {
    id: u32,
    names: HashMap<Symbol, Symbol>,
    families: Vec<(Symbol, u32)>,
}

struct Parser<'a> {
    source: &'a str,
    toks: Vec<Token<'a>>,
    pos: usize,
    interner: Interner,
    version: (u32, u32),
    target: Option<Symbol>,
    address_size: Option<u32>,
    files: Vec<FileDirective>,
    kernels: Vec<Kernel>,
    shared_decls: Vec<SharedDecl>,
    /// Inlined location → the user line it was ultimately inlined at.
    inlined: HashMap<SourceLoc, SourceLoc>,
    scopes: Vec<Scope>,
    scope_count: u32,
    /// Spliced into renamed names: no token of the source contains it.
    scope_tag: String,
    operands: IndexVec<OperandId, Operand>,
    operand_lists: Vec<OperandId>,
    modifier_pool: Vec<Symbol>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        let toks = tokenize(source);
        let mut scope_tag = "$scope".to_owned();
        while toks.iter().any(|t| t.text.contains(scope_tag.as_str())) {
            scope_tag.push('_');
        }
        Parser {
            source,
            toks,
            pos: 0,
            interner: Interner::default(),
            version: (0, 0),
            target: None,
            address_size: None,
            files: Vec::new(),
            kernels: Vec::new(),
            shared_decls: Vec::new(),
            inlined: HashMap::new(),
            scopes: Vec::new(),
            scope_count: 0,
            scope_tag,
            operands: IndexVec::new(),
            operand_lists: Vec::new(),
            modifier_pool: Vec::new(),
        }
    }

    // -- token helpers ----------------------------------------------------

    fn peek(&self) -> &Token<'a> {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    fn peek_at(&self, ahead: usize) -> &Token<'a> {
        &self.toks[(self.pos + ahead).min(self.toks.len() - 1)]
    }

    fn bump(&mut self) -> Token<'a> {
        let t = *self.peek();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        let offset = self.peek().offset as usize;
        let line = self.source[..offset.min(self.source.len())]
            .bytes()
            .filter(|&b| b == b'\n')
            .count() as u32
            + 1;
        ParseError {
            line,
            message: message.into(),
        }
    }

    fn expect(&mut self, kind: TokenKind, what: &str) -> Result<Token<'a>, ParseError> {
        if self.at(kind) {
            Ok(self.bump())
        } else {
            Err(self.err(format!("expected {what}, found {:?}", self.peek().text)))
        }
    }

    fn expect_ident(&mut self, what: &str) -> Result<Symbol, ParseError> {
        let t = self.expect(TokenKind::Identifier, what)?;
        Ok(self.interner.intern(t.text))
    }

    fn expect_u32(&mut self, what: &str) -> Result<u32, ParseError> {
        let t = self.expect(TokenKind::Number, what)?;
        parse_int(t.text)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| self.err(format!("{what}: bad number {:?}", t.text)))
    }

    // -- top level ----------------------------------------------------------

    fn run(mut self) -> Result<Module, ParseError> {
        while !self.at(TokenKind::EndOfFile) {
            self.expect(TokenKind::Dot, "a top-level directive")?;
            let name = self.expect_ident("directive name")?;
            match self.interner.resolve(name) {
                "version" => {
                    let t = self.expect(TokenKind::Number, ".version number")?;
                    self.version = parse_version(t.text)
                        .ok_or_else(|| self.err(format!("bad .version {:?}", t.text)))?;
                }
                "target" => {
                    let first = self.expect_ident(".target architecture")?;
                    self.target = Some(first);
                    while self.eat(TokenKind::Comma) {
                        self.expect_ident(".target option")?;
                    }
                }
                "address_size" => {
                    self.address_size = Some(self.expect_u32(".address_size")?);
                }
                "file" => {
                    let index = self.expect_u32(".file index")?;
                    let t = self.expect(TokenKind::String, ".file path")?;
                    let path = self.interner.intern(t.text.trim_matches('"'));
                    if self.eat(TokenKind::Comma) {
                        self.expect(TokenKind::Number, ".file timestamp")?;
                        self.expect(TokenKind::Comma, "',' before the .file size")?;
                        self.expect(TokenKind::Number, ".file size")?;
                    }
                    self.files.push(FileDirective { index, path });
                }
                "section" => {
                    // `.section .debug_str { ... }` — data sections are
                    // parsed-and-skipped.
                    self.expect(TokenKind::Dot, ".section name")?;
                    self.expect_ident(".section name")?;
                    self.skip_balanced_braces()?;
                }
                "visible" | "weak" | "extern" | "common" => {
                    // Linkage prefix; the next directive carries the meat.
                }
                "shared" => match self.parse_shared_decl() {
                    Some(decl) => self.shared_decls.push(decl),
                    None => self.resync_to_semicolon(),
                },
                "global" => {
                    // Module-scope global variable. LLVM's NVPTX backend (Triton,
                    // libdevice) emits e.g. `.global .align 1 .b8 _$_str[11] = {..};`
                    // for its FTZ marker string: a declaration, not analysis
                    // content, so parse-and-discard to the semicolon, stepping
                    // over the brace-delimited initializer as a unit.
                    while !self.at(TokenKind::Semicolon) && !self.at(TokenKind::EndOfFile) {
                        if self.at(TokenKind::LBrace) {
                            self.skip_balanced_braces()?;
                        } else {
                            self.bump();
                        }
                    }
                    self.eat(TokenKind::Semicolon);
                }
                "func" => {
                    while !self.at(TokenKind::Semicolon)
                        && !self.at(TokenKind::LBrace)
                        && !self.at(TokenKind::EndOfFile)
                    {
                        self.bump();
                    }
                    if self.at(TokenKind::LBrace) {
                        self.skip_balanced_braces()?;
                    } else {
                        self.expect(TokenKind::Semicolon, "';' after a .func declaration")?;
                    }
                }
                "entry" => {
                    let kernel = self.parse_kernel()?;
                    self.kernels.push(kernel);
                }
                other => {
                    return Err(self.err(format!(
                        "unknown top-level directive .{other} — if this is a new \
                         producer idiom it needs a parser arm and a fixture"
                    )));
                }
            }
        }

        let target = self.target.unwrap_or_else(|| self.interner.intern(""));
        Ok(Module {
            interner: self.interner,
            version: self.version,
            target,
            address_size: self.address_size,
            files: self.files,
            kernels: self.kernels,
            operands: self.operands,
            operand_lists: self.operand_lists,
            modifier_pool: self.modifier_pool,
            shared_decls: self.shared_decls,
        })
    }

    fn skip_balanced_braces(&mut self) -> Result<(), ParseError> {
        self.expect(TokenKind::LBrace, "'{'")?;
        let mut depth = 1u32;
        loop {
            match self.peek().kind {
                TokenKind::LBrace => depth += 1,
                TokenKind::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        self.bump();
                        return Ok(());
                    }
                }
                TokenKind::EndOfFile => return Err(self.err("unclosed '{'")),
                _ => {}
            }
            self.bump();
        }
    }

    // -- kernels ------------------------------------------------------------

    fn parse_kernel(&mut self) -> Result<Kernel, ParseError> {
        let name = self.expect_ident("kernel name")?;
        let mut kernel = Kernel {
            name,
            params: Vec::new(),
            reg_decls: Vec::new(),
            shared_decls: Vec::new(),
            maxntid: None,
            reqntid: None,
            tuning: Vec::new(),
            stmts: Vec::new(),
        };

        if self.eat(TokenKind::LParen) {
            while !self.at(TokenKind::RParen) {
                kernel.params.push(self.parse_param()?);
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            self.expect(TokenKind::RParen, "')' after params")?;
        }

        // Performance directives between signature and body.
        while self.at(TokenKind::Dot) {
            let save = self.pos;
            self.bump();
            let name = self.expect_ident("kernel directive")?;
            match self.interner.resolve(name) {
                "maxntid" => kernel.maxntid = Some(self.parse_dim3()?),
                "reqntid" => kernel.reqntid = Some(self.parse_dim3()?),
                "maxnreg" | "minnctapersm" | "maxnctapersm" => {
                    let value = self.expect_u32("directive argument")?;
                    kernel.tuning.push((name, value));
                }
                _ => {
                    self.pos = save;
                    break;
                }
            }
        }

        self.expect(TokenKind::LBrace, "'{' starting kernel body")?;
        let first = self.operands.len();
        self.parse_body(&mut kernel)?;
        for stmt in &mut kernel.stmts {
            if let Stmt::Unparsed { registers, .. } = stmt {
                let decls = &kernel.reg_decls;
                registers.retain(|&r| declares(&self.interner, decls, self.interner.resolve(r)));
            }
        }
        for id in (first..self.operands.len()).map(OperandId::from_usize) {
            match self.operands[id] {
                Operand::Register(sym)
                    if let name = self.interner.resolve(sym)
                        && let Some(reg) = SpecialReg::from_name(name)
                        && !declares(&self.interner, &kernel.reg_decls, name) =>
                {
                    self.operands[id] = Operand::Special(reg);
                }
                // PTX ISA §4.4: a register's name need not start with `%`.
                Operand::SymbolRef(sym)
                    if declares(
                        &self.interner,
                        &kernel.reg_decls,
                        self.interner.resolve(sym),
                    ) =>
                {
                    self.operands[id] = Operand::Register(sym);
                }
                _ => {}
            }
        }
        Ok(kernel)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        self.expect(TokenKind::Dot, "'.param'")?;
        let kw = self.expect_ident("'.param'")?;
        if self.interner.resolve(kw) != "param" {
            return Err(self.err("expected .param in signature"));
        }
        let ty;
        let mut align = None;
        loop {
            self.expect(TokenKind::Dot, "param type")?;
            let attr = self.expect_ident("param attribute")?;
            match self.interner.resolve(attr) {
                "align" => {
                    align = Some(self.expect_u32(".align value")?);
                }
                other => {
                    ty = FundamentalType::from_name(other)
                        .ok_or_else(|| self.err(format!("unknown param type .{other}")))?;
                    break;
                }
            }
        }
        // Pointer attributes after the type: `.ptr [.global|.shared|.local|.const] [.align N]`
        // (LLVM's NVPTX backend, e.g. Triton: `.param .u64 .ptr .global .align 1 p0`).
        while self.at(TokenKind::Dot) {
            self.expect(TokenKind::Dot, "param attribute")?;
            let attr = self.expect_ident("param attribute")?;
            match self.interner.resolve(attr) {
                "ptr" | "global" | "shared" | "local" | "const" => {}
                "align" => {
                    self.expect_u32(".align value")?;
                }
                other => {
                    return Err(self.err(format!("unexpected param attribute .{other}")));
                }
            }
        }
        let name = self.expect_ident("param name")?;
        let mut size = None;
        if self.eat(TokenKind::LBracket) {
            if self.at(TokenKind::Number) {
                size = Some(self.expect_u32("param array size")?);
            }
            self.expect(TokenKind::RBracket, "']' after param array size")?;
        }
        Ok(Param {
            ty,
            name,
            align,
            size,
        })
    }

    fn parse_dim3(&mut self) -> Result<[u32; 3], ParseError> {
        let mut dims = [1u32; 3];
        dims[0] = self.expect_u32("launch dim")?;
        for d in dims.iter_mut().skip(1) {
            if !self.eat(TokenKind::Comma) {
                break;
            }
            *d = self.expect_u32("launch dim")?;
        }
        Ok(dims)
    }

    // -- kernel bodies --------------------------------------------------------

    fn parse_body(&mut self, kernel: &mut Kernel) -> Result<(), ParseError> {
        // Depth of flattened `{ ... }` statement blocks (inline-asm
        // expansions); the kernel's own closing brace is depth 0.
        let mut block_depth = 0u32;
        let mut cur_loc: Option<SourceLoc> = None;

        loop {
            match self.peek().kind {
                TokenKind::RBrace => {
                    self.bump();
                    if block_depth == 0 {
                        return Ok(());
                    }
                    block_depth -= 1;
                    self.scopes.pop();
                }
                TokenKind::LBrace => {
                    self.bump();
                    block_depth += 1;
                    self.enter_scope();
                }
                TokenKind::Semicolon => {
                    self.bump();
                }
                TokenKind::EndOfFile => {
                    return Err(self.err("unexpected end of file inside kernel body"));
                }
                TokenKind::Identifier if self.peek_at(1).kind == TokenKind::Colon => {
                    let text = self.peek().text;
                    let label = self.interner.intern(text);
                    let label = self.scoped(label);
                    self.bump();
                    self.bump();
                    kernel.stmts.push(Stmt::Label(label));
                }
                TokenKind::Dot => {
                    self.parse_body_directive(kernel, &mut cur_loc)?;
                }
                _ => {
                    let stmt = self.parse_instruction(cur_loc);
                    kernel.stmts.push(stmt);
                }
            }
        }
    }

    /// A `{ }` block just opened: find the labels it defines (at its own
    /// depth) and give them scope-unique names, so a branch inside the
    /// block resolves to its own label and not a sibling scope's.
    fn enter_scope(&mut self) {
        self.scope_count += 1;
        let id = self.scope_count;
        let mut depth = 1u32;
        let mut names = Vec::new();
        let mut k = 0;
        loop {
            let tok = self.peek_at(k);
            match tok.kind {
                TokenKind::LBrace => depth += 1,
                TokenKind::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                TokenKind::EndOfFile => break,
                TokenKind::Identifier
                    if depth == 1 && self.peek_at(k + 1).kind == TokenKind::Colon =>
                {
                    names.push(tok.text);
                }
                _ => {}
            }
            k += 1;
        }
        let map = names
            .into_iter()
            .map(|n| {
                let renamed = format!("{n}{}{id}", self.scope_tag);
                (self.interner.intern(n), self.interner.intern(&renamed))
            })
            .collect();
        self.scopes.push(Scope {
            id,
            names: map,
            families: Vec::new(),
        });
    }

    /// The name `sym` refers to: its innermost open block's, if one
    /// declares it.
    fn scoped(&mut self, sym: Symbol) -> Symbol {
        let name = self.interner.resolve(sym);
        let renamed = self.scopes.iter().rev().find_map(|s| {
            if let Some(&r) = s.names.get(&sym) {
                return Some(self.interner.resolve(r).to_owned());
            }
            s.families.iter().find_map(|&(prefix, n)| {
                let prefix = self.interner.resolve(prefix);
                let i = member(name, prefix, n)?;
                Some(format!("{prefix}{}{}${i}", self.scope_tag, s.id))
            })
        });
        renamed.map_or(sym, |r| self.interner.intern(&r))
    }

    /// A name declared in an open block, a register family's prefix with
    /// its count, takes a name of its own there.
    fn declare_scoped(&mut self, name: Symbol, count: Option<u32>) -> Symbol {
        let Some(scope) = self.scopes.last_mut() else {
            return name;
        };
        let text = self.interner.resolve(name);
        let renamed = match count {
            None => format!("{text}{}{}", self.scope_tag, scope.id),
            Some(_) => format!("{text}{}{}$", self.scope_tag, scope.id),
        };
        let renamed = self.interner.intern(&renamed);
        match count {
            None => {
                scope.names.insert(name, renamed);
            }
            Some(n) => scope.families.push((name, n)),
        }
        renamed
    }

    fn parse_body_directive(
        &mut self,
        kernel: &mut Kernel,
        cur_loc: &mut Option<SourceLoc>,
    ) -> Result<(), ParseError> {
        let offset = self.peek().offset;
        self.bump(); // consume '.'
        let Ok(name) = self.expect_ident("body directive name") else {
            kernel.stmts.push(Stmt::Unparsed {
                offset,
                registers: Vec::new(),
            });
            self.resync_to_semicolon();
            return Ok(());
        };
        match self.interner.resolve(name) {
            "loc" => self.parse_loc(cur_loc),
            "reg" => {
                if let Some(mut decls) = self.parse_reg_decl() {
                    for d in &mut decls {
                        d.prefix = self.declare_scoped(d.prefix, d.count);
                    }
                    kernel.reg_decls.extend(decls);
                } else {
                    kernel.stmts.push(Stmt::Unparsed {
                        offset,
                        registers: Vec::new(),
                    });
                    self.resync_to_semicolon();
                }
            }
            "shared" => {
                if let Some(mut decl) = self.parse_shared_decl() {
                    decl.name = self.declare_scoped(decl.name, None);
                    kernel.shared_decls.push(decl);
                } else {
                    kernel.stmts.push(Stmt::Unparsed {
                        offset,
                        registers: Vec::new(),
                    });
                    self.resync_to_semicolon();
                }
            }
            "extern" => {
                // `.extern .shared .align A .b8 name[];` — dynamic smem.
                if self.eat(TokenKind::Dot)
                    && self.at(TokenKind::Identifier)
                    && self.peek().text == "shared"
                {
                    self.bump();
                    if let Some(mut decl) = self.parse_shared_decl() {
                        decl.name = self.declare_scoped(decl.name, None);
                        kernel.shared_decls.push(decl);
                        return Ok(());
                    }
                }
                kernel.stmts.push(Stmt::Unparsed {
                    offset,
                    registers: Vec::new(),
                });
                self.resync_to_semicolon();
            }
            "pragma" => {
                // `.pragma "nounroll";` — annotation, no analysis content.
                self.resync_to_semicolon();
            }
            "branchtargets" => {
                let mut targets = Vec::new();
                while self.at(TokenKind::Identifier) {
                    let t = self.bump();
                    let sym = self.interner.intern(t.text);
                    targets.push(self.scoped(sym));
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.resync_to_semicolon();
                kernel.stmts.push(Stmt::BranchTargets(targets));
            }
            "calltargets" | "callprototype" => self.resync_to_semicolon(),
            "local" | "const" | "global" | "param" | "align" => {
                // In-body variable declarations we don't model; the
                // instructions touching them are still counted, and in
                // a block their names hide the kernel's.
                let mut after_dot = true;
                while !matches!(
                    self.peek().kind,
                    TokenKind::Semicolon
                        | TokenKind::Equals
                        | TokenKind::RBrace
                        | TokenKind::EndOfFile
                ) {
                    let t = self.bump();
                    if !after_dot && matches!(t.kind, TokenKind::Identifier | TokenKind::Register) {
                        let name = self.interner.intern(t.text);
                        self.declare_scoped(name, None);
                    }
                    after_dot = t.kind == TokenKind::Dot;
                }
                self.resync_to_semicolon();
            }
            _ => {
                kernel.stmts.push(Stmt::Unparsed {
                    offset,
                    registers: Vec::new(),
                });
                self.resync_to_semicolon();
            }
        }
        Ok(())
    }

    /// `.loc f l c` or
    /// `.loc f l c, function_name $sym, inlined_at f l c`.
    /// The effective location is the `inlined_at` one when present,
    /// followed through earlier `.loc`s when that site was itself
    /// inlined (LLVM writes the chain outer-first): attribution wants
    /// the user's source line, not the header line the code came from.
    fn parse_loc(&mut self, cur_loc: &mut Option<SourceLoc>) {
        let Some(inner) = self.parse_loc_triple() else {
            return;
        };
        let mut loc = inner;
        while self.at(TokenKind::Comma) {
            self.bump();
            if !self.at(TokenKind::Identifier) {
                break;
            }
            let kw = self.bump().text;
            match kw {
                "function_name" => {
                    if self.at(TokenKind::Identifier) {
                        self.bump();
                    }
                }
                "inlined_at" => {
                    if let Some(outer) = self.parse_loc_triple() {
                        loc = self.inlined.get(&outer).copied().unwrap_or(outer);
                        self.inlined.insert(inner, loc);
                    }
                }
                _ => break,
            }
        }
        *cur_loc = Some(loc);
    }

    fn parse_loc_triple(&mut self) -> Option<SourceLoc> {
        let next = |p: &mut Self| -> Option<u32> {
            if p.at(TokenKind::Number) {
                parse_int(p.bump().text)?.try_into().ok()
            } else {
                None
            }
        };
        let file = next(self)?;
        let line = next(self)?;
        let col = next(self)?;
        Some(SourceLoc { file, line, col })
    }

    /// `.reg .f32 %f<789>;`, scoped `.reg .pred p;`, or a list
    /// `.reg .b16 lo, hi;`.
    fn parse_reg_decl(&mut self) -> Option<Vec<RegDecl>> {
        if !self.eat(TokenKind::Dot) {
            return None;
        }
        if !self.at(TokenKind::Identifier) {
            return None;
        }
        let class = FundamentalType::from_name(self.bump().text)?;
        let mut decls = Vec::new();
        loop {
            let prefix = match self.peek().kind {
                TokenKind::Register | TokenKind::Identifier => {
                    let t = self.bump();
                    self.interner.intern(t.text)
                }
                _ => return None,
            };
            let mut count = None;
            if self.eat(TokenKind::Lt) {
                count = parse_int(self.peek().text).and_then(|v| u32::try_from(v).ok());
                if !self.at(TokenKind::Number) {
                    return None;
                }
                self.bump();
                if !self.eat(TokenKind::Gt) {
                    return None;
                }
            }
            decls.push(RegDecl {
                class,
                prefix,
                count,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.eat(TokenKind::Semicolon).then_some(decls)
    }

    /// After `.shared`: `.align A .b8 name[N];` (size empty for the
    /// `.extern` dynamic form).
    fn parse_shared_decl(&mut self) -> Option<SharedDecl> {
        let mut align = None;
        let mut ty = None;
        while self.eat(TokenKind::Dot) {
            let attr = if self.at(TokenKind::Identifier) {
                self.bump().text
            } else {
                return None;
            };
            if attr == "align" {
                if !self.at(TokenKind::Number) {
                    return None;
                }
                align = parse_int(self.bump().text).and_then(|v| u32::try_from(v).ok());
            } else {
                ty = Some(FundamentalType::from_name(attr)?);
                break;
            }
        }
        let name = if self.at(TokenKind::Identifier) {
            let t = self.bump();
            self.interner.intern(t.text)
        } else {
            return None;
        };
        let mut size = None;
        if self.eat(TokenKind::LBracket) {
            if self.at(TokenKind::Number) {
                size = parse_int(self.bump().text).and_then(|v| u64::try_from(v).ok());
            }
            if !self.eat(TokenKind::RBracket) {
                return None;
            }
        }
        self.eat(TokenKind::Semicolon).then_some(SharedDecl {
            name,
            align,
            size,
            ty: ty?,
        })
    }

    // -- instructions -----------------------------------------------------

    fn resync_to_semicolon(&mut self) {
        while !self.at(TokenKind::Semicolon) && !self.at(TokenKind::EndOfFile) {
            // Never run past the end of the kernel body on a malformed
            // final statement.
            if self.at(TokenKind::RBrace) {
                return;
            }
            self.bump();
        }
        self.eat(TokenKind::Semicolon);
    }

    fn parse_instruction(&mut self, loc: Option<SourceLoc>) -> Stmt {
        let offset = self.peek().offset;
        let start = self.pos;
        let unparsed = |p: &mut Self| {
            p.resync_to_semicolon();
            let names: Vec<&str> = (p.toks[start..p.pos].iter())
                .filter(|t| matches!(t.kind, TokenKind::Register | TokenKind::Identifier))
                .map(|t| t.text.split('.').next().unwrap_or(t.text))
                .collect();
            let registers = (names.into_iter())
                .map(|n| {
                    let sym = p.interner.intern(n);
                    p.scoped(sym)
                })
                .collect();
            Stmt::Unparsed { offset, registers }
        };

        // Optional predicate: @p / @!p; the name may be a register
        // (`%p6`) or a scoped-declaration identifier (`p`).
        let mut predicate = None;
        if self.eat(TokenKind::At) {
            let negated = self.eat(TokenKind::Bang);
            let reg = match self.peek().kind {
                TokenKind::Register | TokenKind::Identifier => {
                    let t = self.bump();
                    let sym = self.interner.intern(t.text);
                    self.scoped(sym)
                }
                _ => return unparsed(self),
            };
            predicate = Some(Predicate { reg, negated });
        }

        if !self.at(TokenKind::Identifier) {
            return unparsed(self);
        }
        let Some(mnemonic) = Opcode::from_name(self.bump().text) else {
            return unparsed(self);
        };

        // Modifiers: '.' identifier (digit-leading names like `5d` were
        // already promoted to identifiers by the lexer; plain numeric
        // modifiers do not occur, but v1 accepted them — kept).
        let mod_start = self.modifier_pool.len() as u32;
        while self.eat(TokenKind::Dot) {
            match self.peek().kind {
                TokenKind::Identifier | TokenKind::Number => {
                    let t = self.bump();
                    let sym = self.interner.intern(t.text);
                    self.modifier_pool.push(sym);
                }
                _ => {
                    self.modifier_pool.truncate(mod_start as usize);
                    return unparsed(self);
                }
            }
        }
        let modifiers = IdxRange::new(mod_start, self.modifier_pool.len() as u32 - mod_start);

        // Operand list (until ';').
        let mut ids = Vec::new();
        if !self.at(TokenKind::Semicolon) && !self.at(TokenKind::RBrace) {
            loop {
                match self.parse_operand() {
                    Some(id) => ids.push(id),
                    None => return unparsed(self),
                }
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
        }
        if !self.eat(TokenKind::Semicolon) {
            return unparsed(self);
        }

        let start = self.operand_lists.len() as u32;
        self.operand_lists.extend(ids);
        let operands = IdxRange::new(start, self.operand_lists.len() as u32 - start);
        Stmt::Instr(Instr {
            mnemonic,
            modifiers,
            operands,
            predicate,
            loc,
            offset,
        })
    }

    fn push_operand(&mut self, op: Operand) -> OperandId {
        self.operands.push(op)
    }

    fn parse_operand(&mut self) -> Option<OperandId> {
        let first = self.parse_single_operand()?;
        if !self.at(TokenKind::Pipe) {
            return Some(first);
        }
        let mut children = vec![first];
        while self.eat(TokenKind::Pipe) {
            children.push(self.parse_single_operand()?);
        }
        let start = self.operand_lists.len() as u32;
        self.operand_lists.extend(children);
        let span = IdxRange::new(start, self.operand_lists.len() as u32 - start);
        Some(self.push_operand(Operand::MultipleDestinations { children: span }))
    }

    fn parse_single_operand(&mut self) -> Option<OperandId> {
        match self.peek().kind {
            TokenKind::Register => {
                let t = self.bump();
                let (name, selector) = match t.text.split_once('.') {
                    Some((name, sel @ ("b0" | "b1" | "b2" | "b3" | "h0" | "h1"))) => {
                        (name, Some(sel))
                    }
                    _ => (t.text, None),
                };
                let sym = self.interner.intern(name);
                let sym = self.scoped(sym);
                let base = self.push_operand(Operand::Register(sym));
                Some(match selector {
                    Some(sel) => {
                        let selector = self.interner.intern(sel);
                        self.push_operand(Operand::Selected { base, selector })
                    }
                    None => base,
                })
            }
            TokenKind::Number => {
                let t = self.bump();
                let sym = self.interner.intern(t.text);
                Some(self.push_operand(Operand::Immediate(sym)))
            }
            TokenKind::Identifier => {
                let t = self.bump();
                let sym = self.interner.intern(t.text);
                let sym = self.scoped(sym);
                let base = self.push_operand(Operand::SymbolRef(sym));
                if !(self.at(TokenKind::Plus) && self.peek_at(1).kind == TokenKind::Number) {
                    return Some(base);
                }
                self.bump();
                let offset = parse_int(self.bump().text)?;
                Some(self.push_operand(Operand::Offset { base, offset }))
            }
            TokenKind::LBracket => {
                self.bump();
                let base = match self.peek().kind {
                    TokenKind::Register => {
                        let t = self.bump();
                        let sym = self.interner.intern(t.text);
                        Operand::Register(self.scoped(sym))
                    }
                    TokenKind::Identifier => {
                        let t = self.bump();
                        let sym = self.interner.intern(t.text);
                        Operand::SymbolRef(self.scoped(sym))
                    }
                    // `[immAddr]` (PTX ISA §6.4.1).
                    TokenKind::Number => {
                        let t = self.bump();
                        Operand::Immediate(self.interner.intern(t.text))
                    }
                    _ => return None,
                };
                let base = self.push_operand(base);
                let mut offset = 0i64;
                if self.eat(TokenKind::Plus) {
                    if !self.at(TokenKind::Number) {
                        return None;
                    }
                    offset = parse_int(self.bump().text)?;
                }
                let mut rest = Vec::new();
                while self.eat(TokenKind::Comma) {
                    rest.push(self.parse_single_operand()?);
                }
                if !self.eat(TokenKind::RBracket) {
                    return None;
                }
                let start = self.operand_lists.len() as u32;
                self.operand_lists.extend(rest);
                let rest = IdxRange::new(start, self.operand_lists.len() as u32 - start);
                let unified = self.at(TokenKind::Dot) && self.peek_at(1).text == "unified";
                if unified {
                    self.bump();
                    self.bump();
                }
                Some(self.push_operand(Operand::Memory {
                    base,
                    offset,
                    rest,
                    unified,
                }))
            }
            TokenKind::LParen => {
                self.bump();
                let mut children = Vec::new();
                while !self.eat(TokenKind::RParen) {
                    children.push(self.parse_single_operand()?);
                    if !self.eat(TokenKind::Comma) && !self.at(TokenKind::RParen) {
                        return None;
                    }
                }
                let start = self.operand_lists.len() as u32;
                self.operand_lists.extend(children);
                let span = IdxRange::new(start, self.operand_lists.len() as u32 - start);
                Some(self.push_operand(Operand::ParamList { children: span }))
            }
            TokenKind::Bang => {
                self.bump();
                let inner = self.parse_single_operand()?;
                Some(self.push_operand(Operand::Negated { inner }))
            }
            TokenKind::LBrace => {
                self.bump();
                let mut children = Vec::new();
                if !self.at(TokenKind::RBrace) {
                    loop {
                        children.push(self.parse_single_operand()?);
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                if !self.eat(TokenKind::RBrace) {
                    return None;
                }
                let start = self.operand_lists.len() as u32;
                self.operand_lists.extend(children);
                let span = IdxRange::new(start, self.operand_lists.len() as u32 - start);
                Some(self.push_operand(Operand::VectorList { children: span }))
            }
            _ => None,
        }
    }
}

fn declares(interner: &Interner, decls: &[RegDecl], name: &str) -> bool {
    decls.iter().any(|d| {
        let prefix = interner.resolve(d.prefix);
        match d.count {
            None => prefix == name,
            Some(n) => member(name, prefix, n).is_some(),
        }
    })
}

/// `name`'s index in `.reg prefix<count>`, which declares `prefix0` to
/// `prefix{count-1}` spelled in decimal.
fn member(name: &str, prefix: &str, count: u32) -> Option<u32> {
    let i: u32 = name.strip_prefix(prefix)?.parse().ok()?;
    (i < count && name.len() == prefix.len() + i.to_string().len()).then_some(i)
}

fn parse_version(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap_body(body: &str) -> String {
        format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        )
    }

    fn parse_body(body: &str) -> Module {
        parse(&wrap_body(body)).expect("test body parses")
    }

    fn instrs(module: &Module) -> Vec<&Instr> {
        module.kernels[0]
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::Instr(i) => Some(i),
                _ => None,
            })
            .collect()
    }

    fn unparsed_count(module: &Module) -> usize {
        module.kernels[0]
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Unparsed { .. }))
            .count()
    }

    #[test]
    fn module_header() {
        let m = parse_body("ret;");
        assert_eq!(m.version, (8, 7));
        assert_eq!(m.interner.resolve(m.target), "sm_80");
        assert_eq!(m.address_size, Some(64));
    }

    #[test]
    fn module_scope_functions_are_skipped() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .extern .func  (.param .b32 func_retval0) vprintf\n(\n\
                   .param .b64 vprintf_param_0,\n.param .b64 vprintf_param_1\n)\n;\n\
                   .func  (.param .b32 func_retval0) _Z2sqf(\n.param .b32 _Z2sqf_param_0\n)\n\
                   {\n.reg .f32 %f<3>;\nld.param.f32 %f1, [_Z2sqf_param_0];\n\
                   mul.f32 %f2, %f1, %f1;\nst.param.f32 [func_retval0+0], %f2;\nret;\n}\n\
                   .visible .entry k()\n{\nret;\n}\n";
        let m = parse(src).expect("nvcc's .func declaration and definition parse");
        assert_eq!(m.kernels.len(), 1);
        assert_eq!(m.interner.resolve(m.kernels[0].name), "k");
    }

    #[test]
    fn special_registers_unless_a_reg_reuses_the_name() {
        use crate::ptx::ir::{Element, SpecialScalar, SpecialVector};
        let m = parse_body(
            ".reg .b32 %tid;\n.reg .b32 %pm<8>;\nmov.u32 %r1, %laneid;\n\
             mov.u32 %r2, %ctaid.r;\nmov.u32 %tid, %pm3;\nret;",
        );
        let source = |i: usize| m.operand(m.operand_ids(instrs(&m)[i].operands)[1]).clone();
        assert_eq!(
            source(0),
            Operand::Special(SpecialReg::Scalar(SpecialScalar::Laneid))
        );
        assert_eq!(
            source(1),
            Operand::Special(SpecialReg::Vector(SpecialVector::Ctaid, Some(Element::X)))
        );
        let dest = m.operand(m.operand_ids(instrs(&m)[2].operands)[0]);
        assert!(matches!(dest, Operand::Register(r) if m.interner.resolve(*r) == "%tid"));
        assert!(matches!(source(2), Operand::Register(r) if m.interner.resolve(r) == "%pm3"));
    }

    #[test]
    fn unified_address_is_kept() {
        let m = parse_body("ld.b32 %r1, [%rd1].unified;\nret;");
        assert_eq!(unparsed_count(&m), 0);
        let address = m.operand(m.operand_ids(instrs(&m)[0].operands)[1]);
        assert!(
            matches!(address, Operand::Memory { unified: true, .. }),
            "{address:?}"
        );
    }

    #[test]
    fn mnemonics_outside_the_isa_are_unparsed() {
        let m = parse_body("nop;\nfuture.op %r1;\nret;");
        assert_eq!(unparsed_count(&m), 2);
        assert_eq!(instrs(&m)[0].mnemonic, Opcode::Ret);
    }

    #[test]
    fn opaque_params_are_a_parse_error() {
        let src =
            ".version 8.7\n.target sm_80\n.visible .entry k(\n.param .texref t\n)\n{\nret;\n}\n";
        let err = parse(src).expect_err(".texref is not a fundamental type");
        assert!(
            err.to_string().contains("unknown param type .texref"),
            "{err}"
        );
    }

    #[test]
    fn file_with_timestamp_and_size() {
        let src = wrap_body("ret;") + ".file 1 \"kernel.cu\", 1339013327, 64118\n";
        let m = parse(&src).expect("PTX ISA §11.5.3 example parses");
        assert_eq!(m.file_path(1), Some("kernel.cu"));
    }

    #[test]
    fn instruction_modifiers_and_operands() {
        let m = parse_body("fma.rn.f32 \t%f22, %f14, %f15, %f34;");
        let i = instrs(&m)[0];
        assert_eq!(i.mnemonic, Opcode::Fma);
        let mods: Vec<_> = m
            .modifiers(i)
            .iter()
            .map(|&s| m.interner.resolve(s))
            .collect();
        assert_eq!(mods, ["rn", "f32"]);
        assert_eq!(m.operand_ids(i.operands).len(), 4);
    }

    #[test]
    fn predicated_branch_and_label() {
        let m = parse_body("$L__BB0_4:\n@!%p6 bra $L__BB0_4;");
        let k = &m.kernels[0];
        assert!(matches!(k.stmts[0], Stmt::Label(_)));
        let i = instrs(&m)[0];
        let p = i.predicate.expect("predicate parsed");
        assert!(p.negated);
        assert_eq!(m.interner.resolve(p.reg), "%p6");
        let ops = m.operand_ids(i.operands);
        assert!(matches!(m.operand(ops[0]), Operand::SymbolRef(s)
            if m.interner.resolve(*s) == "$L__BB0_4"));
    }

    #[test]
    fn address_expressions() {
        let m = parse_body(
            "mov.u64 %rd1, buf+8;\nmapa.shared::cluster.u32 %r1, buf+-4, %r2;\n\
             ld.local.u32 %r3, [16];\nret;",
        );
        assert_eq!(unparsed_count(&m), 0);
        let is = instrs(&m);
        let offset = |i: &Instr| match m.operand(m.operand_ids(i.operands)[1]) {
            Operand::Offset { offset, .. } => *offset,
            other => panic!("{other:?}"),
        };
        assert_eq!((offset(is[0]), offset(is[1])), (8, -4));
        let Operand::Memory { base, .. } = m.operand(m.operand_ids(is[2].operands)[1]) else {
            panic!("not an address");
        };
        assert!(matches!(m.operand(*base), Operand::Immediate(_)));
    }

    #[test]
    fn bracketed_coordinates_are_operands() {
        let m = parse_body(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes \
             [%r1], [%rd1, {%r2, %r3}], [%r4];\n\
             tex.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [t, s, {%f5, %f6}];\nret;",
        );
        let is = instrs(&m);
        let rest = |i: &Instr, k: usize| match m.operand(m.operand_ids(i.operands)[k]) {
            Operand::Memory { rest, .. } => format!(
                "{:?}",
                m.operand_ids(*rest)
                    .iter()
                    .map(|&id| m.operand(id))
                    .collect::<Vec<_>>()
            ),
            other => panic!("{other:?}"),
        };
        assert!(
            rest(is[0], 1).starts_with("[VectorList"),
            "{}",
            rest(is[0], 1)
        );
        assert_eq!(rest(is[0], 0), "[]");
        assert!(
            rest(is[1], 1).starts_with("[SymbolRef"),
            "{}",
            rest(is[1], 1)
        );
        assert_eq!(unparsed_count(&m), 0);
    }

    #[test]
    fn memory_operands() {
        let m = parse_body(
            "ld.global.u16 %rs1, [%rd25+8];\nst.global.u16 [%rd24], %rs12;\n\
             ld.param.u64 %rd11, [k_param_4];\nld.global.f32 %f1, [%rd1+-4];",
        );
        let is = instrs(&m);
        let mem = |i: &Instr| {
            let ops = m.operand_ids(i.operands);
            ops.iter()
                .find_map(|&id| match m.operand(id) {
                    Operand::Memory { base, offset, .. } => {
                        Some((m.operand(*base).clone(), *offset))
                    }
                    _ => None,
                })
                .expect("memory operand")
        };
        assert_eq!(mem(is[0]).1, 8);
        assert_eq!(mem(is[1]).1, 0);
        assert!(matches!(mem(is[2]).0, Operand::SymbolRef(_)));
        assert_eq!(mem(is[3]).1, -4);
    }

    #[test]
    fn negated_predicate_sources() {
        let m = parse_body(
            "setp.lt.and.s32 %p1, %r1, 10, !%p2;\nbar.red.and.pred %p3, 0, !%p4;\n\
             vote.sync.ballot.b32 %r2, !%p5, -1;\nret;",
        );
        assert_eq!(unparsed_count(&m), 0);
        let negated: Vec<&str> = instrs(&m)[..3]
            .iter()
            .flat_map(|i| m.operand_ids(i.operands).iter())
            .filter_map(|&id| match m.operand(id) {
                Operand::Negated { inner } => match m.operand(*inner) {
                    Operand::Register(r) => Some(m.interner.resolve(*r)),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(negated, ["%p2", "%p4", "%p5"]);
    }

    #[test]
    fn multiple_destination_registers() {
        let m = parse_body(
            "setp.lt.s32 %p1|%p2, %r1, %r2;\nelect.sync %r3|%p3, %r4;\nlop3.or.b32 _|%p4, %r1, %r2, %r3, 0x3f, %p1;",
        );
        assert_eq!(unparsed_count(&m), 0);
        let is = instrs(&m);
        assert_eq!(is.len(), 3);
        for i in &is {
            let dst = m.operand_ids(i.operands)[0];
            let Operand::MultipleDestinations { children } = m.operand(dst) else {
                panic!("first operand is not a p|q pair: {:?}", m.operand(dst));
            };
            assert_eq!(m.operand_ids(*children).len(), 2);
        }
        assert_eq!(m.operand_ids(is[2].operands).len(), 6);
    }

    #[test]
    fn vector_list_operand() {
        let m = parse_body("ld.global.v2.f32 {%f1, %f2}, [%rd1];");
        let i = instrs(&m)[0];
        let ops = m.operand_ids(i.operands);
        match m.operand(ops[0]) {
            Operand::VectorList { children } => assert_eq!(children.len(), 2),
            other => panic!("expected vector list, got {other:?}"),
        }
    }

    #[test]
    fn inlined_at_chains_resolve_to_the_user_line() {
        // Triton: a standard.py reduction inlined at gluon_ce.py:50, whose
        // helper is inlined at the reduction's own line.
        let m = parse_body(
            ".loc 1 50 28\n\
             .loc 2 293 12, function_name $L__info_string0, inlined_at 1 50 28\n\
             .loc 2 263 12, function_name $L__info_string0, inlined_at 2 293 12\n\
             add.f32 %f1, %f1, %f1;\n\
             .loc 2 293 12, function_name $L__info_string0, inlined_at 1 64 10\n\
             .loc 2 263 12, function_name $L__info_string0, inlined_at 2 293 12\n\
             add.f32 %f1, %f1, %f1;",
        );
        let lines: Vec<_> = instrs(&m)
            .iter()
            .map(|i| i.loc.map(|l| (l.file, l.line)))
            .collect();
        assert_eq!(lines, [Some((1, 50)), Some((1, 64))]);
    }

    #[test]
    fn loc_tracking_plain_and_inlined() {
        let m = parse_body(
            ".loc 1 15 13\nadd.f32 %f1, %f1, %f1;\n\
             .loc 2 778 1, function_name $L__info_string0, inlined_at 1 15 13\n\
             cvt.f32.f16 %f2, %rs1;\n.loc 1 0 5\nmov.u32 %r1, 0;",
        );
        let is = instrs(&m);
        assert_eq!(
            is[0].loc,
            Some(SourceLoc {
                file: 1,
                line: 15,
                col: 13
            })
        );
        // Extended form: attribution uses the inlined_at location.
        assert_eq!(
            is[1].loc,
            Some(SourceLoc {
                file: 1,
                line: 15,
                col: 13
            })
        );
        // Line 0 is representable; attribution skips it later.
        assert_eq!(is[2].loc.map(|l| l.line), Some(0));
    }

    #[test]
    fn reg_and_shared_decls() {
        let m = parse_body(
            ".reg .pred %p<9>;\n.reg .f32 %f<35>;\n\
             .shared .align 2 .b8 As[1024];\n\
             .extern .shared .align 4 .b8 dyn[];\nret;",
        );
        let k = &m.kernels[0];
        assert_eq!(k.reg_decls.len(), 2);
        assert_eq!(k.reg_decls[0].count, Some(9));
        assert_eq!(m.interner.resolve(k.reg_decls[1].prefix), "%f");
        assert_eq!(k.shared_decls.len(), 2);
        assert_eq!(k.shared_decls[0].size, Some(1024));
        assert_eq!(k.shared_decls[1].size, None); // dynamic
    }

    #[test]
    fn reg_lists_in_asm_scopes_declare_every_name() {
        let m = parse_body(
            "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %r1, %r2; \
             mov.b32 %r3, {lo, hi}; }\nret;",
        );
        let k = &m.kernels[0];
        let names: Vec<_> = k
            .reg_decls
            .iter()
            .map(|d| m.interner.resolve(d.prefix))
            .collect();
        assert_eq!(names, ["lo$scope1", "hi$scope1"]);
        assert_eq!(k.reg_decls[1].class, FundamentalType::B16);
        assert!(!k.stmts.iter().any(|s| matches!(s, Stmt::Unparsed { .. })));
        assert_eq!(instrs(&m).len(), 3);
    }

    #[test]
    fn registers_declared_without_percent_are_registers() {
        let m = parse_body(
            "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %f1, %f2; \
             mov.b32 %r3, {lo, hi}; }\n{ .reg .pred done; @!done bra $L; }\n$L:\nret;",
        );
        let is = instrs(&m);
        let name = |id: OperandId| match m.operand(id) {
            Operand::Register(r) => m.interner.resolve(*r),
            other => panic!("{other:?}"),
        };
        assert_eq!(name(m.operand_ids(is[0].operands)[0]), "lo$scope1");
        let Operand::VectorList { children } = m.operand(m.operand_ids(is[1].operands)[1]) else {
            panic!("{{lo, hi}} is not a vector");
        };
        let packed: Vec<_> = m
            .operand_ids(*children)
            .iter()
            .map(|&id| name(id))
            .collect();
        assert_eq!(packed, ["lo$scope1", "hi$scope1"]);
        let guard = is[2].predicate.expect("guarded");
        assert_eq!(m.interner.resolve(guard.reg), "done$scope2");
    }

    #[test]
    fn a_scoped_variable_of_any_space_hides_a_register() {
        let m = parse_body(
            ".reg .b64 a;\n.reg .b64 b;\n.reg .b64 c;\n\
             { .shared .align 4 .b8 a[16]; .local .b32 b; .param .align 8 .b8 c[8]; \
             mov.u64 %rd1, a; mov.u64 %rd2, b; mov.u64 %rd3, c; }\nmov.u64 %rd4, a;\nret;",
        );
        let sources: Vec<String> = instrs(&m)[..4]
            .iter()
            .map(|i| match m.operand(m.operand_ids(i.operands)[1]) {
                Operand::SymbolRef(s) => format!("symbol {}", m.interner.resolve(*s)),
                Operand::Register(r) => format!("register {}", m.interner.resolve(*r)),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            sources,
            [
                "symbol a$scope1",
                "symbol b$scope1",
                "symbol c$scope1",
                "register a"
            ]
        );
        assert_eq!(
            m.interner.resolve(m.kernels[0].shared_decls[0].name),
            "a$scope1"
        );
    }

    #[test]
    fn a_video_selector_is_not_part_of_the_name() {
        let m = parse_body("vadd.u32.u32.u32 %r1.h0, %r2.b3, %r3, %r1;\nret;");
        let ops = m.operand_ids(instrs(&m)[0].operands);
        let selected = |id: OperandId| match m.operand(id) {
            Operand::Selected { base, selector } => match m.operand(*base) {
                Operand::Register(r) => {
                    format!(
                        "{}.{}",
                        m.interner.resolve(*r),
                        m.interner.resolve(*selector)
                    )
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        assert_eq!(selected(ops[0]), "%r1.h0");
        assert_eq!(selected(ops[1]), "%r2.b3");
        assert!(matches!(m.operand(ops[2]), Operand::Register(_)));
    }

    #[test]
    fn renamed_names_are_new_names() {
        let names = |body: &str| -> Vec<String> {
            let m = parse_body(body);
            let ops = m.operand_ids(instrs(&m)[0].operands).to_vec();
            ops.iter()
                .map(|&id| match m.operand(id) {
                    Operand::Register(r) => m.interner.resolve(*r).to_owned(),
                    other => panic!("{other:?}"),
                })
                .collect()
        };
        assert_eq!(
            names("{ .reg .b32 %r<2>; .reg .b32 %r0<2>; mov.u32 %r1, %r01; }\nret;"),
            ["%r$scope1$1", "%r0$scope1$1"]
        );
        assert_eq!(
            names(".reg .b32 %r$scope1;\n{ .reg .b32 %r; mov.u32 %r, %r$scope1; }\nret;"),
            ["%r$scope_1", "%r$scope1"]
        );
    }

    #[test]
    fn a_scoped_register_shadows_only_in_its_block() {
        let m = parse_body(
            ".reg .b32 %r<4>;\n.reg .pred p;\n\
             { .reg .b32 %r<2>; .reg .pred p; mov.u32 %r1, %r3; @p bra L; L: }\n\
             { .reg .pred p; @p mov.u32 %r1, 0; }\n@p mov.u32 %r1, 1;\nret;",
        );
        let names: Vec<String> = instrs(&m)
            .iter()
            .flat_map(|i| {
                let pred = i.predicate.map(|p| m.interner.resolve(p.reg).to_owned());
                let ops = m.operand_ids(i.operands).iter();
                pred.into_iter()
                    .chain(ops.filter_map(|&id| match m.operand(id) {
                        Operand::Register(r) => Some(m.interner.resolve(*r).to_owned()),
                        _ => None,
                    }))
            })
            .collect();
        assert_eq!(
            names,
            [
                "%r$scope1$1",
                "%r3",
                "p$scope1",
                "p$scope2",
                "%r1",
                "p",
                "%r1"
            ]
        );
    }

    #[test]
    fn module_scope_shared_decl() {
        // nvcc emits `extern __shared__` as a module-scope `.extern .shared`
        // declaration ahead of the `.entry`. The module must still parse.
        let m = parse(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .extern .shared .align 16 .b8 smem[];\n\
             .extern .shared .align 16 .b8 s[];\n\
             .visible .entry k()\n{\nbar.sync 0;\nret;\n}\n",
        )
        .expect("module with top-level .extern .shared parses");
        assert_eq!(m.kernels.len(), 1);
        assert_eq!(m.shared_decls.len(), 2);
        assert_eq!(m.interner.resolve(m.shared_decls[0].name), "smem");
        assert_eq!(m.shared_decls[0].size, None);
    }

    #[test]
    fn brace_blocks_are_flattened() {
        let m = parse_body("{ cvt.f32.f16 %f14, %rs1;}\nret;");
        assert_eq!(instrs(&m).len(), 2);
        assert_eq!(unparsed_count(&m), 0);
    }

    #[test]
    fn labels_in_asm_scopes_are_scope_local() {
        // Triton's mbarrier wait: the same label in sibling scopes, each
        // branch bound to its own; a kernel-level label is untouched.
        let m = parse_body(
            "{ spin: add.s32 %r1, %r1, 1; @%p1 bra spin; }\n\
             { spin: add.s32 %r2, %r2, 1; @%p2 bra spin; }\n\
             @%p3 bra $L__END;\n$L__END:\nret;",
        );
        let k = &m.kernels[0];
        let labels: Vec<&str> = k
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::Label(l) => Some(m.interner.resolve(*l)),
                _ => None,
            })
            .collect();
        assert_eq!(labels, ["spin$scope1", "spin$scope2", "$L__END"]);
        let targets: Vec<&str> = instrs(&m)
            .iter()
            .filter(|i| i.mnemonic == Opcode::Bra)
            .map(|i| match m.operand(m.operand_ids(i.operands)[0]) {
                Operand::SymbolRef(s) => m.interner.resolve(*s),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(targets, ["spin$scope1", "spin$scope2", "$L__END"]);
    }

    #[test]
    fn pragma_and_branchtargets() {
        let m = parse_body(".pragma \"nounroll\";\n$L_tbl: .branchtargets $L_a, $L_b;\nret;");
        let k = &m.kernels[0];
        assert!(
            k.stmts
                .iter()
                .any(|s| matches!(s, Stmt::BranchTargets(t) if t.len() == 2))
        );
        assert_eq!(unparsed_count(&m), 0);
    }

    #[test]
    fn bad_statement_recovers_without_poisoning() {
        let m = parse_body("fma.rn.f32 %f1, ;\nadd.f32 %f1, %f1, %f1;");
        assert_eq!(unparsed_count(&m), 1);
        assert_eq!(instrs(&m).len(), 1); // the add survived
    }

    #[test]
    fn an_unparsed_statement_keeps_the_registers_it_names() {
        let m = parse_body(
            ".reg .b32 %r<3>;\n.local .u32 a[4];\n\
             { .reg .b32 t; ld.local.u32 t, a[1]; }\n\
             ld.local.u32 %r1, a[%r2];\nfuture.op %tid.x;\nret;",
        );
        let registers: Vec<Vec<&str>> = (m.kernels[0].stmts.iter())
            .filter_map(|s| match s {
                Stmt::Unparsed { registers, .. } => {
                    Some(registers.iter().map(|&r| m.interner.resolve(r)).collect())
                }
                _ => None,
            })
            .collect();
        let none: Vec<&str> = Vec::new();
        assert_eq!(registers, [vec!["t$scope1"], vec!["%r1", "%r2"], none]);
    }

    #[test]
    fn unknown_top_level_directive_is_loud() {
        let err = parse(".version 8.7\n.target sm_80\n.frobnicate 3\n").unwrap_err();
        assert!(err.to_string().contains("frobnicate"), "{err}");
        assert_eq!(err.line, 3);
    }

    #[test]
    fn unterminated_body_is_loud() {
        let err = parse(".version 8.7\n.target sm_80\n.entry k() {\nret;\n").unwrap_err();
        assert!(err.to_string().contains("end of file"), "{err}");
    }

    #[test]
    fn params_with_types() {
        let src = ".version 8.7\n.target sm_80\n.visible .entry k(\n\
                   .param .u32 k_param_0,\n.param .f32 k_param_1,\n\
                   .param .u64 k_param_2,\n.param .align 64 .b8 k_param_3[128]\n)\n\
                   {\nret;\n}\n";
        let m = parse(src).expect("parses");
        let k = &m.kernels[0];
        let tys: Vec<_> = k.params.iter().map(|p| p.ty.name()).collect();
        assert_eq!(tys, ["u32", "f32", "u64", "b8"]);
        assert_eq!(m.interner.resolve(k.params[2].name), "k_param_2");
        assert_eq!((k.params[3].align, k.params[3].size), (Some(64), Some(128)));
        assert_eq!((k.params[0].align, k.params[0].size), (None, None));
    }

    #[test]
    fn maxntid_directive() {
        let src = ".version 8.7\n.target sm_80\n.visible .entry k()\n\
                   .maxntid 16, 16, 1\n.maxnreg 168\n{\nret;\n}\n";
        let m = parse(src).expect("parses");
        assert_eq!(m.kernels[0].maxntid, Some([16, 16, 1]));
        let tuning = &m.kernels[0].tuning;
        assert_eq!(tuning.len(), 1);
        assert_eq!(
            (m.interner.resolve(tuning[0].0), tuning[0].1),
            ("maxnreg", 168)
        );
    }

    #[test]
    fn debug_section_is_skipped() {
        let src = ".version 8.7\n.target sm_80\n.visible .entry k()\n{\nret;\n}\n\
                   .file 1 \"a.cu\"\n.section .debug_str\n{\n\
                   $L__info_string0:\n.b8 95,90,78\n}\n";
        let m = parse(src).expect("parses");
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.kernels.len(), 1);
    }
}
