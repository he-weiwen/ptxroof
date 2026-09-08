//! Recursive-descent parser: PTX text → flat module IR.
//!
//! Transcribed from v1's `lib/PTX/Parser.cpp` (v1 lives only in git
//! history now, last at 690d81d; statement/operand grammar,
//! per-statement error recovery, tolerant memory-operand skip)
//! and extended from inline-asm bodies to full programs: module
//! directives, kernel signatures with param tables, `.reg`/`.shared`
//! declarations, labels, extended `.loc` (plain and
//! `function_name/inlined_at` forms — both appear in k2), `.pragma` in
//! body position, `.section .debug_*` data sections (parsed and
//! skipped), `{ ... }` statement blocks (inline-asm expansions —
//! flattened; the braces scope nothing we model), and `.branchtargets`.
//!
//! Error policy: the library never panics on malformed
//! input. Inside a kernel body a bad statement becomes `Stmt::Unparsed`
//! and parsing resumes after the next `;` — one malformed instruction
//! never poisons the kernel. Outside bodies the structure is rigid and
//! small, so a malformed module-level construct is a loud `ParseError`
//! naming the line.

use crate::core::{
    FileDirective, IdxRange, IndexVec, Instr, Interner, Kernel, Module, Operand, OperandId, Param,
    Predicate, RegDecl, SharedDecl, SourceLoc, Stmt, Symbol,
};
use crate::parse::lexer::{Token, TokenKind, tokenize};

#[derive(Debug, thiserror::Error)]
#[error("parse error at line {line}: {message}")]
pub struct ParseError {
    pub line: u32,
    pub message: String,
}

pub fn parse(source: &str) -> Result<Module, ParseError> {
    Parser::new(source).run()
}

struct Parser<'a> {
    source: &'a str,
    toks: Vec<Token<'a>>,
    pos: usize,
    interner: Interner,
    version: (u32, u32),
    target: Option<Symbol>,
    address_size: u32,
    files: Vec<FileDirective>,
    kernels: Vec<Kernel>,
    operands: IndexVec<OperandId, Operand>,
    operand_lists: Vec<OperandId>,
    modifier_pool: Vec<Symbol>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Parser {
            source,
            toks: tokenize(source),
            pos: 0,
            interner: Interner::default(),
            version: (0, 0),
            target: None,
            address_size: 64,
            files: Vec::new(),
            kernels: Vec::new(),
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
                    self.address_size = self.expect_u32(".address_size")?;
                }
                "file" => {
                    let index = self.expect_u32(".file index")?;
                    let t = self.expect(TokenKind::String, ".file path")?;
                    let path = self.interner.intern(t.text.trim_matches('"'));
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
                "shared" => {
                    // Module-scope shared variable. nvcc emits dynamic shared
                    // memory as `.extern .shared .align A .b8 name[];` at module
                    // scope (the `.extern` linkage prefix is consumed above).
                    // These are declarations, not analysis content, and a
                    // module-scope dynamic decl is 0 static bytes, so parse-and-
                    // discard keeps the module parseable without affecting the
                    // per-kernel static shared-per-CTA report.
                    if self.parse_shared_decl().is_none() {
                        self.resync_to_semicolon();
                    }
                }
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
                    self.expect_u32("directive argument")?;
                }
                _ => {
                    self.pos = save;
                    break;
                }
            }
        }

        self.expect(TokenKind::LBrace, "'{' starting kernel body")?;
        self.parse_body(&mut kernel)?;
        Ok(kernel)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        self.expect(TokenKind::Dot, "'.param'")?;
        let kw = self.expect_ident("'.param'")?;
        if self.interner.resolve(kw) != "param" {
            return Err(self.err("expected .param in signature"));
        }
        let ty;
        loop {
            self.expect(TokenKind::Dot, "param type")?;
            let attr = self.expect_ident("param attribute")?;
            match self.interner.resolve(attr) {
                "align" => {
                    self.expect_u32(".align value")?;
                }
                _ => {
                    ty = attr;
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
        if self.eat(TokenKind::LBracket) {
            if self.at(TokenKind::Number) {
                self.bump();
            }
            self.expect(TokenKind::RBracket, "']' after param array size")?;
        }
        Ok(Param { ty, name })
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
                }
                TokenKind::LBrace => {
                    self.bump();
                    block_depth += 1;
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

    fn parse_body_directive(
        &mut self,
        kernel: &mut Kernel,
        cur_loc: &mut Option<SourceLoc>,
    ) -> Result<(), ParseError> {
        let offset = self.peek().offset;
        self.bump(); // consume '.'
        let Ok(name) = self.expect_ident("body directive name") else {
            kernel.stmts.push(Stmt::Unparsed { offset });
            self.resync_to_semicolon();
            return Ok(());
        };
        match self.interner.resolve(name) {
            "loc" => self.parse_loc(cur_loc),
            "reg" => {
                if let Some(decls) = self.parse_reg_decl() {
                    kernel.reg_decls.extend(decls);
                } else {
                    kernel.stmts.push(Stmt::Unparsed { offset });
                    self.resync_to_semicolon();
                }
            }
            "shared" => {
                if let Some(decl) = self.parse_shared_decl() {
                    kernel.shared_decls.push(decl);
                } else {
                    kernel.stmts.push(Stmt::Unparsed { offset });
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
                    if let Some(decl) = self.parse_shared_decl() {
                        kernel.shared_decls.push(decl);
                        return Ok(());
                    }
                }
                kernel.stmts.push(Stmt::Unparsed { offset });
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
                    targets.push(self.interner.intern(t.text));
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.resync_to_semicolon();
                kernel.stmts.push(Stmt::BranchTargets(targets));
            }
            "local" | "const" | "global" | "align" => {
                // In-body variable declarations we don't model; the
                // instructions touching them are still counted.
                self.resync_to_semicolon();
            }
            _ => {
                kernel.stmts.push(Stmt::Unparsed { offset });
                self.resync_to_semicolon();
            }
        }
        Ok(())
    }

    /// `.loc f l c` or
    /// `.loc f l c, function_name $sym, inlined_at f l c`.
    /// The effective location is the `inlined_at` one when present:
    /// attribution wants the user's source line, not the header line
    /// the code was inlined from.
    fn parse_loc(&mut self, cur_loc: &mut Option<SourceLoc>) {
        let Some(mut loc) = self.parse_loc_triple() else {
            return;
        };
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
                        loc = outer;
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
        let class = if self.at(TokenKind::Identifier) {
            let t = self.bump();
            self.interner.intern(t.text)
        } else {
            return None;
        };
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
                ty = Some(self.interner.intern(attr));
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
        let unparsed = |p: &mut Self| {
            p.resync_to_semicolon();
            Stmt::Unparsed { offset }
        };

        // Optional predicate: @p / @!p; the name may be a register
        // (`%p6`) or a scoped-declaration identifier (`p`).
        let mut predicate = None;
        if self.eat(TokenKind::At) {
            let negated = self.eat(TokenKind::Bang);
            let reg = match self.peek().kind {
                TokenKind::Register | TokenKind::Identifier => {
                    let t = self.bump();
                    self.interner.intern(t.text)
                }
                _ => return unparsed(self),
            };
            predicate = Some(Predicate { reg, negated });
        }

        if !self.at(TokenKind::Identifier) {
            return unparsed(self);
        }
        let mnemonic_tok = self.bump();
        let mnemonic = self.interner.intern(mnemonic_tok.text);

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
                let sym = self.interner.intern(t.text);
                Some(self.push_operand(Operand::Register(sym)))
            }
            TokenKind::Number => {
                let t = self.bump();
                let sym = self.interner.intern(t.text);
                Some(self.push_operand(Operand::Immediate(sym)))
            }
            TokenKind::Identifier => {
                let t = self.bump();
                let sym = self.interner.intern(t.text);
                Some(self.push_operand(Operand::SymbolRef(sym)))
            }
            TokenKind::LBracket => {
                self.bump();
                let base = match self.peek().kind {
                    TokenKind::Register => {
                        let t = self.bump();
                        Operand::Register(self.interner.intern(t.text))
                    }
                    TokenKind::Identifier => {
                        let t = self.bump();
                        Operand::SymbolRef(self.interner.intern(t.text))
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
                // Tolerant skip to the matching ']' (v1 rule): TMA tensor
                // operands carry `[base, {dim_list}]`, which we don't
                // model at the memory-operand level.
                let mut depth = 0u32;
                loop {
                    match self.peek().kind {
                        TokenKind::RBracket if depth == 0 => break,
                        TokenKind::LBracket | TokenKind::LBrace => depth += 1,
                        TokenKind::RBracket | TokenKind::RBrace => depth = depth.checked_sub(1)?,
                        TokenKind::EndOfFile => return None,
                        _ => {}
                    }
                    self.bump();
                }
                self.bump(); // consume ']'
                Some(self.push_operand(Operand::Memory { base, offset }))
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

/// Parse a PTX integer literal: decimal, hex (`0x..`), optionally
/// negative.
fn parse_int(text: &str) -> Option<i64> {
    let (neg, rest) = match text.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, text),
    };
    let val = if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()?
    } else {
        rest.parse::<i64>().ok()?
    };
    Some(if neg { -val } else { val })
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
        assert_eq!(m.address_size, 64);
    }

    #[test]
    fn instruction_modifiers_and_operands() {
        let m = parse_body("fma.rn.f32 \t%f22, %f14, %f15, %f34;");
        let i = instrs(&m)[0];
        assert_eq!(m.interner.resolve(i.mnemonic), "fma");
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
                    Operand::Memory { base, offset } => Some((m.operand(*base).clone(), *offset)),
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
        assert_eq!(names, ["lo", "hi"]);
        assert_eq!(m.interner.resolve(k.reg_decls[1].class), "b16");
        assert!(!k.stmts.iter().any(|s| matches!(s, Stmt::Unparsed { .. })));
        assert_eq!(instrs(&m).len(), 3);
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
    }

    #[test]
    fn brace_blocks_are_flattened() {
        let m = parse_body("{ cvt.f32.f16 %f14, %rs1;}\nret;");
        assert_eq!(instrs(&m).len(), 2);
        assert_eq!(unparsed_count(&m), 0);
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
                   .param .u64 k_param_2\n)\n{\nret;\n}\n";
        let m = parse(src).expect("parses");
        let k = &m.kernels[0];
        let tys: Vec<_> = k.params.iter().map(|p| m.interner.resolve(p.ty)).collect();
        assert_eq!(tys, ["u32", "f32", "u64"]);
        assert_eq!(m.interner.resolve(k.params[2].name), "k_param_2");
    }

    #[test]
    fn maxntid_directive() {
        let src = ".version 8.7\n.target sm_80\n.visible .entry k()\n\
                   .maxntid 16, 16, 1\n{\nret;\n}\n";
        let m = parse(src).expect("parses");
        assert_eq!(m.kernels[0].maxntid, Some([16, 16, 1]));
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
