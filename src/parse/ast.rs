//! Canonical dumper: the human/debug view of the flat IR. Resolves
//! every `Symbol`/`OperandId` so no raw index ever reaches output;
//! snapshots and expected outputs route through here.
//!
//! The dump is itself valid PTX, which makes idempotence testable:
//! `dump(parse(dump(parse(src))))` must equal `dump(parse(src))` on the
//! whole corpus. Normalizations applied (all loss-free for analysis):
//! comments, `.section` data payloads, `.pragma`, and inline-asm scope
//! braces are dropped; extended `.loc` collapses to its effective
//! (inlined-at) location; whitespace is canonical.

use crate::core::{Kernel, Module, Operand, OperandId, SourceLoc, Stmt};
use std::fmt::Write;

pub fn dump(module: &Module) -> String {
    let mut out = String::new();
    let (major, minor) = module.version;
    let _ = writeln!(out, ".version {major}.{minor}");
    let _ = writeln!(out, ".target {}", module.interner.resolve(module.target));
    let _ = writeln!(out, ".address_size {}", module.address_size);
    for kernel in &module.kernels {
        dump_kernel(module, kernel, &mut out);
    }
    for file in &module.files {
        let _ = writeln!(
            out,
            ".file {} \"{}\"",
            file.index,
            module.interner.resolve(file.path)
        );
    }
    out
}

fn dump_kernel(m: &Module, k: &Kernel, out: &mut String) {
    let _ = writeln!(out, ".visible .entry {}(", m.interner.resolve(k.name));
    for (i, p) in k.params.iter().enumerate() {
        let comma = if i + 1 < k.params.len() { "," } else { "" };
        let _ = writeln!(
            out,
            ".param .{} {}{comma}",
            m.interner.resolve(p.ty),
            m.interner.resolve(p.name)
        );
    }
    let _ = writeln!(out, ")");
    if let Some([x, y, z]) = k.maxntid {
        let _ = writeln!(out, ".maxntid {x}, {y}, {z}");
    }
    if let Some([x, y, z]) = k.reqntid {
        let _ = writeln!(out, ".reqntid {x}, {y}, {z}");
    }
    let _ = writeln!(out, "{{");

    let mut seen = std::collections::HashSet::new();
    for decl in &k.reg_decls {
        if !seen.insert((decl.class, decl.prefix, decl.count)) {
            continue;
        }
        let prefix = m.interner.resolve(decl.prefix);
        let class = m.interner.resolve(decl.class);
        match decl.count {
            Some(n) => {
                let _ = writeln!(out, ".reg .{class} {prefix}<{n}>;");
            }
            None => {
                let _ = writeln!(out, ".reg .{class} {prefix};");
            }
        }
    }
    for decl in &k.shared_decls {
        let align = decl
            .align
            .map(|a| format!(".align {a} "))
            .unwrap_or_default();
        let name = m.interner.resolve(decl.name);
        let ty = m.interner.resolve(decl.ty);
        match decl.size {
            Some(n) => {
                let _ = writeln!(out, ".shared {align}.{ty} {name}[{n}];");
            }
            None => {
                let _ = writeln!(out, ".extern .shared {align}.{ty} {name}[];");
            }
        }
    }

    let mut last_loc: Option<SourceLoc> = None;
    for stmt in &k.stmts {
        match stmt {
            Stmt::Label(sym) => {
                let _ = writeln!(out, "{}:", m.interner.resolve(*sym));
            }
            Stmt::BranchTargets(targets) => {
                let names: Vec<_> = targets.iter().map(|&t| m.interner.resolve(t)).collect();
                let _ = writeln!(out, ".branchtargets {};", names.join(", "));
            }
            Stmt::Unparsed { offset } => {
                // Comments are dropped on reparse; the corpus check keeps
                // committed fixtures free of these.
                let _ = writeln!(out, "// unparsed statement at byte {offset}");
            }
            Stmt::Instr(instr) => {
                if instr.loc != last_loc
                    && let Some(loc) = instr.loc
                {
                    let _ = writeln!(out, ".loc {} {} {}", loc.file, loc.line, loc.col);
                }
                last_loc = instr.loc;
                let pred = match instr.predicate {
                    Some(p) => format!(
                        "@{}{} ",
                        if p.negated { "!" } else { "" },
                        m.interner.resolve(p.reg)
                    ),
                    None => String::new(),
                };
                let ops: Vec<String> = m
                    .operand_ids(instr.operands)
                    .iter()
                    .map(|&id| dump_operand(m, id))
                    .collect();
                let sep = if ops.is_empty() { "" } else { " " };
                let _ = writeln!(out, "{pred}{}{sep}{};", m.opcode(instr), ops.join(", "));
            }
        }
    }
    let _ = writeln!(out, "}}");
}

fn dump_operand(m: &Module, id: OperandId) -> String {
    match m.operand(id) {
        Operand::Register(sym) | Operand::Immediate(sym) | Operand::SymbolRef(sym) => {
            m.interner.resolve(*sym).to_owned()
        }
        Operand::Memory { base, offset } => {
            let base = dump_operand(m, *base);
            if *offset == 0 {
                format!("[{base}]")
            } else {
                format!("[{base}+{offset}]")
            }
        }
        Operand::VectorList { children } => {
            let inner: Vec<String> = m
                .operand_ids(*children)
                .iter()
                .map(|&id| dump_operand(m, id))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Operand::MultipleDestinations { children } => {
            let inner: Vec<String> = m
                .operand_ids(*children)
                .iter()
                .map(|&id| dump_operand(m, id))
                .collect();
            inner.join("|")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    #[test]
    fn dump_is_reparseable_and_idempotent() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   .reg .pred %p<2>;\n.loc 1 5 9\n\
                   ld.global.v2.f32 {%f1, %f2}, [%rd1+-8];\n\
                   setp.lt.s32 %p1|%p2, %r1, %r2;\n\
                   @!%p1 bra $L__X;\n$L__X:\nret;\n}\n.file 1 \"a.cu\"\n";
        let d1 = dump(&parse(src).expect("src parses"));
        let d2 = dump(&parse(&d1).expect("dump reparses"));
        assert_eq!(d1, d2);
    }
}
