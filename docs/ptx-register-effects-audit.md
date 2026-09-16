# PTX register effects and SSA readiness audit

Audited 2026-09-16 against source revision
`10e15b27567612c0c9e72f5b3e8143c589460e5c`.
No functional code was changed for this audit.

## Conclusion

A pure function that decodes **register effects** is practical. A record containing
only a guard, a list of reads, and a list of writes is **not a complete account of
PTX instruction effects**. It needs precise operand roles, implicit carry state,
and a way to distinguish supported decoding from unknown effects. Asynchronous
register results and conditionally valid outputs need additional treatment before
analyses can use them as ordinary values.

The current project does not have that decoder. Its instruction classifier answers
a counting question, and its affine tracer uses a separate, incomplete destination
heuristic. Building SSA directly on that heuristic would preserve existing mistakes.
The first priority is lossless operands and resolved register identities, then an
explicit, fallible register-effects decoder. Full memory SSA, an MLIR-like operation
framework, and a simulator are not prerequisites.

## Scope, evidence, and reproducibility

The primary reference is the repository's locally available NVIDIA manual snapshot,
[PTX ISA 9.3](../refs/ptx-isa.html). SHA-256 of the audited HTML:

```text
940cc68f858cefdf82425b47ee3bac3afde447c8a85b95f43d7d6fb1f46b4413
```

The existing [counting coverage audit](ptx-instruction-coverage.md) records its fetch
date as 2026-07-16. That date is inherited provenance, not independently established
by this audit. The hash identifies the actual artifact read. `refs/` is intentionally
untracked; obtain that snapshot to reproduce the exact section content. The
[fetch script](../tools/fetch-manuals.sh) fetches the **latest** page and does not
reproduce a pin.

I independently extracted instruction sections from that HTML, inspected their
syntax and relevant descriptions/semantics, and reconciled them with the parser,
classifier, CFG, and scalar tracer. The ledger links to each exact local section.
The [official online manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
was checked and identifies itself as **9.4** on the audit date. Its same-named
anchors are useful for navigation, but its version and section numbers can differ.
This report is exhaustive for the project's pinned inventory, **not a claim of full
9.4 coverage**. In particular, newer instructions such as `spcompress`,
`spdecompress`, and `applypriority.async.bulk*` need a separate version update and
inventory audit before being treated as supported. Some features in the local
snapshot's 9.3 release notes also appear in the live 9.4 notes; use the artifact
hash, not a version label alone, to settle that discrepancy.

Scope includes every non-excluded instruction section, whether currently classified,
explicitly planned, or deferred in the counting audit. This deliberately includes
`multimem`, warpgroup/tensor-memory operations, deprecated forms, and stack operations
so that roadmap wording cannot hide an effect-model gap. Excluded as requested:
**fabric (§9.7.10), texture (§9.7.11), surface (§9.7.12), and video (§9.7.19)**.
This excludes `istypep` even though the current classifier recognizes it, and
fabric-specific proxy-fence variants. Tensor maps/TMA are included: they are not
texture or surface instructions. `ld.global.nc` is included despite its cache path.
`nop`, an implementation-recognized spelling without an instruction section in the
pinned inventory, is discussed separately below.

This is an **instruction-effects and SSA audit**, not a proof of every numerical
operation, exhaustive target/qualifier validation, or a hardware conformance test.
The ledger accounts for operand-role-changing variants and execution-effect
categories. It does not enumerate the Cartesian product of shapes, types, rounding
modes, architectures, and cache modifiers. Numerical transfer functions and legal
encoding validation still require their own tests. No GPU execution or `ptxas`
validation was used to claim semantic correctness here.

For calls, the additional authoritative reference is NVIDIA's
[PTX Writer's Guide, function calling sequence](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#function-calling-sequence):
PTX registers are virtual, physical allocation happens in translation, and ABI
parameters/returns use parameter space. Its older blanket statement about no
PTX-level stack manipulation must not override the newer ISA's explicit
`stacksave`/`stackrestore`/`alloca` sections. ISA §7 and the particular function
signature determine which non-ABI register-parameter forms are permitted.

## Shared rules for every instruction

### Register identity comes from declarations

An identifier beginning with `%` is not the definition of a register. PTX permits
bare register names. Resolve declaration, lexical scope, kernel/function ownership,
register arrays and vector components before assigning a `RegisterId`. Distinguish
ordinary registers, read-only special registers, symbols/addresses, labels,
immediates, and the sink `_`. Do not give the sink an SSA value.

The current `Operand::Register(Symbol)` is a syntactic category, not this resolved
identity. Scoped declarations are especially important because flattening two
blocks that both declare `%r` must not merge their definitions.

Sources: [identifiers](../refs/ptx-isa.html#identifiers),
[operand type information](../refs/ptx-isa.html#operand-type-information),
[scope construct](../refs/ptx-isa.html#control-flow-instructions-curly-braces), and
[IR](../src/ptx/ir.rs#L38). See the concrete parser findings below.

### Uses, definitions, and execution guards are separate

All explicit input values must refer to the pre-instruction register versions,
including when a destination aliases a source. Flatten register tuples for def-use
purposes but retain their operand/component positions, types, and ordering.
`[address]` writes memory where appropriate; it **reads** the address register.
Descriptor, policy, mask, selector, count, stride, coordinate, and predicate operands
can be register uses even when they do not look like arithmetic inputs.

For a valid guarded instruction, the guard is a use. A false guard preserves old
ordinary destinations and suppresses the instruction's execution effects. An SSA
representation can use guarded operations or a branch plus merges. For a pure,
speculatable arithmetic operation, a select can express the resulting value; this
does not license executing guarded loads, atomics, calls, or traps unconditionally.
Retain instruction-specific predication/uniformity restrictions. A warp-collective
instruction is not legal under an arbitrary per-lane guard merely because the
syntax can spell one.

Distinguish **may-use** information for def-use/liveness from exact value dependence.
For example, conservatively recording both `selp` alternatives is fine; proving
which value it returns is a separate analysis. For `wgmma`, an unknown `scale-d`
may read old accumulators, but a known false value does not. Losing that distinction
can create spurious undefined-input complaints.

Source: [predicated execution](../refs/ptx-isa.html#predicated-execution), plus the
individual collective instruction sections in the ledger.

### Narrow or packed does not automatically mean a partial write

Ordinary arithmetic writes a complete destination of the required width. Packed
lanes are generally bits within that value. For permitted wider destinations,
`ld` and `cvt` use the ISA's sign/zero-extension rules. `bfi` gets untouched bits
from an explicit source; small `cvt.pack` forms get the remaining bits from `c`.
Packing/unpacking `mov` defines all non-sink destination components. These cases
do not justify a universal implicit read of the old destination.

Keep **undefined/unspecified result bits**, **invalid-to-use outputs**, **preserved
old state**, and **a skipped instruction** distinct. For example, unsuccessful
`mbarrier` waits do not entitle an analysis to reuse the old report value; the
manual prohibits inspecting those report outputs on that path. Tensor-memory
output-lane masks and bulk byte masks preserve selected **memory**, not the
register containing its address.

Sources: [operand width rules](../refs/ptx-isa.html#operand-size-exceeding-instruction-type-size),
`mov`, `bfi`, `cvt.pack`, `mbarrier.test_wait`, and `tcgen05.mma` ledger rows.

### Implicit and environmental state

- `CC.CF` is a per-thread implicit carry/borrow bit. Model it explicitly for carry
  chains, including guarded updates. Calls do not preserve it. PTX does not expose
  general CPU-style flags for every arithmetic instruction.
- Stack operations observe/update stack and allocation state. Calls/returns also
  carry control and parameter-state effects; physical-register ABI clobbers are
  not a list of arbitrary PTX virtual-register definitions.
- Special registers are read-only observations, not ordinary mutable registers
  needing local definitions. Stable launch coordinates can be entry values;
  clocks/timers, performance counters, and scheduling-dependent observations must
  not all be folded into one immutable entry value. `activemask` and election
  depend on execution participation.
- Cross-lane operations depend on other participating lanes' values. A per-thread
  def-use graph can keep the collective operation opaque; it cannot interpret
  a shuffle or reduction as arithmetic on only that lane's local SSA values.

Sources: [extended precision](../refs/ptx-isa.html#extended-precision-integer-arithmetic-instructions),
[special registers](../refs/ptx-isa.html#special-registers), and the call/stack/
collective rows below.

### Register SSA is not the complete execution dependency graph

A load can define an opaque SSA value without implementing memory SSA. That does
not make the load referentially transparent. Memory effects need state spaces,
addresses/ranges or unknown aliases, read/write/RMW distinctions, ordering, scope,
proxy, volatility/MMIO, and asynchronous completion if a consumer reasons about
memory or transformations. A single `Memory(Read|Write)` label cannot settle these.

`wgmma` and `tcgen05.ld` can define **pending register results**. `.sync` may only
mean collective rendezvous. Preserve issue, completion, and input-lifetime rules;
ordinary register accesses before the required wait may be undefined. Initial
SSA work can reject those families with a precise unsupported reason instead of
implementing tokens and async scheduling. Similarly, memory-resident `mbarrier`
state is not simply another scalar register, and `tcgen05` collector buffers,
allocation state, and scheduler cancellation require effects outside register SSA.

The analysis function can nevertheless be pure: immutable program/context in,
owned result or structured error out. The represented instruction need not be
pure for its **analysis** to be pure.

## Complete instruction ledger

The ledger covers **186 instruction/construct sections**, independently extracted from the pinned HTML. Combined manual sections retain combined rows with each instruction named; repeated mnemonics in different type families remain separate rows. Two rows are syntax constructs (`{}` and `@`). This is a count of manual sections, not unique mnemonics or legal encodings.

`W` lists explicit register definitions; `R` lists explicit uses and selected implicit dependencies. Only register-valued operands contribute ordinary register uses: immediates, labels, sink `_`, and symbol addresses do not. Every instruction also reads its execution guard when present. The shared rules above are part of every row. `—` means no explicit register output/input, **not** no effect.

Current **counting route**: `A` non-flop arithmetic, `F` flop classification, `M` memory/copy, `S` sync, `C` communication, `B` control, `I` ignored for counting, `U` unknown, `F/U` selected matrix forms only, `syntax` parser construct. `*` flags a particularly misleading route or finding below. These are routes through the current classifier for parseable forms, not semantic-validation or SSA-support claims. Broad arms also accept invalid forms.


### 9.7.1. Integer Arithmetic Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.1.1](../refs/ptx-isa.html#integer-arithmetic-instructions-add) `add` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.2](../refs/ptx-isa.html#integer-arithmetic-instructions-sub) `sub` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.3](../refs/ptx-isa.html#integer-arithmetic-instructions-mul) `mul` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.4](../refs/ptx-isa.html#integer-arithmetic-instructions-mad) `mad` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.1.5](../refs/ptx-isa.html#integer-arithmetic-instructions-clmad) `clmad` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.1.6](../refs/ptx-isa.html#integer-arithmetic-instructions-mul24) `mul24` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.7](../refs/ptx-isa.html#integer-arithmetic-instructions-mad24) `mad24` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.1.8](../refs/ptx-isa.html#integer-arithmetic-instructions-sad) `sad` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.1.9](../refs/ptx-isa.html#integer-arithmetic-instructions-div) `div` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.10](../refs/ptx-isa.html#integer-arithmetic-instructions-rem) `rem` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.11](../refs/ptx-isa.html#integer-arithmetic-instructions-abs) `abs` | d | a | Whole register result; no implicit CC effect. | A |
| [9.7.1.12](../refs/ptx-isa.html#integer-arithmetic-instructions-neg) `neg` | d | a | Whole register result; no implicit CC effect. | A |
| [9.7.1.13](../refs/ptx-isa.html#integer-arithmetic-instructions-min) `min` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.14](../refs/ptx-isa.html#integer-arithmetic-instructions-max) `max` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.15](../refs/ptx-isa.html#integer-arithmetic-instructions-popc) `popc` | d | a | Result is u32 even when the input is 64-bit. | A |
| [9.7.1.16](../refs/ptx-isa.html#integer-arithmetic-instructions-clz) `clz` | d | a | Result is u32 even when the input is 64-bit. | A |
| [9.7.1.17](../refs/ptx-isa.html#integer-arithmetic-instructions-bfind) `bfind` | d | a | Result is u32 even when the input is 64-bit. | A |
| [9.7.1.18](../refs/ptx-isa.html#integer-arithmetic-instructions-fns) `fns` | d | mask, base, offset | Whole register result; no implicit CC effect. | A |
| [9.7.1.19](../refs/ptx-isa.html#integer-arithmetic-instructions-brev) `brev` | d | a | Whole register result; no implicit CC effect. | A |
| [9.7.1.20](../refs/ptx-isa.html#integer-arithmetic-instructions-bfe) `bfe` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.1.21](../refs/ptx-isa.html#integer-arithmetic-instructions-bfi) `bfi` | f | a, b, c, d | Inserted bits come from a and remaining bits from b; no implicit old-f read. | A |
| [9.7.1.22](../refs/ptx-isa.html#integer-arithmetic-instructions-szext) `szext` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.23](../refs/ptx-isa.html#integer-arithmetic-instructions-bmsk) `bmsk` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.1.24](../refs/ptx-isa.html#integer-arithmetic-instructions-dp4a) `dp4a` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.1.25](../refs/ptx-isa.html#integer-arithmetic-instructions-dp2a) `dp2a` | d | a, b, c | Whole register result; no implicit CC effect. | A |

### 9.7.2. Extended-Precision Integer Arithmetic Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.2.1](../refs/ptx-isa.html#extended-precision-arithmetic-instructions-add-cc) `add.cc` | d; CC.CF when .cc | a, b | Carry/borrow is implicit per-thread state. Plain .cc writes it; *c reads it and optionally writes it. | A |
| [9.7.2.2](../refs/ptx-isa.html#extended-precision-arithmetic-instructions-addc) `addc` | d; CC.CF when .cc | a, b; CC.CF | Carry/borrow is implicit per-thread state. Plain .cc writes it; *c reads it and optionally writes it. | A |
| [9.7.2.3](../refs/ptx-isa.html#extended-precision-arithmetic-instructions-sub-cc) `sub.cc` | d; CC.CF when .cc | a, b | Carry/borrow is implicit per-thread state. Plain .cc writes it; *c reads it and optionally writes it. | A |
| [9.7.2.4](../refs/ptx-isa.html#extended-precision-arithmetic-instructions-subc) `subc` | d; CC.CF when .cc | a, b; CC.CF | Carry/borrow is implicit per-thread state. Plain .cc writes it; *c reads it and optionally writes it. | A |
| [9.7.2.5](../refs/ptx-isa.html#extended-precision-arithmetic-instructions-mad-cc) `mad.cc` | d; CC.CF when .cc | a, b, c | Carry/borrow is implicit per-thread state. Plain .cc writes it; *c reads it and optionally writes it. | A |
| [9.7.2.6](../refs/ptx-isa.html#extended-precision-arithmetic-instructions-madc) `madc` | d; CC.CF when .cc | a, b, c; CC.CF | Carry/borrow is implicit per-thread state. Plain .cc writes it; *c reads it and optionally writes it. | A |

### 9.7.3. Floating-Point Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.3.1](../refs/ptx-isa.html#floating-point-instructions-testp) `testp` | p | a | Whole register result; no implicit CC effect. | A |
| [9.7.3.2](../refs/ptx-isa.html#floating-point-instructions-copysign) `copysign` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.3.3](../refs/ptx-isa.html#floating-point-instructions-add) `add` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.3.4](../refs/ptx-isa.html#floating-point-instructions-sub) `sub` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.3.5](../refs/ptx-isa.html#floating-point-instructions-mul) `mul` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.3.6](../refs/ptx-isa.html#floating-point-instructions-fma) `fma` | d | a, b, c | Whole register result; no implicit CC effect. | F |
| [9.7.3.7](../refs/ptx-isa.html#floating-point-instructions-mad) `mad` | d | a, b, c | Whole register result; no implicit CC effect. | F |
| [9.7.3.8](../refs/ptx-isa.html#floating-point-instructions-div) `div` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.3.9](../refs/ptx-isa.html#floating-point-instructions-abs) `abs` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.10](../refs/ptx-isa.html#floating-point-instructions-neg) `neg` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.11](../refs/ptx-isa.html#floating-point-instructions-min) `min` | d | a, b, optional c | Both two-input and three-input f32 forms; whole result. | F |
| [9.7.3.12](../refs/ptx-isa.html#floating-point-instructions-max) `max` | d | a, b, optional c | Both two-input and three-input f32 forms; whole result. | F |
| [9.7.3.13](../refs/ptx-isa.html#floating-point-instructions-rcp) `rcp` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.14](../refs/ptx-isa.html#floating-point-instructions-rcp-approx-ftz-f64) `rcp.approx.ftz.f64` | d | a | Whole f64 result; low result bits are defined as zero, not preserved. | F |
| [9.7.3.15](../refs/ptx-isa.html#floating-point-instructions-sqrt) `sqrt` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.16](../refs/ptx-isa.html#floating-point-instructions-rsqrt) `rsqrt` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.17](../refs/ptx-isa.html#floating-point-instructions-rsqrt-approx-ftz-f64) `rsqrt.approx.ftz.f64` | d | a | Whole f64 result; low result bits are defined as zero, not preserved. | F |
| [9.7.3.18](../refs/ptx-isa.html#floating-point-instructions-sin) `sin` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.19](../refs/ptx-isa.html#floating-point-instructions-cos) `cos` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.20](../refs/ptx-isa.html#floating-point-instructions-lg2) `lg2` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.21](../refs/ptx-isa.html#floating-point-instructions-ex2) `ex2` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.3.22](../refs/ptx-isa.html#floating-point-instructions-tanh) `tanh` | d | a | Whole register result; no implicit CC effect. | F |

### 9.7.4. Half Precision Floating-Point Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.4.1](../refs/ptx-isa.html#half-precision-floating-point-instructions-add) `add` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.4.2](../refs/ptx-isa.html#half-precision-floating-point-instructions-sub) `sub` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.4.3](../refs/ptx-isa.html#half-precision-floating-point-instructions-mul) `mul` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.4.4](../refs/ptx-isa.html#half-precision-floating-point-instructions-fma) `fma` | d | a, b, c | Whole register result; no implicit CC effect. | F |
| [9.7.4.5](../refs/ptx-isa.html#half-precision-floating-point-instructions-neg) `neg` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.4.6](../refs/ptx-isa.html#half-precision-floating-point-instructions-abs) `abs` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.4.7](../refs/ptx-isa.html#half-precision-floating-point-instructions-min) `min` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.4.8](../refs/ptx-isa.html#half-precision-floating-point-instructions-max) `max` | d | a, b | Whole register result; no implicit CC effect. | F |
| [9.7.4.9](../refs/ptx-isa.html#half-precision-floating-point-instructions-tanh) `tanh` | d | a | Whole register result; no implicit CC effect. | F |
| [9.7.4.10](../refs/ptx-isa.html#half-precision-floating-point-instructions-ex2) `ex2` | d | a | Whole register result; no implicit CC effect. | F |

### 9.7.5. Mixed Precision Floating-Point Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.5.1](../refs/ptx-isa.html#mixed-precision-floating-point-instructions-add) `add` | d | a, c | Mixed input types, single f32 destination. | F |
| [9.7.5.2](../refs/ptx-isa.html#mixed-precision-floating-point-instructions-sub) `sub` | d | a, c | Mixed input types, single f32 destination. | F |
| [9.7.5.3](../refs/ptx-isa.html#mixed-precision-floating-point-instructions-fma) `fma` | d | a, b, c | Whole register result; no implicit CC effect. | F |

### 9.7.6. Comparison and Selection Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.6.1](../refs/ptx-isa.html#comparison-and-selection-instructions-set) `set` | d | a, b, optional (possibly negated) c | Boolean modifier adds a predicate input; packed result is one register. | A |
| [9.7.6.2](../refs/ptx-isa.html#comparison-and-selection-instructions-setp) `setp` | p and optional q | a, b, optional (possibly negated) c | Both outputs defined; q combines inverted comparison with c, and need not equal !p. | A |
| [9.7.6.3](../refs/ptx-isa.html#comparison-and-selection-instructions-selp) `selp` | d | a, b, c | Selection inputs and selector are uses; selection is not an instruction execution guard. | A |
| [9.7.6.4](../refs/ptx-isa.html#comparison-and-selection-instructions-slct) `slct` | d | a, b, c | Selection inputs and selector are uses; selection is not an instruction execution guard. | A |

### 9.7.7. Half Precision Comparison Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.7.1](../refs/ptx-isa.html#half-precision-comparison-instructions-set) `set` | d | a, b, optional (possibly negated) c | Boolean modifier adds a predicate input; packed result is one register. | A |
| [9.7.7.2](../refs/ptx-isa.html#half-precision-comparison-instructions-setp) `setp` | p; p and q for x2 | a, b, optional (possibly negated) c | Packed comparisons produce separate lane predicates, not a complementary pair. | A |

### 9.7.8. Logic and Shift Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.8.1](../refs/ptx-isa.html#logic-and-shift-instructions-and) `and` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.8.2](../refs/ptx-isa.html#logic-and-shift-instructions-or) `or` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.8.3](../refs/ptx-isa.html#logic-and-shift-instructions-xor) `xor` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.8.4](../refs/ptx-isa.html#logic-and-shift-instructions-not) `not` | d | a | Whole register result; no implicit CC effect. | A |
| [9.7.8.5](../refs/ptx-isa.html#logic-and-shift-instructions-cnot) `cnot` | d | a | Whole register result; no implicit CC effect. | A |
| [9.7.8.6](../refs/ptx-isa.html#logic-and-shift-instructions-lop3) `lop3` | d and optional p | a, b, c, q for Boolean form | LUT is immediate; Boolean form d\|p has two outputs and an extra predicate input. | A |
| [9.7.8.7](../refs/ptx-isa.html#logic-and-shift-instructions-shf) `shf` | d | a, b, c | Whole register result; no implicit CC effect. | A |
| [9.7.8.8](../refs/ptx-isa.html#logic-and-shift-instructions-shl) `shl` | d | a, b | Whole register result; no implicit CC effect. | A |
| [9.7.8.9](../refs/ptx-isa.html#logic-and-shift-instructions-shr) `shr` | d | a, b | Whole register result; no implicit CC effect. | A |

### 9.7.9. Data Movement and Conversion Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.9.3](../refs/ptx-isa.html#data-movement-and-conversion-instructions-mov) `mov` | d | a / special register / symbol address | Special-register observations need stability classification; symbol address is not a memory load. | A |
| [9.7.9.4](../refs/ptx-isa.html#data-movement-and-conversion-instructions-mov-2) `mov` | all destination components | all source components | Pack/unpack; skip sink _, preserve component order and widths. | A |
| [9.7.9.5](../refs/ptx-isa.html#data-movement-and-conversion-instructions-shfl) `shfl (deprecated)` | d and optional p | a, b, c | Cross-lane exchange; p reports source-range validity. Invalid/inactive source rules are not old-d preservation. Deprecated, target restrictions apply. | C |
| [9.7.9.6](../refs/ptx-isa.html#data-movement-and-conversion-instructions-shfl-sync) `shfl.sync` | d and optional p | a, b, c, membermask | Cross-lane exchange; p reports source-range validity. Invalid/inactive source rules are not old-d preservation. Warp rendezvous and membership constraints. | C |
| [9.7.9.7](../refs/ptx-isa.html#data-movement-and-conversion-instructions-prmt) `prmt` | d | a, b, c | Byte selection/sign replication; whole d, no implicit old destination. | A |
| [9.7.9.8](../refs/ptx-isa.html#data-movement-and-conversion-instructions-ld) `ld` | all d components | address a; optional cache_policy where legal | Memory read; retain state space, width, cache/coherence, volatile/MMIO, ordering and scope as applicable. Narrow loads extend the destination. | M |
| [9.7.9.9](../refs/ptx-isa.html#data-movement-and-conversion-instructions-ld-global-nc) `ld.global.nc` | all d components | address a; optional cache_policy where legal | Memory read; retain state space, width, cache/coherence, volatile/MMIO, ordering and scope as applicable. Narrow loads extend the destination. | M |
| [9.7.9.10](../refs/ptx-isa.html#data-movement-and-conversion-instructions-ldu) `ldu` | all d components | address a; optional cache_policy where legal | Memory read; retain state space, width, cache/coherence, volatile/MMIO, ordering and scope as applicable. Narrow loads extend the destination. | M |
| [9.7.9.11](../refs/ptx-isa.html#data-movement-and-conversion-instructions-st) `st` | — | address a, all b components, optional cache_policy | Memory write; address register is read, never defined. Retain volatile/MMIO and release/scope. | M |
| [9.7.9.12](../refs/ptx-isa.html#data-movement-and-conversion-instructions-st-async) `st.async` | — | address a, b; mbar address on cluster form | Async memory write; cluster form signals barrier transactions; global release/MMIO form has different completion semantics. | M* |
| [9.7.9.13](../refs/ptx-isa.html#data-movement-and-conversion-instructions-multimem-st-async) `multimem.st.async` | — | address a, b | Async multi-address store with release/scope; not a single ordinary address. | U |
| [9.7.9.14](../refs/ptx-isa.html#data-movement-and-conversion-instructions-st-bulk) `st.bulk` | — | address a, size (if register) | Bulk shared zero fill; initval is zero immediate; size is data, not an opcode type width. | M* |
| [9.7.9.15](../refs/ptx-isa.html#data-movement-and-conversion-instructions-multimem) `multimem.ld_reduce , multimem.st , multimem.red` | d components only for ld_reduce | address a; b components for st/red | ld_reduce reads multiple locations; st writes them; red performs RMW. Retain scope/order and vector atomicity rules. | U |
| [9.7.9.16](../refs/ptx-isa.html#data-movement-and-conversion-instructions-prefetch-prefetchu) `prefetch , prefetchu` | — | address a | Cache/tensormap prefetch hint; no ordinary destination; lack of counted bytes does not imply no input. | I |
| [9.7.9.17](../refs/ptx-isa.html#data-movement-and-conversion-instructions-applypriority) `applypriority` | — | address a | Cache eviction-priority effect; size is constrained immediate. | I |
| [9.7.9.18](../refs/ptx-isa.html#data-movement-and-conversion-instructions-discard) `discard` | — | address a | Discard changes validity/content guarantees for cache lines; not an ordinary store or a pure no-op. Size is immediate. | I |
| [9.7.9.19](../refs/ptx-isa.html#data-movement-and-conversion-instructions-createpolicy) `createpolicy` | cache_policy | range: address, primary-size, total-size; fractional: optional fraction; cvt: access-property | Produces a 64-bit policy value, even though counting ignores it. Read only operands actually present and register-valued. | I |
| [9.7.9.20](../refs/ptx-isa.html#data-movement-and-conversion-instructions-isspacep) `isspacep` | p | a | Tests address space, not pointed-to memory. | A |
| [9.7.9.21](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cvta) `cvta` | p (address register) | a or symbol address | Address conversion; p is not necessarily a predicate despite manual operand name. | A |
| [9.7.9.22](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cvt) `cvt` | d | a; b / input tuple / rbits / scale-factor by form | Packed and stochastic/scaled conversions have extra explicit inputs. Random bits are supplied, not hidden RNG state. Whole result including padding/extension. | A |
| [9.7.9.23](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cvt-pack) `cvt.pack` | d | a, b; c for small packed types | Remaining result bits are explicitly copied from c, not implicit old d. | A |
| [9.7.9.24](../refs/ptx-isa.html#data-movement-and-conversion-instructions-mapa) `mapa` | d | a (if register), CTA rank b | Address mapping; symbol address is not a memory read. | A |
| [9.7.9.25](../refs/ptx-isa.html#data-movement-and-conversion-instructions-getctarank) `getctarank` | d | a (if register) | Address query; no load through a. | A |
| [9.7.9.26.3.1](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async) `cp.async` | — | dst/src address registers; src-size or ignore-src; cache_policy where present | Async global→shared copy. cp-size is immediate. ignore-src suppresses source read and zero fills; src-size controls partial zero fill. Completion is separate. | M* |
| [9.7.9.26.3.2](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-commit-group) `cp.async.commit_group` | — | — | Commit non-bulk async group, no register definitions. | S |
| [9.7.9.26.3.3](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-wait-group) `cp.async.wait_group / cp.async.wait_all` | — | — | Wait for non-bulk async groups; N is immediate. wait_all includes commit/wait behavior. | S |
| [9.7.9.26.4.1](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-bulk) `cp.async.bulk` | — | dst/src/mbar addresses as applicable, size, ignoreBytesLeft/Right, ctaMask, cache_policy, byteMask where present | Async copy; choose actual direction and barrier vs bulk-group completion. cp_mask preserves masked destination bytes; ignore_oob yields indeterminate bytes. | U |
| [9.7.9.26.4.2](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk) `cp.reduce.async.bulk` | — | dst/src/mbar addresses as applicable, size, optional cache_policy | Read source plus RMW destination; async completion and ordering differ by direction. | U |
| [9.7.9.26.4.3](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch) `cp.async.bulk.prefetch` | — | src address, size, optional cache_policy | Async cache prefetch, not a register load. | U |
| [9.7.9.26.4.4](../refs/ptx-isa.html#data-movement-and-conversion-instructions-multimem-cp-async-bulk) `multimem.cp.async.bulk` | — | dst/src addresses, size, optional byteMask | Async copy to multiple global locations; bulk-group completion and masked writes. | U |
| [9.7.9.26.4.5](../refs/ptx-isa.html#data-movement-and-conversion-instructions-multimem-cp-reduce-async-bulk) `multimem.cp.reduce.async.bulk` | — | dst/src addresses, size | Async reduction into multiple global locations; RMW, bulk-group completion. | U |
| [9.7.9.26.5.2](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) `cp.async.bulk.tensor` | — | all address/descriptor/coordinate registers; mbar, im2col offsets, ctaMask, cache_policy where present | TMA descriptor and coordinates determine memory accesses. Tile/gather/scatter/im2col variants; barrier or bulk-group completion. Parser currently loses coordinates. | U |
| [9.7.9.26.5.3](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor) `cp.reduce.async.bulk.tensor` | — | tensorMap, all coordinates, src address, optional cache_policy | TMA reduction: source read and destination RMW, bulk-group completion. Parser loses coordinates. | U |
| [9.7.9.26.5.4](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor) `cp.async.bulk.prefetch.tensor` | — | tensorMap, all coordinates, optional im2col inputs and cache_policy | TMA prefetch; no register definitions. Parser loses coordinates. | U |
| [9.7.9.26.6.1](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group) `cp.async.bulk.commit_group` | — | — | Commit bulk async group, distinct from non-bulk cp.async groups. | U |
| [9.7.9.26.6.2](../refs/ptx-isa.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group) `cp.async.bulk.wait_group` | — | — | Bulk-group wait; .read waits for source reads, not necessarily destination writes. N is immediate. | U |
| [9.7.9.27](../refs/ptx-isa.html#data-movement-and-conversion-instructions-tensormap-replace) `tensormap.replace` | — | addr, register-valued new_val | Updates a field of an opaque descriptor in memory. Field ordinal and enumerated values have form-specific immediate restrictions. | U |

### 9.7.13. Control Flow Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.13.1](../refs/ptx-isa.html#control-flow-instructions-curly-braces) `{}` | — | — | Lexical scope construct, not an instruction. Resolve declarations before flattening. | syntax |
| [9.7.13.2](../refs/ptx-isa.html#control-flow-instructions-at) `@` | — | guard p | Execution guard, not an ordinary operand or opcode. False preserves preexisting destinations and suppresses execution effects. | syntax |
| [9.7.13.3](../refs/ptx-isa.html#control-flow-instructions-bra) `bra` | — | guard if present | Label branch; taken/fallthrough edges; .uni constrains uniformity. | B |
| [9.7.13.4](../refs/ptx-isa.html#control-flow-instructions-brx-idx) `brx.idx` | — | index; guard if present | Indirect branch through declared target list; require complete successor set. | B |
| [9.7.13.5](../refs/ptx-isa.html#control-flow-instructions-call) `call` | register return slots if signature uses .reg | register actuals and indirect fptr; parameter slots separately | Call/return state, parameter memory and arbitrary callee effects. CC not preserved. Do not invent SASS caller-clobbers for all PTX virtual registers. | B* |
| [9.7.13.6](../refs/ptx-isa.html#control-flow-instructions-ret) `ret` | — | return parameter state implicitly | Return to caller, potentially divergent; guarded return has fallthrough. | B |
| [9.7.13.7](../refs/ptx-isa.html#control-flow-instructions-exit) `exit` | — | — | Thread termination also affects synchronization participation; guarded exit has fallthrough. | B |

### 9.7.14. Parallel Synchronization and Communication Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.14.1](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-bar) `bar , barrier` | d/p for .red only | barrier id a, optional count b, reduction predicate c (possibly negated) | sync/arrive have no output; reduction has a register output plus CTA collective state. Current tracer excludes both base mnemonics. | S* |
| [9.7.14.2](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-bar-warp-sync) `bar.warp.sync` | — | membermask | Warp synchronization; no output. | S |
| [9.7.14.3](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-barrier-cluster) `barrier.cluster` | — | — | Cluster arrival/wait and scoped ordering, not pure. | S |
| [9.7.14.4](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-membar) `membar / fence` | — | address in tensormap acquire form | Ordering/proxy effects; ordinary forms have no operands. Range size is immediate. Fabric proxy forms excluded. | S |
| [9.7.14.5](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-atom) `atom` | all d components except sinks | a address, b, CAS c, optional cache_policy | Memory RMW and old-value result; CAS condition affects memory update, not whether d is defined. Atomicity granularity is form-specific. | M |
| [9.7.14.6](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-red) `red` | — | a address, b components, optional cache_policy | Memory RMW despite classifier Store convention; vector atomicity is per specified element. | M* |
| [9.7.14.7](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-red-async) `red.async` | — | a address, b; mbar address where present | Async RMW; cluster barrier completion vs global release/MMIO variant. | M* |
| [9.7.14.8](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-multimem-red-async) `multimem.red.async` | — | a address, b | Async multi-address RMW, release/scope. | U |
| [9.7.14.9](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-vote) `vote (deprecated)` | d | predicate a (possibly negated) | Warp-wide reduction/ballot; result depends on participating lanes. Deprecated form. | C |
| [9.7.14.10](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-vote-sync) `vote.sync` | d | predicate a (possibly negated), membermask | Warp-wide reduction/ballot; result depends on participating lanes. Membership/rendezvous constraints. | C |
| [9.7.14.11](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-match-sync) `match.sync` | d and optional p for .all | a, membermask | Cross-lane match. Failure result follows manual (not implicit old d); both outputs are defs when present. | C |
| [9.7.14.12](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-activemask) `activemask` | d | implicit execution participation | Observation of current active lanes, not a function of explicit operands alone. | C |
| [9.7.14.13](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-redux-sync) `redux.sync` | dst | src, membermask | Cross-lane reduction, including floating min/max form; no memory effect implied by name. | C |
| [9.7.14.14](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-griddepcontrol) `griddepcontrol` | — | — | Dependent-grid launch/wait ordering. Ignore is only counting policy. | I |
| [9.7.14.15](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-elect-sync) `elect.sync` | d (unless _) and p | membermask; implicit active participation | Election observes participating lanes; two outputs, no implicit old-d read. | C |
| [9.7.14.16.12](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-init) `mbarrier.init` | — | addr, count | Memory-resident barrier state initialization/invalidation/transaction update; not an ordinary register destination. | S |
| [9.7.14.16.13](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-inval) `mbarrier.inval` | — | addr | Memory-resident barrier state initialization/invalidation/transaction update; not an ordinary register destination. | S |
| [9.7.14.16.14](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx) `mbarrier.expect_tx` | — | addr, txCount | Memory-resident barrier state initialization/invalidation/transaction update; not an ordinary register destination. | S |
| [9.7.14.16.15](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx) `mbarrier.complete_tx` | — | addr, txCount | Memory-resident barrier state initialization/invalidation/transaction update; not an ordinary register destination. | S |
| [9.7.14.16.16](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive) `mbarrier.arrive` | state unless _ | addr, optional count / txCount by form | Barrier state update plus returned phase token; cluster destination can be sink. arrive_drop also changes future arrival count. | S |
| [9.7.14.16.17](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop) `mbarrier.arrive_drop` | state unless _ | addr, optional count / txCount by form | Barrier state update plus returned phase token; cluster destination can be sink. arrive_drop also changes future arrival count. | S |
| [9.7.14.16.18](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive) `cp.async.mbarrier.arrive` | — | addr | Schedules barrier arrival after prior non-bulk copies; .noinc alters barrier bookkeeping. | S |
| [9.7.14.16.19](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait) `mbarrier.test_wait / mbarrier.try_wait` | waitComplete; optional reportPredicate and separate reportValue | addr, state or phaseParity, optional timeHint | Barrier observation; try_wait may suspend. Report outputs are invalid to inspect when waitComplete is false; do not model as preserving old outputs. Acquire effect only on successful wait. | S* |
| [9.7.14.16.20](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count) `mbarrier.pending_count` | count | state | Pure decode of a valid noComplete arrival token, not a memory/barrier update. | S |
| [9.7.14.16.21](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-mbarrier-check-layout) `mbarrier.check_layout` | p | addr | Reads barrier layout from memory; produces predicate. | S |
| [9.7.14.17](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy) `tensormap.cp_fenceproxy` | — | dst/src addresses | Descriptor copy plus proxy release fence; size immediate, not merely synchronization. | U |
| [9.7.14.18](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel) `clusterlaunchcontrol.try_cancel` | — | addr, mbar addresses | Async cancellation request; writes shared response and signals barrier. Scheduler effect. | U |
| [9.7.14.19](../refs/ptx-isa.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel) `clusterlaunchcontrol.query_cancel` | predicate / coordinate register(s), by query form | try_cancel_response | Response decode. Coordinate query requires success; fourth vector component unspecified (sink in syntax). | U |

### 9.7.15. Warp Level Matrix Multiply-Accumulate Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.15.4.3](../refs/ptx-isa.html#warp-level-matrix-instructions-wmma-ld) `wmma.load` | all fragment r components | p address, optional stride | Warp collective memory load; layout/shape/fragment mapping required. | M |
| [9.7.15.4.4](../refs/ptx-isa.html#warp-level-matrix-instructions-wmma-st) `wmma.store` | — | p address, all r components, optional stride | Warp collective memory store. | M |
| [9.7.15.4.5](../refs/ptx-isa.html#warp-level-matrix-instructions-wmma-mma) `wmma.mma` | all d components | all a, b, c components | Register fragments collectively compute A×B+C. c is explicit input even when aliased with d. | F/U |
| [9.7.15.5.14](../refs/ptx-isa.html#warp-level-matrix-instructions-mma) `mma` | all d components | a, b, c fragments; block-scale metadata and selector tuples where present | Warp collective arithmetic. Block-scale/type variants change arity; no memory read merely because matrices are involved. | F/U |
| [9.7.15.5.15](../refs/ptx-isa.html#warp-level-matrix-instructions-ldmatrix) `ldmatrix` | all r components | p address | Shared-memory collective load; packed formats alter shape/width, not destination role. | M |
| [9.7.15.5.16](../refs/ptx-isa.html#warp-level-matrix-instructions-stmatrix) `stmatrix` | — | p address, all r components | Shared-memory collective store. | M |
| [9.7.15.5.17](../refs/ptx-isa.html#warp-level-matrix-instructions-movmatrix) `movmatrix` | all d components | a components across lanes | Warp matrix transpose is register exchange, not memory movement. | U |
| [9.7.15.6.3](../refs/ptx-isa.html#warp-level-matrix-instructions-sparse-mma) `mma.sp / mma.sp::ordered_metadata` | all d components | a, b, c fragments, sparsity e; scale data and selector tuples for block_scale | Sparse/ordered metadata variants. Sparsity selector f is immediate. Collective register arithmetic. | U |

### 9.7.16. Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.16.5.2](../refs/ptx-isa.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma) `wgmma.mma_async` | all d accumulator components (pending) | old d when scale-d; a fragment or a-desc, b-desc, scale-d | Async register output and descriptor-based shared reads. Fence/commit/wait protocol; A input-register lifetime also extends to completion. Scale/transpose immediates are not register reads. | U |
| [9.7.16.6.3](../refs/ptx-isa.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma-sp) `wgmma.mma_async.sp` | all d accumulator components (pending) | old d when scale-d; a fragment or a-desc, b-desc, scale-d; sp-meta (sp-sel immediate) | Async register output and descriptor-based shared reads. Fence/commit/wait protocol; A input-register lifetime also extends to completion. Scale/transpose immediates are not register reads. | U |
| [9.7.16.7.1](../refs/ptx-isa.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence) `wgmma.fence` | — | implicit prior register-access ordering | Fence accumulator/A register accesses before wgmma; distinct from shared-memory proxy fence. | U |
| [9.7.16.7.2](../refs/ptx-isa.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group) `wgmma.commit_group` | — | — | Commit wgmma group; no ordinary register destination. | U |
| [9.7.16.7.3](../refs/ptx-isa.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group) `wgmma.wait_group` | — | — | Wait makes relevant pending accumulator results safe to access; N immediate. | U |

### 9.7.17. TensorCore 5th Generation Family Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.17.7.1](../refs/ptx-isa.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit) `tcgen05.alloc , tcgen05.dealloc , tcgen05.relinquish_alloc_permit` | — | alloc: dst address, nCols; dealloc: taddr, nCols; relinquish: none | alloc writes tensor-memory address into shared memory, not dst register. Allocation pool/lifetime and collective constraints. | U |
| [9.7.17.8.3](../refs/ptx-isa.html#tcgen05-instructions-tcgen05-ld) `tcgen05.ld` | all r components; separate redval for .red (pending) | taddr | Async tensor-memory load, optionally also reduction result. immHalfSplitoff immediate. .sync is rendezvous, not result completion; wait::ld required. | U |
| [9.7.17.8.4](../refs/ptx-isa.html#tcgen05-instructions-tcgen05-st) `tcgen05.st` | — | taddr, all r components | Async tensor-memory store. Operand order differs for split shape; wait::st completion. | U |
| [9.7.17.8.5](../refs/ptx-isa.html#tcgen05-instructions-tcgen05-wait) `tcgen05.wait` | — | — | Separate load/store completion waits, no explicit register defs. | U |
| [9.7.17.9.2](../refs/ptx-isa.html#tcgen05-instructions-tcgen05-cp) `tcgen05.cp` | — | taddr, s-desc | Async shared→tensor-memory copy with multicast/layout semantics. | U |
| [9.7.17.9.3](../refs/ptx-isa.html#tcgen05-instructions-tcgen05-shift) `tcgen05.shift` | — | taddr | Async tensor-memory RMW shift; address register remains unchanged. | U |
| [9.7.17.10.9.1](../refs/ptx-isa.html#tcgen05-mma-instructions-mma) `tcgen05.mma` | — | d-tmem address; a-desc or a-tmem; b-desc, idesc, enable-input-d; scale addresses, output-lane mask registers by form | Async MMA into tensor memory, conditional accumulator read; output lane suppression, block scaling, collector A, optional .ashift. Descriptor/address operands are uses, not defs. | U |
| [9.7.17.10.9.2](../refs/ptx-isa.html#tcgen05-mma-instructions-mma-sp) `tcgen05.mma.sp` | — | d-tmem address; a-desc or a-tmem; b-desc, idesc, enable-input-d; sparse metadata address; scale addresses, output-lane mask registers by form | Async MMA into tensor memory, conditional accumulator read; output lane suppression, block scaling, collector A, optional .ashift. Descriptor/address operands are uses, not defs. | U |
| [9.7.17.10.9.3](../refs/ptx-isa.html#tcgen05-mma-instructions-mma-ws) `tcgen05.mma.ws` | — | d-tmem address; a-desc or a-tmem; b-desc, idesc, enable-input-d; optional zero-column-mask-desc | Async MMA into tensor memory, conditional accumulator read; collector B state, zero-column mask. Descriptor/address operands are uses, not defs. | U |
| [9.7.17.10.9.4](../refs/ptx-isa.html#tcgen05-mma-instructions-mma-ws-sp) `tcgen05.mma.ws.sp` | — | d-tmem address; a-desc or a-tmem; b-desc, idesc, enable-input-d; sparse metadata address; optional zero-column-mask-desc | Async MMA into tensor memory, conditional accumulator read; collector B state, zero-column mask. Descriptor/address operands are uses, not defs. | U |
| [9.7.17.11.1](../refs/ptx-isa.html#tcgen05-special-sync-operations-fence) `tcgen05.fence` | — | — | Orders tcgen05 async operations around thread synchronization. | U |
| [9.7.17.12.1](../refs/ptx-isa.html#tcgen-async-sync-operations-commit) `tcgen05.commit` | — | mbar address, optional ctaMask | Barrier completion tracking for the specified tcgen05 operations; not interchangeable with wait::ld/st. | U |

### 9.7.18. Stack Manipulation Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.18.1](../refs/ptx-isa.html#stack-manipulation-instructions-stacksave) `stacksave` | d | implicit stack pointer | Reads stack state; no ordinary explicit source. | A |
| [9.7.18.2](../refs/ptx-isa.html#stack-manipulation-instructions-stackrestore) `stackrestore` | — | a | Writes implicit stack pointer; invalidates allocation lifetimes. Current first-destination heuristic is wrong. | A* |
| [9.7.18.3](../refs/ptx-isa.html#stack-manipulation-instructions-alloca) `alloca` | ptr | size; implicit stack/allocation state | Updates stack pointer and creates local allocation. Alignment immediate; not a pure address conversion. | A |

### 9.7.20. Miscellaneous Instructions

| ISA section / instruction | W | R | Execution effects and decoding requirements | Counting route |
|---|---|---|---|---|
| [9.7.20.1](../refs/ptx-isa.html#miscellaneous-instructions-brkpt) `brkpt` | — | — | Execution suspension/debug effect; not an unconditional permanent exit. | B |
| [9.7.20.2](../refs/ptx-isa.html#miscellaneous-instructions-nanosleep) `nanosleep` | — | t if register | Sleep/time/scheduling effect; current heuristic invents a definition of t. | I* |
| [9.7.20.3](../refs/ptx-isa.html#miscellaneous-instructions-pmevent) `pmevent` | — | — | Performance event; event number/mask immediate. | I |
| [9.7.20.4](../refs/ptx-isa.html#miscellaneous-instructions-trap) `trap` | — | — | Abort/host interrupt; must not have normal executing fallthrough. Current CFG does. | B* |
| [9.7.20.5](../refs/ptx-isa.html#miscellaneous-instructions-setmaxnreg) `setmaxnreg` | — | — | Register-resource ownership adjustment, collective constraints; inc may block. Immediate count, no named PTX-register destination. | I |

## Findings in the current implementation

These are source-level findings at the revision above. “Classified” below means a
counting category was returned, not that an instruction was semantically decoded.

| Priority | Finding and evidence | Consequence / required change before relying on SSA |
|---|---|---|
| P0 | [Parser memory operand handling](../src/ptx/parse/parser.rs#L793) retains only a base and optional numeric offset, then skips to `]`. The TMA probe below loses both coordinate registers. | Successful parse does not establish complete register uses. Preserve structured address operands, or mark the affected instruction explicitly unsuitable for effect decoding. |
| P0 | [Register operand parsing](../src/ptx/parse/parser.rs#L775) classifies bare names as `SymbolRef`; [scope renaming](../src/ptx/parse/parser.rs#L409) renames labels only. The declaration/guard paths accept bare register names. | Resolve symbols from declarations; distinguish same-named scoped registers. The printed spelling alone is insufficient for SSA identity. |
| P0 | [Definition discovery](../src/analysis/scalar/trace.rs#L65) accepts only `Operand::Register` in operand 0; the same pattern occurs in [induction discovery](../src/analysis/scalar/trace.rs#L177). | Misses vector and pipe-separated destinations, separately positioned outputs (`tcgen05.ld.red`, reporting waits), bare register destinations, and implicit carry. A prior scalar definition can incorrectly appear to survive a later tuple definition. |
| P0 | [`defines_dest`](../src/analysis/scalar/trace.rs#L905) is a negative base-mnemonic list. It excludes all `bar`/`barrier`, but assumes the first register of nearly everything else is a destination. | Misses barrier reduction results; invents definitions for `stackrestore %r` and `nanosleep %r`; also misreads `tcgen05.dealloc` if reached. Replace with explicit full-form decoding and an unknown result, not a longer blacklist. |
| P0 | [Scalar tracer](../src/analysis/scalar/trace.rs) does not inspect `instr.predicate`; its definition and arithmetic paths therefore have no false-guard preservation semantics. | A guarded `mov`/increment can be followed as an unconditional assignment. Counting's `predicated` bit does not repair value analysis. Add guarded definitions/merges or conservatively refuse these traces. |
| P0 | [CFG terminator set](../src/ptx/cfg.rs#L80) recognizes `bra`, `brx`, `ret`, `exit`, not `trap`; unresolved targets are recorded separately. | `trap` is not ordinary fallthrough. Sound SSA requires complete control flow and a policy for unresolved branches/unparsed instructions. Reporting an unknown elsewhere does not make dominance over missing edges sound. |
| P1 | [Operand grammar](../src/ptx/parse/parser.rs#L775) has no `!` source operand or parenthesized argument-list branch; probes reproduce `Unparsed` for both. [Top-level parser](../src/ptx/parse/parser.rs#L151) has no `.func` arm. | Boolean source negation and normal call syntax need explicit representation; function/prototype/parameter support precedes interprocedural effects. Do not treat a failed call parse as zero effects. |
| P1 | [IR](../src/ptx/ir.rs#L38) and [classifier](../src/analysis/instruction_counts/classify.rs#L180) have no resolved opcode/signature or typed register-effects result. The classifier matches mostly base mnemonics and selected modifiers. | An `OpClass` cannot be repurposed as an effect contract. Known register effects and unknown arithmetic semantics need different states; so do unknown effects and a known empty effect set. |
| P1 | [`cp` classifier](../src/analysis/instruction_counts/classify.rs#L502) gathers immediate operands after the addresses and assumes the first/second immediate are write/read sizes. It ignores a register `ignore-src`/runtime size in this calculation. | Probe with `ignore-src=%p` returns fixed 16-byte source read; PTX can suppress that read. Operand roles, types, and predicates must drive counts. An immediate cache policy can also be mistaken for a source size. |
| P1 | [Broad memory/sync arms](../src/analysis/instruction_counts/classify.rs#L263) route `st.async`, `red.async`, `st.bulk`, and all `mbarrier` forms into existing classes. | Async completion, dynamic bulk size, RMW, conditional report outputs and actual barrier memory effects are absent from these classes. `st.bulk` is unquantified under the type-width helper, not measured by its size operand. |
| P2 | [Ignore arm](../src/analysis/instruction_counts/classify.rs#L315) includes `createpolicy`, `griddepcontrol`, `discard`, `setmaxnreg`, and `nanosleep`. | Zero roofline contribution is not proof of no register or execution effects, nor permission to delete/reorder the instruction. `createpolicy` even defines an explicit register. |

`nop` is recognized by the classifier and excluded from the tracer's destination
heuristic, but has no standalone instruction section in this pinned manual. Treat
it as an implementation spelling with an explicit provenance/validation decision,
not an ISA-verified family. The syntax-only parser can accept arbitrary invented
mnemonics; that is not support for their semantics.

### Reproduced parser/classifier probes

Built the current library with `cargo build --lib --quiet`, then compiled a temporary
Rust client against `target/debug/libptxroof.rlib`. Each body was wrapped in a
`.version 9.3`, `.target sm_100`, `.address_size 64`, `.entry k()` shell. These probes
test **this parser**, not target legality or assembler acceptance.

| Body / observation | Actual result |
|---|---|
| `setp.lt.s32 %p\|%q, %r1, %r2;` | Parsed; first operand is a `MultipleDestinations` list; counting returns Predicate. Tracer's scalar-first check cannot record either output. |
| `setp.lt.and.s32 %p, %r1, %r2, !%q;` | `Stmt::Unparsed`. |
| `call (%r1), foo, (%r2);` | `Stmt::Unparsed`; therefore it does not become a recorded parsed call site. |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r3}], [%r4];` | Parsed as three `Memory` operands based on `%r1`, `%rd1`, `%r4`; `%r2`, `%r3` absent from operands. Counting returns Unknown. |
| `.reg .u32 r; mov.u32 r, 1;` | Destination is `SymbolRef(r)`, not `Register(r)`; counting returns Move. |
| Two sibling scopes each declare `.reg .u32 %r` and write it | Both destinations have the same interned register symbol `%r`. |
| `cp.async.ca.shared.global [%r1], [%rd1], 16, %p;` | Copy: `read_bytes=Some(16)`, `written_bytes=Some(16)` regardless of the ignore-source predicate. |
| `bar.red.popc.u32 %r1, 0, %p;` | Sync, with an explicit first-operand register result. |
| `stackrestore.u64 %rd1;` | Move, with first operand `%rd1`; the manual identifies it as a source. |
| `tcgen05.ld.red.sync.aligned.32x32b.x2.min.u32 {%r1,%r2}, %r3, [%r4];` | Parsed; Unknown counting category; outputs occupy both first and second operand positions. |

A minimal client to independently inspect these results (substitute a body above):

```rust
use ptxroof::ptx::{ir::Stmt, parse::parser::parse};
use ptxroof::analysis::instruction_counts::classify::classify;

fn main() {
    let body = "setp.lt.s32 %p|%q, %r1, %r2;";
    let source = format!(
        ".version 9.3\n.target sm_100\n.address_size 64\n.entry k() {{ {body} }}"
    );
    let m = parse(&source).unwrap();
    for s in &m.kernels[0].stmts {
        println!("{s:?}");
        if let Stmt::Instr(i) = s {
            println!("{:?}", classify(&m, i));
            for &id in m.operand_ids(i.operands) {
                println!("{:?}", m.operand(id));
            }
        }
    }
    // Debug output includes pooled children and the symbol interner.
    println!("{m:#?}");
}
```

```sh
cargo build --lib --quiet
rustc --edition=2024 /tmp/ptx-effects-probe.rs \
  --extern ptxroof=target/debug/libptxroof.rlib \
  -L dependency=target/debug/deps -o /tmp/ptx-effects-probe
/tmp/ptx-effects-probe
```

### Documentation reconciliation

The earlier counting audit is about FLOPs/bytes, not SSA coverage. Its `setp` and
`match.sync` rows still describe pipe destinations as unparsed although the parser
now has `MultipleDestinations` and a regression test. The correct remaining issue
is effect discovery, not parsing that particular syntax. Its introductory historical
note is more current than those rows. Also, the trailing “unsupported families”
comment in `classify.rs` still names several now-implemented families. Neither
comment was used as evidence of current behavior. This report's routes come from
the actual match arms/helpers; it does not certify the older manually maintained
“77 of 232” aggregate as a freshly recomputed support metric.

## A small implementation plan supported by the audit

1. **Make register uses recoverable.** Resolve scoped declarations and bare names;
   retain negated operands, tuples, call lists, and structured tensor addresses.
   Until each form is represented faithfully, return a structured unsupported
   reason. Preserve the original statement span so unsupported syntax is inspectable.
2. **Add one fallible register-effects decoder.** Conceptually:
   `decode_register_effects(&ResolvedProgram, InstructionId) -> Result<RegisterEffects, UnsupportedEffects>`.
   Return guard, input operand sites, explicit output sites, and carry-state uses/
   definitions. Include type/width information from the resolved program. Decode
   opcode plus modifiers **and operand form**; validate form/version/target before
   promising a complete effect set. Call signatures are explicit inputs if calls
   are supported. Unknown operations must not default to “first operand is a def”.
3. **Start SSA with a declared synchronous subset.** Include ordinary arithmetic,
   conversions, complete load/atomic outputs as opaque values, tuples, predicates,
   and carry chains. Use guarded operations or explicit merges. Require complete
   CFGs, resolved identities, and known effects. An unsupported instruction either
   blocks complete SSA for the function or creates an explicitly partial result
   whose consumers cannot mistake it for complete SSA. A guessed clobber list is
   not a substitute for missing control flow or operands.
4. **Keep value analysis separate.** A decoder can know that an unsupported
   arithmetic operation defines a register even if the affine evaluator cannot
   compute its value. Preserve that opaque definition so an earlier constant does
   not leak through it. Arithmetic overflow, signedness, shift behavior, rounding,
   NaNs, FTZ, saturation, and approximate results belong to transfer functions,
   not destination guessing.
5. **Add async support when a consumer needs it.** Model pending results, conditional
   accumulator uses, completion groups, and validity/lifetime constraints for
   `wgmma` and `tcgen05.ld`; otherwise reject them explicitly. Add separate memory/
   synchronization summaries only to the detail required by the consuming pass.
   Memory optimization needs far more information than scalar SSA construction.

There is no need to encode the entire ISA as hundreds of Rust operation structs.
Small shared decoder helpers are appropriate **after** selecting a known legal
form: ordinary destination-plus-sources, tuples, no-destination memory operations,
carry operations, and specialized multi-output forms. An owned result structure
and typed IDs are sufficient to make the analysis boundary pure. Avoid naming
that limited result `InstructionEffects` if it deliberately omits memory, control,
resource, and completion effects.

A practical type distinction is between a complete register-effects result, an
unsupported-decoding error, and an SSA graph verified for the chosen subset.
Consumers requiring completeness should accept the verified graph type. A public
constructor accepting arbitrary vectors would not enforce that invariant; the
builder/verifier must own its construction. This still leaves semantic correctness
to the decoder and tests—the Rust type system cannot prove its PTX table correct.

## Required validation when implementation begins

This audit added no functional implementation or permanent tests. The following
are acceptance criteria for the proposed decoder/SSA work, not tests claimed to
pass today:

- One inventory entry per manual section above, with explicit supported/unsupported
  form outcomes. Verify arity-changing variants independently: Boolean `lop3`,
  packed `setp`, three-input min/max, stochastic/scaled `cvt`, reporting waits,
  `tcgen05.ld.red`, block-scaled/sparse matrices, and call signatures.
- Probe every destination shape: scalar, vector, pipe pair, separate operand output,
  sink, and a source aliased with an output. Ensure reads use the old version.
- Exercise `%` and bare names, duplicate names in separate scopes, declared vectors,
  register arrays, special-register components, address/descriptor inputs, and
  tensor coordinates; reject forms whose dependencies were not preserved.
- Test guarded overwrite, guarded self-update, guard also written by the instruction,
  and guarded carry chains. Confirm false guards retain old values and suppress
  side effects. Exercise branch diamonds, backedges, unreachable blocks, predicated
  exits, traps, and unresolved targets.
- Confirm known-but-opaque definitions kill old values; unparsed/unknown effects
  block completeness rather than disappearing. Cover calls and their parameter
  memory, and carry invalidation across calls.
- For optional async support, test scale-d false/true, issue→commit→wait, insufficient
  waits, use-before-completion, A-register lifetimes, conditional wait results,
  barrier phases, and source-read-only bulk waits.
- Use `ptxas` with the correct toolkit/target to validate representative legal forms
  and negative forms. Hardware tests can confirm selected execution behavior, but
  neither successful assembly nor corpus counting proves the def-use contract.

Inventory completeness was checked by comparing the ledger's 186 section anchors
with the independently extracted in-scope Syntax sections of the pinned HTML;
there were no missing or extra sections. All relative file links and manual anchors
were checked locally. The library build and the temporary probes above succeeded.
The full CI suite was not rerun: this deliverable changes documentation only, and
the probes specifically establish the behavioral claims made here.
