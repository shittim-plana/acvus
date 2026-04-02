# acvus MIR — Compiler Engineer's Guide

This document is for compiler engineers modifying the `acvus-mir` codebase. Part I maps the territory — crate hierarchy, data structures, and the full compilation pipeline. Part II explains why things are the way they are, and what breaks if you change them.

---

# Part I — The Territory

## Crate Hierarchy

```
acvus-utils          Foundation: Interner, Astr, Freeze, LocalId, QualifiedRef
  |
acvus-ast            Parser: Template/Script → AST (Expr, Stmt, Pattern, Pipe)
  |
acvus-mir            Compiler core: type system, IR, analysis, optimization
  |
  +-- acvus-interpreter    Runtime: register-based VM, Executor trait, Spawn/Eval
  |     |
  +-- acvus-ext            Standard library: builtins (iter, map, filter, fold, ...)
  |     |
  +-- acvus-orchestration  Multi-source compilation, incremental rebuild
  |
acvus-lsp            Language server (shares exact same pipeline as compiler)
```

**Key rule:** `acvus-mir` knows nothing about the interpreter. The IR is designed for *any* backend. `acvus-interpreter` depends on `acvus-mir`, never the reverse.

---

## Core Data Structures

### IR Representations

```
MirBody                          CfgBody
+-- insts: Vec<Inst>             +-- blocks: Vec<Block>
+-- val_types: Map<ValueId,Ty>       +-- params: Vec<ValueId>
+-- param_regs: Vec<ValueId>         +-- insts: Vec<Inst>      (no control flow)
+-- capture_regs: Vec<ValueId>       +-- terminator: Terminator
+-- val_factory: LocalFactory    +-- label_to_block: Map<Label,BlockIdx>
+-- debug: DebugInfo             +-- val_types, val_factory, ... (shared)
+-- label_count: usize

MirModule                        Terminator
+-- main: MirBody                +-- Jump { label, args }
+-- closures: Map<Label,MirBody> +-- JumpIf { cond, then/else_label+args }
                                 +-- ListStep { dst, list, index_src/dst, done }
                                 +-- Return(ValueId)
                                 +-- Fallthrough
```

**MirBody** is the flat representation — a linear stream of instructions including control flow (`BlockLabel`, `Jump`, `JumpIf`, `Return`). This is what lowering produces and what the interpreter consumes.

**CfgBody** is the structured representation — basic blocks own their instructions, control flow is in terminators. This is what all analysis and optimization passes operate on.

**Lifecycle:** `promote(MirBody) → CfgBody` → run passes → `demote(CfgBody) → MirBody`.

### InstKind — The 40 Instructions

| Category | Instructions | Notes |
|----------|-------------|-------|
| Constants | `Const`, `Undef`, `Poison` | Undef = SSA placeholder, Poison = type error marker |
| Context | `ContextProject`, `ContextLoad`, `ContextStore` | Project+Load always paired by lowerer |
| Variables | `VarLoad`, `VarStore`, `ParamLoad` | Disappear after SSA (become block params) |
| Arithmetic | `BinOp`, `UnaryOp`, `Cast` | Cast = subtype coercion (DequeToList, RangeToList, Extern) |
| Field access | `FieldGet`, `ObjectGet`, `ListIndex`, `ListGet`, `ListSlice`, `TupleIndex`, `UnwrapVariant` | All pure |
| Construction | `MakeDeque`, `MakeObject`, `MakeTuple`, `MakeRange`, `MakeVariant`, `MakeClosure` | All pure |
| Test predicates | `TestLiteral`, `TestRange`, `TestVariant`, `TestListLen`, `TestObjectKey` | Return Bool, used by JumpIf |
| Functions | `LoadFunction`, `FunctionCall` | FunctionCall has context_uses/context_defs (populated by SSA pass) |
| Async | `Spawn`, `Eval` | Created by SpawnSplit from IO FunctionCalls |
| Control flow | `BlockLabel`, `Jump`, `JumpIf`, `Return`, `ListStep` | Become terminators in CfgBody |
| Utility | `Nop` | Placeholder after instruction removal |

### Type System

```
Ty::Int | Float | String | Bool | Unit | Range | Byte
   | List(Box<Ty>)
   | Deque(Box<Ty>, Box<Ty>)       -- mutable list with Identity tracking
   | Object(Map<Astr,Ty>)          -- structural typed record
   | Tuple(Vec<Ty>)
   | Fn { params, ret, captures, effect }
   | Handle(Box<Ty>, Effect)       -- async handle from Spawn
   | UserDefined(QualifiedRef)     -- plugin-defined types (Iterator, Sequence, ...)
   | Option(Box<Ty>)
   | Enum(Map<Astr,Option<Ty>>)    -- open enum (variants auto-declared on use)
   | Identity(IdentityKind)        -- deque origin tracking
   | Param(ParamToken)             -- generic type variable
   | Error(ErrorToken)             -- type error sentinel
```

### Effect System

```
Effect::Resolved(EffectSet) | Var(u32)

EffectSet {
    reads:  BTreeSet<EffectTarget>,   -- contexts/tokens this function reads
    writes: BTreeSet<EffectTarget>,   -- contexts/tokens this function writes
    io: bool,                         -- opaque IO (network, filesystem, ...)
    self_modifying: bool,             -- value consumed on use (iterator advance)
}

EffectTarget::Context(QualifiedRef)   -- SSA-compatible, becomes value flow
EffectTarget::Token(TokenId)          -- NOT SSA-compatible, ordering preserved
```

**The Context/Token split is the key design decision.** Contexts become SSA values after the SSA pass (no mutable state). Tokens remain ordering constraints (functions sharing a Token execute sequentially). This enables automatic parallelization of context-using code while preserving correctness for external shared state.

### Identity System (Collection Provenance)

```
Identity::Concrete(IdentityId)   -- fixed identity from [] literals
Identity::Fresh(IdentityId)      -- signature-level marker, becomes Concrete on instantiate
```

Deques carry an identity: `Deque(elem_ty, Identity)`. Two deques with different identities cannot be mixed — unification fails. This prevents silently merging data from unrelated sources (e.g., `extend(@history, @other_history)` is a type error unless explicitly coerced).

**Identity is invariant.** No covariance, no subtyping. Exact equality or error.

**On instantiation:** `Fresh(id)` → `Concrete(new_id)`. All `Fresh(id)` with the same ID within one instantiation map to the same new `Concrete`. This allows function signatures to express "returns a deque with a fresh, unique identity."

**LUB rule:** When two deques with different identities meet (e.g., at a branch merge), the LUB is `List<T>` — identity is erased. This only succeeds in covariant position; in invariant position, it's an error.

### Type Inference Variables

```
Param { token: ParamToken, constraint: Option<ParamConstraint> }

ParamToken(u32)                -- opaque, allocated by TySubst.fresh_param()
ParamConstraint(Vec<Ty>)       -- allowed-type set (e.g., scalar = {Int,Float,String,Bool,Byte})
```

Params are unification variables. They start unbound and get bound through `unify()`. Once bound, `resolve()` follows the binding chain.

- `Param + concrete`: check constraint (if any), then bind.
- `Param + Param`: merge constraints via **intersection**. Empty intersection = immediate type error.
- `Param + self`: no-op (already unified).
- Occurs check prevents infinite types (Param cannot bind to a type containing itself).

### Polarity (Variance)

```
Polarity::Covariant       -- a ≤ b (subtype allowed)
Polarity::Contravariant   -- b ≤ a (reversed)
Polarity::Invariant       -- a = b (exact match)
```

There is no separate Variance enum. Variance is encoded through Polarity propagation during unification.

| Position | Polarity | Why |
|----------|----------|-----|
| Function params | `pol.flip()` (contravariant) | Consumer: accepts supertypes |
| Function return | `pol` (covariant) | Producer: returns subtypes |
| Collection elements | `Invariant` | Mutable container: read + write |
| Type args (UserDefined) | `Invariant` | No implicit widening |
| Effect args | Caller's polarity | Follows function position |
| Identity | `Invariant` | Provenance must match exactly |

### Unification Algorithm (`TySubst::unify`)

```
unify(a: &Ty, b: &Ty, pol: Polarity) → Result<(), (Ty, Ty)>
```

1. **Shallow resolve** both sides (follow Param bindings without recursing into structure).
2. **Trivial cases**: `Error` unifies with anything (poison). Identical scalars succeed.
3. **Param cases**: occurs check → constraint intersection → bind.
4. **Structural recursion**:
   - `List(a)` vs `List(b)` → `unify(a, b, Invariant)`
   - `Object(fields_a)` vs `Object(fields_b)` → merge fields, unify shared keys invariantly
   - `Fn{params_a, ret_a, eff_a}` vs `Fn{params_b, ret_b, eff_b}` → params contravariant, ret covariant, effect via `unify_effects`
   - `Enum` → merge variant sets (open enum), unify overlapping payloads
5. **Mismatch in non-invariant context** → try `try_coerce(sub, sup)` or `lub_or_err`.

### Coercion Rules (`try_coerce`)

Built-in coercions (only in covariant/contravariant context, never invariant):

| From | To | Rule |
|------|----|------|
| `Deque<T, O>` | `List<T>` | Identity erased, becomes immutable |
| `Range` | `List<Int>` | Range expanded to list |
| `UserDefined(A)` | any `B` | Via ExternCast `from_rules` |
| any `A` | `UserDefined(B)` | Via ExternCast `to_rules` |

**ExternCast** — plugin-registered coercion rules:

```
CastRule { from: Ty, to: Ty, fn_ref: QualifiedRef }
```

Resolution is two-phase: (1) probe all matching rules with snapshot+rollback to count matches, (2) if exactly one matches, apply it with persistent bindings. Ambiguity (multiple matches) = error. The matching ExternFn is recorded for the lowerer to emit a `Cast { kind: Extern(fn_ref) }` instruction.

### Least Upper Bound (LUB)

`try_lub(a, b) → Option<Ty>` — finds the least common supertype when unification fails in non-invariant context.

| Mismatch | LUB | Condition |
|----------|-----|-----------|
| `Deque<T, O1>` vs `Deque<T, O2>` | `List<T>` | Different identity → erase |
| `Fn{eff1}` vs `Fn{eff2}` (same structure) | `Fn{eff1 ∪ eff2}` | Effect union |
| `UserDefined(A)` vs `UserDefined(B)` | CastRule target | Find common target both can coerce to |
| Invariant position | — | LUB never attempted; immediate error |

When LUB succeeds, any Params bound to the original types are **rebound** to the LUB type.

### Effect Unification

```
unify_effects(a: &Effect, b: &Effect, pol: Polarity) → Result<(), ()>
```

- `Pure` ≤ `Effectful`: pure is a sub-effect of any effect.
- `Covariant`: pure on left, effectful on right → OK (producer is safer than expected).
- `Contravariant`: effectful on left, pure on right → OK (consumer accepts more).
- `Invariant`: both must be identical (or both resolved → union).
- `Effect::Var`: binds to concrete regardless of polarity.
- When both sides are `Resolved`, effects are **unioned** (reads ∪ reads, writes ∪ writes, io ∨ io).

### Type Inference Pipeline (SCC-based)

```
[Build call graph]  function → callees
        |
[Tarjan's SCC]  reverse topological order (leaves first)
        |
[Per-SCC inference]
   For each function in SCC:
     1. Allocate fresh Param (return type) + fresh Effect::Var
     2. Build tentative Ty::Fn with fresh vars
     3. Run TypeChecker on AST body (fills subst via unify)
     4. unify_effect(effect_var, body_effect)
     5. Resolve all vars → concrete Ty::Fn
   Available to subsequent SCCs as concrete types.
        |
[Effect propagation]  transitive closure: if f calls g, f.effect ⊇ g.effect
        |
[Completeness check]  all Params resolved? Effect constraints satisfied?
        |
InferResult { outcomes, fn_params, context_types }
```

**Key property:** Within an SCC, functions see each other's **tentative** types (with unbound vars). After the SCC finishes, all vars are resolved. The next SCC sees only concrete types. This allows polymorphic recursion within an SCC while maintaining concrete types across SCC boundaries.

### Materiality (Serialization Boundaries)

```
Materiality::Concrete    -- scalars: always safe across boundaries
Materiality::Composite   -- containers: safe if contents are safe
Materiality::Ephemeral   -- opaque: never safe (UserDefined, Fn, Handle)
```

`is_materializable()` determines if a type can be stored to context (`ContextStore`). Functions, handles, and opaque user-defined types are **ephemeral** — they exist only as SSA values, never persisted.

### Compilation Graph

```
CompilationGraph {
    functions: Vec<Function>,    -- all callable entities (local + builtin + extern)
    contexts:  Vec<Context>,     -- all named external values (@user, @items, ...)
}

Function { qref: QualifiedRef, kind: FnKind, constraint: FnConstraint }
FnKind::Local(ParsedAst)   -- user-written template/script
FnKind::Extern             -- plugin-provided handler

Context { qref: QualifiedRef, constraint: Constraint }
Constraint::Exact(Ty) | Inferred | DerivedFnOutput(..) | DerivedContext(..)
```

---

## The Full Compilation Pipeline

```
Source text
    |
    v
[Parse]  acvus_ast::parse / parse_script
    |
    v
ParsedAst (Template | Script)
    |
    v
[Build CompilationGraph]  register builtins + externs + local functions + contexts
    |
    v
CompilationGraph { functions, contexts }
    |
    v
[Phase 0: Extract]  graph/extract.rs
    |    Parse source, discover context references, trace projection chains
    |    to determine which contexts are read vs written.
    |    Output: ExtractResult { fn_refs: Map<QualifiedRef, FnRefs> }
    v
[Phase 1: Infer]  graph/infer.rs
    |    Constraint-based type inference across all functions.
    |    SCC analysis for mutually recursive functions.
    |    Resolves: function signatures, context types, effect footprints.
    |    Output: InferResult { outcomes: Map<QualifiedRef, FnInferOutcome> }
    v
[Phase 2: Lower]  graph/lower.rs
    |    Typed AST → flat MIR instructions (pre-SSA).
    |    Context mutations → ContextProject/ContextStore pairs.
    |    Pattern matching → TestXxx + JumpIf chains.
    |    Output: Map<QualifiedRef, MirModule>
    v
[Phase 3: Optimize]  graph/optimize.rs
    |
    |  ┌─── Pass 1 (cross-module) ───────────────────────────────┐
    |  │  For each module:                                        │
    |  │    promote(MirBody) → CfgBody                           │
    |  │    SSA pass  (context forwarding, dead store elim)       │
    |  │    demote(CfgBody) → MirBody                            │
    |  │  Then:                                                   │
    |  │    Inline (cross-module, devirtualization)               │
    |  └──────────────────────────────────────────────────────────┘
    |
    |  ┌─── Pass 2 (per-module) ─────────────────────────────────┐
    |  │  For each module (main + closures):                      │
    |  │    promote(MirBody) → CfgBody                           │
    |  │    SpawnSplit   IO FunctionCall → Spawn + Eval           │
    |  │    CodeMotion   Hoist Spawn above branches               │
    |  │    Reorder      Spawn early, Eval late within blocks     │
    |  │    SSA pass     Re-normalize after transforms            │
    |  │    RegColor     SSA-aware greedy register coloring       │
    |  │    demote(CfgBody) → MirBody                            │
    |  │  Then:                                                   │
    |  │    Validate (type check + move check on final MirBody)  │
    |  └──────────────────────────────────────────────────────────┘
    |
    v
Map<QualifiedRef, MirModule>  (optimized, validated)
    |
    v
[Interpreter]  acvus-interpreter
    Executable::Module(MirModule) | Builtin(handler) | Extern(handler)
    Register-based VM, SequentialExecutor (async executor planned)
```

---

## Pass 2 Pipeline — Detail

Each pass operates on `&mut CfgBody` with access to `&FxHashMap<QualifiedRef, Ty>` (fn_types).

### 1. SpawnSplit (`optimize/spawn_split.rs`)

**Input:** CfgBody with `FunctionCall` instructions.
**Output:** IO FunctionCalls replaced with `Spawn` + `Eval` pairs.

```
BEFORE:  r1 = call fetch(r0)  use(@ctx=r2)  def(@out=r3)
AFTER:   handle = spawn fetch(r0)  use(@ctx=r2)
         r1 = eval handle  def(@out=r3)
```

- Only splits `Callee::Direct` with `io: true` in EffectSet.
- `context_uses` → Spawn (callee reads at spawn time).
- `context_defs` → Eval (callee writes committed at eval time).
- Handle type registered: `Ty::Handle(ret_ty, effect)`.
- Pure calls and indirect calls pass through unchanged.

### 2. CodeMotion (`optimize/code_motion.rs`)

**Input:** CfgBody with Spawn/Eval and pure instructions.
**Output:** Pure instructions (especially Spawn) hoisted to dominator ancestors.

Algorithm:
1. Build dominator tree + token liveness.
2. For each hoistable instruction, walk UP the dominator chain to find the **highest ancestor** where all operands are available and no token conflicts exist.
3. `def_block` updated after each decision → operand chains resolved in one pass.
4. Fixpoint loop (typically 1 iteration).

**Hoistability (allowlist, default deny):**
- Hoistable: arithmetic, value construction, field access, test predicates, `Spawn` (Direct only), `LoadFunction`, pure `FunctionCall` (where `EffectSet::is_pure()` = true).
- NOT hoistable: `Eval`, context ops, variable ops, indirect calls.

### 3. Reorder (`optimize/reorder.rs`)

**Input:** CfgBody with instructions in each block.
**Output:** Instructions reordered within each block by dependency + priority.

Three dependency chains:
1. **SSA use-def** — use must follow def.
2. **Token ordering** — same TokenId preserves original order.
3. **ContextStore ordering** — stores to same context preserve original order.

Priority (topological sort with BinaryHeap):
- `Spawn` → priority 0 (schedule earliest).
- `Scheduled(first_use, 0)` → Eval placed just before first use of its result.
- `Scheduled(first_use, 1)` → normal instructions by first-use position.

### 4. SSA Pass (`optimize/ssa_pass.rs`)

**Input:** CfgBody (possibly post-reorder).
**Output:** SSA-normalized CfgBody.

- `VarLoad`/`VarStore` → block parameters (variables disappear).
- `ContextStore` inside branches → write-back at merge point.
- Dominator-tree DFS scoped context forwarding (replaces `ContextLoad` with forwarded values when possible).
- Trivial PHI elimination (all incoming edges provide same value → replace PHI with that value).
- Chained substitution: `var_subst ∘ fwd_subst`.

### 5. RegColor (`optimize/reg_color.rs`)

**Input:** CfgBody in SSA form.
**Output:** ValueIds compacted — non-overlapping lifetimes share slots.

- SSA-aware set-based greedy coloring (not interval-based linear scan).
- Backward dataflow liveness over CFG.
- `compute_coloring()` → `apply_coloring()` separation.
- Kill order: color defs → kill dying uses → kill dead defs.
- Entry params/captures colored with shared `entry_live` set.

---

## Analysis Infrastructure

All analyses operate on `&CfgBody`.

### Dataflow Framework (`analysis/dataflow.rs`)

Generic forward/backward dataflow engine.

```rust
trait DataflowAnalysis {
    type Key;                        // What we track (ValueId, TokenId, ...)
    type Domain: SemiLattice;        // Lattice element (Liveness, TokenSet, ...)

    fn transfer_inst(&self, inst, state);       // Per-instruction transfer
    fn terminator_uses(&self, term, state);     // Terminator read effects
    fn terminator_defs(&self, term, state);     // Terminator write effects (ListStep)
    fn propagate_forward(&self, ...);           // Source exit → target entry
    fn propagate_backward(&self, ...);          // Successor entry → block exit
}
```

Output: `DataflowResult { block_entry, block_exit }` per block.

### Dominator Tree (`analysis/domtree.rs`)

Cooper-Harvey-Kennedy algorithm. `DomTree::build(&CfgBody)`.

- `idom(block) → Option<BlockIdx>` — immediate dominator.
- `dominates(a, b) → bool` — does `a` dominate `b`?
- `depth(block) → usize` — depth in dominator tree.
- Used by: CodeMotion (hoist target), SSA pass (context forwarding scope).

### Liveness (`analysis/liveness.rs`)

Backward dataflow: which ValueIds are live at each block entry/exit.

- `analyze(&CfgBody) → LivenessResult`.
- `is_live_in(block, val)`, `is_live_out(block, val)`.
- Used by: RegColor (interference detection).

### Token Liveness (`analysis/token_liveness.rs`)

Backward dataflow: which TokenIds are live at each block entry/exit.

- `analyze(&CfgBody, &fn_types) → TokenLivenessResult`.
- `is_live_in(block, token)`, `is_live_out(block, token)`.
- Used by: CodeMotion (token conflict detection prevents hoisting past concurrent token use).

### Reachable Context (`analysis/reachable_context.rs`)

Classifies context loads as eager/lazy/pruned by analyzing which branches are reachable given known context values.

- Two-pass: ValueDomainTransfer → Reach BFS using `cfg.successors()`.
- Used by: orchestration (determines which contexts to fetch eagerly vs lazily).

### Val Def (`analysis/val_def.rs`)

Maps each ValueId to the instruction index that defines it.

- `build(&MirModule) → ValDefMap`.
- Used by: extract (projection chain tracing).

### Inst Info (`analysis/inst_info.rs`)

Pure utility: `defs(&InstKind) → SmallVec<ValueId>`, `uses(&InstKind) → SmallVec<ValueId>`.

Used by everything — dataflow, liveness, reorder, code_motion, reg_color.

---

## Validation (`validate/`)

Runs on **final MirBody after all optimizations and demote**. Catches optimizer bugs.

### Type Check (`validate/type_check.rs`)

Every instruction's operands match expected types. Arity, constructor shape, materiality (functions can't be stored to context).

### Move Check (`validate/move_check.rs`)

Move-only values (`Handle<T,E>`, self-modifying iterators) consumed exactly once. Use-after-move and unused handles are errors. Uses CfgBody internally (promotes a clone).

---

## IO Parallelization — End-to-End Example

Source: `imports = fetch_a(); types = fetch_b(); refs = fetch_by(imports); checked = refs + types; extra = fetch_c(); checked + extra`

Where `fetch_a`, `fetch_b`, `fetch_c`, `fetch_by` are IO ExternFns.

Optimized MIR:
```
 0 │ r0 = spawn fetch_a()       ← 3-way parallel IO start
 1 │ r1 = spawn fetch_b()       ←
 2 │ r2 = spawn fetch_c()       ←
 3 │ r3 = eval r0               ← imports ready
 4 │ r0 = spawn fetch_by(r3)    ← dependent IO starts immediately
 5 │ r3 = eval r1               ← types ready
 6 │ r4 = eval r0               ← refs ready
 7 │ r5 = r4 + r3               ← checked = refs + types
 8 │ r3 = eval r2               ← extra ready
 9 │ r4 = r5 + r3               ← final result
10 │ return r4
```

Pipeline contribution:
- **SpawnSplit**: 4 FunctionCalls → 4 Spawn+Eval pairs.
- **CodeMotion**: All independent Spawns hoisted above any Eval.
- **Reorder**: Eval placed just before first use of result.
- **RegColor**: 6 virtual registers (r0-r5) reused across non-overlapping lifetimes.

---

# Part II — What Breaks If You Change It

## SSA: Block Parameters vs PHI Nodes

We use Cranelift-style block parameters, not LLVM-style PHI nodes.

```
Jump { label: L0, args: [v1, v2] }
BlockLabel { label: L0, params: [v3, v4] }
```

Block params are definitions (defs), jump args are uses. Consequences of this choice:

- **Inlining is simple.** Remap args and you're done. No PHI surgery required.
- **Terminator args are uses.** This affects liveness, dataflow, and reg_color. Args live inside terminators that are not in the instruction array, so flat instruction indexing misses them. This is why liveness analysis treats terminator position as `block_end + 1`.
- **SSABuilder has a seal ordering requirement.** `seal_block()` may only be called after all predecessors of that block have been added. Calling it earlier resolves PHIs against an incomplete predecessor set, silently losing values.

### ENTRY_BLOCK Sentinel

`ENTRY_BLOCK = Label(u32::MAX)` is a sentinel identifying the implicit entry block. Using `u32::MAX` as an actual label will collide. Labels are allocated via `label_count`, so collision is unrealistic in practice, but not formally guaranteed.

### Loop Back-Edge Cycle Breaking

In `use_var_sealed()`, the PHI value is pre-defined in `current_defs` before resolving it. Without this, a loop back-edge causes: PHI resolve → same PHI use → same PHI resolve → infinite recursion. The pre-definition breaks the cycle.

### Trivial PHI Elimination

If all incoming edges provide the same value (excluding the PHI itself), the PHI is replaced with that value. The self-exclusion filter is critical — a loop back-edge referencing the PHI itself does not count as "all incoming values are identical".

---

## SSA Pass: Context vs Local Variable Asymmetry

The most important design decision: **contexts require write-back, local variables do not.**

Local variables (`VarLoad`/`VarStore`) disappear entirely after SSA. Values flow through block params — nothing else.

Contexts (`ContextProject`/`ContextLoad`/`ContextStore`) are different. Contexts are external state, so stores inside branches must persist after merge. The SSA pass removes branch-internal `ContextStore`s and inserts a single write-back `ContextStore` after each merge block.

### ContextProject/ContextLoad Always Come in Pairs

The lowerer always emits `ContextProject` immediately followed by `ContextLoad`. The SSA pass's dead load elimination removes these pairs using **index-1 arithmetic** — if a `ContextLoad` is dead, the immediately preceding `ContextProject` is removed with it.

**When this breaks:** If someone inserts an instruction between `ContextProject` and `ContextLoad`, the index-1 removal deletes the wrong instruction. The lowerer currently guarantees adjacency, but this invariant is not enforced in code.

### Undef Initialization for Write-Only Contexts

When a context has `ContextStore` but no preceding `ContextLoad` (write-only), the SSA pass splices an `Undef` instruction at position 0. This splice happens after CFG construction but before other optimizations. If the splice timing changes, instruction indices drift and the def_map breaks.

### Chained Substitution

The SSA pass chains two substitution maps: `var_subst` (from SSABuilder) + `fwd_subst` (from forward context values). If `var_subst` maps r10→r5 and `fwd_subst` maps r5→r3, the final result is r10→r3. Reversing the application order produces incomplete forwarding.

---

## Inliner: Name-Based Parameter Binding

The inliner binds parameters **by name**, not by position.

`$x + $x` produces two `ParamLoad { name: "x" }` instructions. The inliner registers the first occurrence in `param_name_to_arg` and reuses the cached arg for subsequent occurrences of the same name.

**Why positional binding breaks:** If the same parameter is used twice, the second occurrence consumes the next arg instead of reusing the first.

### val_remap Chain

When sequentially inlining multiple calls, each inline's result dst is added to `val_remap`. The next inline's args are remapped through the current `val_remap`. In `double(inc(3))`, if inc's result is remapped r5→r10, then double's arg must use r10 instead of r5.

**Ordering matters:** Args must be remapped before inlining. After inlining, new ValueIds are mixed in and remap becomes unreliable.

### Label Offset Accumulation

When inlining, all callee labels are offset by `label_offset = current.label_count`, then `current.label_count += callee.label_count`. This relies on labels being allocated linearly and never reused.

### Devirtualization Conditions

To devirtualize an `Indirect(v)` callee:
1. `v`'s definition must be exactly one `MakeClosure` (traced through def_map)
2. It must not pass through a PHI/block param (which would mean multiple possible definitions)

Devirtualizing through a PHI is unsound — at runtime, the closure could be a different one.

---

## Dataflow: Forward and Backward Are Not Simple Inverses

### Forward: propagate_to_successor

Maps jump args from source block's exit state to target block's params. Additionally joins the entire exit state into the target entry (non-param values flow through).

### Backward: propagate_from_successor

If successor params are live at successor's entry, marks the corresponding terminator args as live at this block's exit. **Additionally** joins non-param values from successor's entry into this block's exit.

**Why the asymmetry:** In backward analysis, values live at a successor's entry that bypass params (= values live across block boundaries without going through block params) must also be live at this block's exit. In forward analysis, such values propagate naturally through the join. In backward, this flow-through must be explicit.

### JumpIf Cond Is Marked Live in Backward

The JumpIf terminator itself uses the cond value. In backward analysis, cond is set to `D::top()` in the block's exit state. Forward analysis doesn't need this — forward tracks "what is true at this point", not "what is used".

### Return(ValueId)

`Terminator::Return(val)` sets val to `D::top()`. Previously, `Terminator::Return` did not carry a ValueId, so the return value was not marked live in backward analysis — this was a bug and has been fixed.

### ListStep's Dual Role

`ListStep` is a terminator that also **defines values** (dst, index_dst). Other terminators (Jump, JumpIf, Return) don't define values. This asymmetry requires special handling in every analysis that tracks definitions: dataflow (`terminator_defs`), code_motion (`build_def_block`, `is_terminator_def`), liveness.

---

## CFG: Block Boundary Rules

The CFG is built from a flat instruction stream by `promote()`. Rules:

1. `BlockLabel` → starts a new block (flushes the previous one)
2. `Jump`/`JumpIf`/`Return` → ends a block (becomes terminator)
3. `ListStep` → ends a block + defines values + two successors (body fallthrough + done)

**Fallthrough successor is `BlockIdx(current + 1)`.** This assumes the block array is sequential. Reordering blocks breaks fallthrough.

**Entry block is always `BlockIdx(0)`.** Instructions before the first `BlockLabel` form the implicit entry block.

---

## Reorder: Three Independent Dependency Chains

The reorder pass builds three dependency chains simultaneously within each basic block:

1. **SSA use-def** — A use must come after the instruction that defines its ValueId
2. **Token ordering** — Instructions sharing the same TokenId must preserve original order
3. **ContextStore ordering** — Stores to the same context must preserve original order

If the three chains conflict, a cycle can occur. The current implementation catches cycles with an assert but has no graceful recovery.

### ContextStore → ContextProject Back-Trace

A ContextStore's dst is the result ValueId of a ContextProject. The reorder pass traces this ValueId through def_map to determine which context the store targets. **The ContextProject must be in the same block** — if it's in a different block, def_map lookup fails and ordering is lost.

### Priority-Based Topological Sort

Uses a `BinaryHeap` (min-heap via Reverse). Spawn = priority 0 (earliest). Eval = `Scheduled(first_use, 0)` (just before first use). Normal = `Scheduled(first_use, 1)`.

---

## Spawn Split: Preconditions

Spawn split must run after the SSA pass. Reason: `FunctionCall`'s `context_uses`/`context_defs` are populated by the SSA pass. Splitting before SSA produces Spawn/Eval with empty context fields, severing context flow.

### Handle Type Registration

On split, a Handle type is registered in `val_types`. The callee's return type and effect are extracted from `fn_types` to construct `Ty::Handle(ret, effect)`. **If the callee is missing from fn_types, the Handle type is not registered**, and downstream type checking fails.

### Only IO Is Split

`is_io_call()` checks only the `io` flag in the effect. Token-only effects are not split. This is a deliberate decision: Token functions must execute sequentially, so there's no benefit in splitting them for reorder.

---

## Code Motion: Soundness by Construction

The hoistability check uses an **allowlist** (default deny). Only provably pure instructions are hoisted. Unknown/new instruction kinds are automatically blocked. This means adding a new InstKind never silently becomes hoistable — you must explicitly opt in.

### Token Conflict Detection

Before hoisting to a candidate block, `token_ids_of` extracts the instruction's tokens from fn_types. If any token is `live_out` at the candidate, hoisting is blocked — another Spawn with the same token is already pending.

**Only `Callee::Direct` Spawns are hoisted.** Indirect Spawns can't be looked up in fn_types, so their tokens are unknown. Hoisting them could violate token ordering.

### `is_pure_call` Uses `EffectSet::is_pure()`

Checks all four fields: `!io && reads.is_empty() && writes.is_empty() && !self_modifying`. A self-modifying function (iterator cursor advance) would change semantics if hoisted.

---

## Typeck: param_types Ordering Matters

`param_types: SmallVec<[(Astr, Ty); 4]>` preserves insertion order. It was previously an `FxHashMap`, but HashMap's non-deterministic iteration order caused extern function parameter ordering to break.

**Why this matters:** `extern_params` iterates `param_types` in order to construct `Signature.params`. The caller places args in this order. If the order drifts, arg-param binding becomes incorrect.

### Lambda Effects Do Not Propagate Upward

When an effectful function is called inside a lambda, the effect is recorded in the lambda's effect — not the enclosing function's effect. The lambda scope freezes the effect on pop. This is deliberate: a lambda produces effects at call site, not at definition site. The enclosing function only needs to know "calling this lambda has effects".

---

## Move Check: VarStore Revives Values

After a move-only value is moved, storing a new value to the same variable (`VarStore`) brings it back to `Alive`.

```
a = move_only_value;  // a: Alive
consume(a);           // a: Moved
a = new_value;        // a: Alive again
use(a);               // OK
```

**`VarLoad`/`ParamLoad` do not revive.** A load is a read (and consumption for move-only types), not a new definition.

### Conservative Join at Branch Merge

If one branch has `Alive` and the other has `Moved(at: 3)`, the merge result is `Moved(at: 3)`. This is conservative — if any path could have moved the value, it's treated as moved.

---

## Reachable Context: Branch Pruning with Known Values

`partition_context_keys` classifies context loads as eager/lazy/pruned. The key mechanism: known context values (provided by the caller) are used to evaluate `TestLiteral` conditions and prune dead branches.

**Pruned contexts still require type injection.** Code in dead branches has already passed type checking, so missing type information causes the type checker to panic. The runtime doesn't need to resolve these values, but types must still be injected.

---

## Register Coloring: SSA-Aware Greedy

The current implementation uses **set-based greedy coloring** over CFG-aware liveness, not interval-based linear scan.

- `Coloring` struct: `assign()`, `color_of()`, `is_colored()`, `is_improvement()`.
- `LiveColors` struct: tracks which colors are currently live at a program point.
- `LastUseMap` struct: precomputes where each value's last use is within a block.
- Kill order within an instruction: color defs → kill dying uses → kill dead defs.
- Entry params/captures: colored with a shared `entry_live` set to prevent conflicts.

**Why not interval-based linear scan?** SSA produces a chordal interference graph, where greedy coloring on a perfect elimination ordering is optimal. The set-based approach naturally handles cross-block liveness without constructing intervals.

---

## "Change This, Break That"

| Change | Consequence |
|--------|-------------|
| Revert `param_types` to HashMap | Extern function parameter order becomes non-deterministic → arg-param binding errors |
| Insert instruction between `ContextProject`/`ContextLoad` | SSA pass's index-1 dead load elimination removes the wrong instruction |
| Remove pre-definition in SSABuilder | Infinite recursion on loop back-edge PHI resolution |
| Run spawn split before SSA | context_uses/context_defs are empty → context flow severed |
| Remove ValueId from Terminator::Return | Return value not marked live in backward analysis → reg_color reuses the return slot |
| Remove VarStore revive logic | Reassignment after move is falsely flagged as use-after-move |
| Allow cross-block ContextProject in reorder back-trace | def_map miss → ordering lost → stores to same context reordered → unsound |
| Propagate lambda effects to enclosing scope | Lambda definition alone marks outer function as effectful → unnecessary inline restrictions |
| Remove ListStep terminator_defs handling | ListStep's dst/index_dst missing from analysis → code_motion hoists past them → value corruption |
| Remove non-param flow-through in backward propagation | Cross-block live values disappear from exit state → reg_color reuses their slots → value corruption |
| Reorder block array | Fallthrough successor idx+1 calculation breaks → wrong successor visited |
| Use `ENTRY_BLOCK` sentinel as a real label | SSA builder confuses entry block with regular block |
| Remove self-reference filter from trivial PHI elimination | Loop back-edge self-reference triggers "all incoming identical" → PHI incorrectly eliminated |
| Hoist Indirect Spawn in CodeMotion | token_ids_of returns empty for Indirect → token conflicts undetected → ordering violation |
| Remove `self_modifying` check from `is_pure_call` | Iterator-advancing function hoisted above branch → mutation happens unconditionally |
| Make collection elements covariant | `List<Deque<T>>` accepts `List<List<T>>` silently → runtime type confusion |
| Remove Identity from Deque | Deques from unrelated sources silently merge → data corruption at API boundaries |
| Allow LUB in invariant position | Branch merge produces widened type that neither branch actually provides → unsound |
| Remove ParamConstraint intersection | Two constrained Params unify without checking compatibility → bind to impossible type |
| Remove occurs check from Param unification | `T = List<T>` creates infinite type → resolver loops forever |
| Change ExternCast to accept multiple matches | Ambiguous coercion silently picks one → non-deterministic type resolution |
| Skip effect propagation after SCC inference | Transitive effects lost → function marked pure when it calls IO → SpawnSplit misses it |
| Resolve types within SCC before all functions type-checked | Mutual recursion: early resolution freezes types before constraints from later functions arrive |

---

## What We Don't Guarantee

- **ListStep only works on lists.** It is not a general-purpose iterator step. UserDefined iterators require a separate mechanism.
- **No speculative execution.** IO inside branches is not spawned speculatively (spawn split only works on IO calls, CodeMotion only hoists pure instructions including Spawn).
- **Plugin effects are not verified.** If a plugin declares pure but performs IO, the system cannot catch this.
- **Integer overflow, out-of-bounds access, and division by zero are not caught at compile time.**
- **Termination is not guaranteed.** Recursive functions and infinite loops are permitted.
- **Undef instructions from SSA are not eliminated.** They are harmless placeholders that will be removed in future bytecode lowering.
