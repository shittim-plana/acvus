# acvus MIR Architecture

## What is this?

acvus is not a general-purpose programming language. It is not trying to replace Rust, C, Java, or any systems/application language. It is a **script and template language** — the kind you embed in a larger system to let users write business logic, data transformations, and LLM orchestration without touching the host codebase.

The difference is that acvus takes this role seriously. Instead of a loosely-typed interpreter with string-based extension points (like most embedded scripting), it has a full compiler backend with type inference, effect tracking, and SSA-based IR. The goal is simple: **scripts and templates should be safe, fast, and pleasant to write** — with the same rigor that systems languages apply to systems code.

A single expression like `@users | filter(active) | map(name) | join(", ")` compiles to the same typed, optimized IR whether it appears inside `{{ }}` template interpolation or as a standalone script. The MIR knows not just *what* your code computes, but *which external state it reads and writes*, enabling automatic parallelization, incremental recompilation, and portable execution.

---

## Design Principles

**One IR, multiple surfaces.** Template and script share the same type system, the same compiler pipeline, the same IR. No special-casing.

**Effects are first-class.** Every function's effect footprint (which contexts it reads, writes, whether it does IO) is tracked at the type level. This isn't an annotation — it's inferred from code and propagated transitively through call chains.

**No opaque IDs where names suffice.** Contexts are identified by qualified names (`@namespace:name`), not opaque integer IDs. This makes the IR deterministic, serializable, and human-readable without a symbol table.

**Plugin-extensible type system.** External functions register their types alongside their handlers. Adding an ext function expands the set of valid programs — the type system grows with the ecosystem.

**Use first, define later.** Most things in acvus are inferred from usage, not declared upfront. Write `@data | map(f) | collect` and the compiler works *backwards* — `@data` must be iterable, `f` must return something, the result is a list of that something. Context types, function parameter types, effect footprints, even generic constraints are all discovered by analyzing how values are used, then propagated outward to the environment. The host system provides concrete types for contexts; the compiler checks that they satisfy the constraints the code imposed. This inverts the traditional "define type, then use" flow — users write code freely, and the system figures out what the environment must provide.

---

## The Pipeline

```
source → extract → infer → resolve → lower → SSA → validate → MirModule
```

Each phase has a single responsibility and a clear contract with the next:

**Extract** parses source and discovers what external state (contexts) the code references, distinguishing reads from writes by tracing projection chains through the IR.

**Infer** is where "use first, define later" happens. The code references contexts (`@data`, `@config`) that may not have known types yet. Instead of erroring, the inferencer assigns type variables and propagates constraints from usage. If `@data | map(f)` appears, the system concludes `@data` must be iterable. If `@config.threshold + 1` appears, `@config` must have an integer `threshold` field. These constraints flow outward — the host system must eventually provide types that satisfy them. Inter-function inference uses Tarjan's SCC algorithm so mutually recursive functions are solved simultaneously.

**Resolve** finalizes all types with no remaining variables. Every expression has a concrete type. Every effect is resolved. This is the point of no return — if types don't work out, errors are reported here.

**Lower** translates the typed AST to MIR instructions in SSA form. Context variables (which can be mutated across branches and loops) are promoted to SSA via a Cranelift-style mem2reg pass that inserts PHI nodes at merge points.

**Validate** runs two passes: type checking (operand types match instruction contracts) and move checking (move-only values like IO handles are consumed exactly once).

---

## Type System Highlights

### Structural Types, No Declarations

Objects are structurally typed. `{ name: "Alice", age: 30 }` has type `{ name: String, age: Int }`. A function that accesses `.name` accepts any object with a `name` field — width subtyping, inferred automatically.

Enums are open — writing `Color::Red` and `Color::Blue` in different places unifies into `Enum { Color: { Red, Blue } }`. No upfront declaration needed.

### Constraint-Based Generics Without Traits

A function's parameter constraints are inferred from how the parameter is used:

```
fn process(x) { x | to_string | length }
```

The compiler infers: `x` must support `to_string` (constraining it to `{ Int, Float, String, Bool, ... }`) and the result must support `length`. No generic syntax, no trait bounds, no `where` clauses.

When two constrained types unify, their constraint sets are **intersected**. This is sound and complete for the closed builtin set, and extends naturally when plugins register new functions.

### Origin Tracking for Deques

Deques carry an *origin* tag that prevents accidentally mixing data from different sources. `append(@history, item)` preserves the origin — you can't accidentally `extend` one deque with another's data without explicit coercion. This is tracked at the type level and enforced at compile time.

### Six Explicit Coercions

Subtype conversions are never implicit in the IR. The type checker identifies where coercion is needed and the lowerer inserts explicit `Cast` instructions: `Deque→List`, `List→Iterator`, `Deque→Iterator`, `Deque→Sequence`, `Sequence→Iterator`, `Range→Iterator`. Every conversion is visible and auditable.

---

## Effect System

Every function has an `EffectSet`:

```
reads:          which contexts are read
writes:         which contexts are written
io:             whether opaque IO occurs
self_modifying: whether internal state changes (e.g., iterator cursor advance)
```

Effects **propagate transitively** — if `f` calls `g` which reads `@users`, then `f` also reads `@users`. This is computed automatically during inference.

**Why this matters:**

1. **Automatic parallelization.** Two functions that read different contexts and write nothing can execute in parallel. The rescheduler (planned) uses effect analysis to find independent IO operations and convert sequential code to `spawn`/`eval` pairs.

2. **Incremental recompilation.** When source changes, the compiler checks if the function's *type and effect signature* changed. If not, downstream dependents don't need recompilation (early cutoff).

3. **Scheduling.** The orchestration layer uses effects to determine execution order. Read-read is safe to parallelize. Write-read must be serialized. This is decided at compile time, not runtime.

4. **Move semantics.** An effectful iterator is move-only — you can't clone it because advancing the cursor is a side effect. The move checker enforces this statically.

---

## SSA Form for Mutable Context

Context variables (`@name`) can be mutated in branches and loops:

```
x in @items {
    @sum = @sum + x;
    @product = @product * x;
};
```

The MIR uses a projection model (similar to LLVM's alloca + mem2reg):

1. `ContextProject` creates a reference to a context slot
2. `ContextStore` writes through the reference
3. The SSA pass promotes these to PHI nodes at loop headers and branch merge points
4. After SSA, context mutations become pure value flow — no mutable state in the IR

This means the MIR is genuinely functional after SSA — all "mutation" is expressed as new values flowing through block parameters.

---

## Concurrency Model

```
Spawn { dst, callee, args, context_uses }   // Pure: schedule work, get Handle
Eval  { dst, src, context_defs }            // Effectful: force Handle, get result
```

`Spawn` is pure — it creates a `Handle<T, E>` without executing anything. `Eval` is where effects actually happen. This separation means:

- Multiple spawns can be issued before any eval (pipelining)
- The runtime decides whether to execute sequentially or in parallel
- Handles are move-only — they must be evaluated exactly once (enforced by move checker)
- **Deadlock is impossible** — the effect system ensures no circular dependencies between spawned computations

---

## Iterator System

Iterators are lazy and SSA-clean:

```
IterStep { dst, iter_src, iter_dst, done, done_args }
```

One instruction pulls an element, produces the rest iterator as a new value (SSA rebinding), and branches to `done` if exhausted. This is a proper CFG terminator with two successors — the SSA pass handles it like any other branch.

Effectful iterators (e.g., from regex `find_all`) carry `self_modifying` in their effect, making them move-only. Pure iterators (from list conversion) are freely copyable.

The `pmap` builtin marks chunks for parallel execution. On native targets with LLVM, pure `pmap` over scalars could theoretically compile to SIMD.

---

## Incremental Compilation

The pipeline is split into four phases (extract → infer → resolve → lower) not because it's architecturally elegant, but because **the LSP needs to cut into the middle**. When a user edits a script in their editor, the system re-runs only the phases that are invalidated — not the entire compilation. This is the primary reason for the phase separation.

The `IncrementalGraph` manages caching at each phase boundary:

- **Extract cache**: keyed by source hash. Same source = skip re-parsing.
- **Infer cache**: per-SCC. If a function's inferred type didn't change, don't re-infer its callers (**early cutoff**).
- **Resolve cache**: per-function. Invalid only when infer results change.

On source edit: re-extract the changed function → check if its call edges changed → if SCC structure unchanged, only re-infer the affected SCC and propagate dirty marks forward. Most edits touch one SCC, so recompilation is O(changed) not O(total).

The LSP is a thin wrapper over `IncrementalGraph` — it maps document IDs to functions and delegates everything else. **LSP diagnostics and build errors are identical because they run the exact same pipeline, the exact same code, the exact same `IncrementalGraph`.** There is no separate "LSP mode" or "analysis mode" that could diverge from the real compiler.

---

## Plugin System (ExternFn)

External functions are registered with their type signature, effect classification, and runtime handler in a single declaration:

```rust
ExternFn::build("format_date")
    .params(vec![opaque_ty(), Ty::String])
    .ret(Ty::String)
    .pure()
    .sync_handler(|args, _| { /* implementation */ })
```

This is simultaneously:
- A **type declaration** (the compiler knows the signature)
- An **effect declaration** (`.pure()` / `.io()`)
- A **runtime handler** (the interpreter knows how to call it)

Adding a plugin **extends the type system**. A new function that accepts `Ty::Param` with constraints expands the set of valid generic programs. The type checker doesn't need to be modified — it just sees a wider function environment.

---

## Validation

Two complementary passes ensure MIR soundness:

**Type checking** validates every instruction's operands match expected types. A `BinOp(+)` with `String` left and `Int` right is caught here, not at runtime.

**Move checking** tracks liveness of move-only values. `Handle<T, E>` must be consumed by exactly one `Eval`. Using it twice, or not at all, is a compile-time error. This extends transitively — a closure capturing a move-only value becomes move-only itself (FnOnce semantics).

---

## Analysis Infrastructure

The MIR ships with a generic dataflow framework (forward analysis with semilattice + transfer functions) and an abstract value domain supporting:

- **Interval analysis** for integers (with graduated widening)
- **Constant propagation** for all scalar types
- **Branch elimination** when conditions are statically determined
- **Variant tracking** for enum tag refinement after pattern matching

These analyses power the **reachable context analysis**, which classifies each context load as *eager* (always needed), *lazy* (conditionally needed), or *pruned* (dead code). This informs the runtime about which contexts to prefetch.

---

## What This Enables

**Write sequentially, execute in parallel.** Users write straightforward sequential code. The effect system identifies independent operations. The rescheduler (planned) automatically introduces parallelism where safe.

**One source, multiple targets.** The same MIR can be interpreted (development), compiled to wasm (browser), or lowered to native (production). Full SSA with alias tracking means LLVM-quality optimization is theoretically achievable.

**Portable computation.** Since all mutable state is explicit (contexts) and effects are tracked, a computation can be serialized mid-execution and resumed on a different machine. Context snapshot + MirModule = portable.

**Type-safe plugin ecosystem.** External functions participate in type inference and effect tracking on equal footing with builtins. The type system grows with the ecosystem, not against it.

---

## Key Invariants

1. **Types are complete.** No unresolved type variables survive past resolve. Every value in lowered MIR has a concrete type.
2. **Effects are sound.** If a function is marked pure, it truly has no side effects. The system never over-promises.
3. **Moves are checked.** Move-only values are consumed exactly once. Use-after-move is a compile error.
4. **Output is deterministic.** Same source always produces the same MIR, regardless of hash map ordering or global state.
5. **Single source of truth.** Value types live in `val_types`, not duplicated in instruction fields. Context identity is the qualified name, not an opaque ID.
