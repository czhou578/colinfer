# Code Slop Review

Review a codebase for signs of low-quality, AI-assisted or "vibe-coded" implementation. Do **not** attempt to determine authorship or claim that code was AI-generated. Evaluate the code based on observable engineering practices, maintainability, correctness, and architectural coherence.

## Review principles

* Prefer concrete evidence over impressions.
* Distinguish genuine problems from intentional design.
* Do not criticize abstraction merely because it is abstract; identify whether it has a concrete purpose.
* Prioritize systemic patterns over isolated mistakes.
* Trace important behavior end-to-end before judging individual components.
* Check the repository's existing conventions before recommending changes.
* Prefer the smallest architectural change that resolves the problem.
* Do not recommend rewrites unless the existing structure genuinely prevents maintainability or correctness.

## Primary smells to detect

### 1. Unnecessary abstraction

Look for:

* Interfaces with only one implementation and no meaningful substitution boundary.
* Factories that only instantiate one concrete type.
* Managers/services/adapters wrapping trivial operations.
* Multiple layers that simply forward calls.
* "Future-proofing" without a current requirement.

Ask:

> What concrete problem does this abstraction solve?

### 2. Duplicated concepts and logic

Look for:

* Multiple implementations of the same business rule.
* Repeated normalization, validation, formatting, or transformation logic.
* Multiple types representing the same conceptual object.
* Parallel helper functions with slightly different behavior.
* Code that reimplements existing repository functionality instead of reusing it.

### 3. Inconsistent conventions

Look for different approaches to the same problem:

* `throw` vs `null` vs `Result` vs `{ success: false }`.
* Different HTTP clients.
* Different validation strategies.
* Different date/time representations.
* Different error/logging conventions.
* Different persistence patterns.

Inconsistency across equivalent code paths is more significant than any individual style preference.

### 4. Excessive defensive programming

Look for:

* Repeated null/undefined checks where invariants already guarantee validity.
* Deep optional chaining masking invalid state.
* Broad fallback behavior.
* Returning `null`/`undefined` instead of surfacing unexpected failures.
* "Robustness" that makes invalid states silently propagate.

### 5. Catch-all error handling

Flag:

* `catch (e) { return null }`
* `except Exception`
* `except: pass`
* Logging an error and continuing without a defined recovery strategy.
* Errors being converted into ambiguous values.

Determine whether the code distinguishes expected failures from programmer/system errors.

### 6. Overly large or multi-purpose modules

Look for files/functions responsible for unrelated concerns such as:

* API handling
* business logic
* persistence
* authentication
* external integrations
* formatting
* logging

Do not flag size alone. Identify conflicting responsibilities and difficult ownership boundaries.

### 7. Junk-drawer utilities

Look for `utils`, `helpers`, or similar modules containing unrelated functionality.

Prefer domain-specific ownership when functions have different responsibilities or consumers.

### 8. Implementation-focused tests

Look for tests dominated by:

* Mock call assertions.
* Internal method verification.
* Assertions that a particular helper was called.
* Tests that reproduce implementation structure rather than externally observable behavior.

Check whether important behavior, failure modes, integration boundaries, and invariants are actually tested.

### 9. Fake test coverage

Look for:

* Large numbers of trivial tests.
* Tests for obvious null/empty cases while important scenarios are absent.
* High apparent coverage with little behavioral coverage.
* Excessive mocking that prevents real components from interacting.
* Tests that would continue passing after the behavior they supposedly protect was broken.

### 10. Type-system avoidance

For typed languages, look for:

* Excessive `any`.
* Unnecessary casts.
* `as unknown as X`.
* Non-null assertions.
* Weak generic types.
* Repeated optional fields used to represent fundamentally different states.
* Runtime uncertainty being pushed through the application instead of validated at boundaries.

### 11. Narrating comments

Flag comments that merely restate code:

```ts
// Increment counter
counter++;
```

or function documentation that simply repeats the signature.

Prefer comments explaining:

* Why something is necessary.
* Non-obvious invariants.
* External constraints.
* Performance considerations.
* Compatibility behavior.
* Reasons an unusual implementation is intentional.

### 12. Meaningless naming

Look for names such as:

```text
processData
handleData
processResult
performOperation
executeTask
manageState
updateInformation
```

when the domain provides more precise terminology.

Names should communicate the actual operation and domain concept.

### 13. Gratuitous async/concurrency patterns

Look for:

* `async` functions with no asynchronous work.
* Artificial `Promise.resolve`.
* Sequential `await`s for independent operations.
* Unbounded concurrency.
* Missing cancellation/timeouts.
* Per-request creation of expensive clients/resources.

Evaluate the actual execution model rather than applying a blanket preference for synchronous or asynchronous code.

### 14. Resource-lifecycle mistakes

Check:

* Database connections.
* Files.
* HTTP clients.
* Streams.
* Timers.
* Workers/tasks.
* Locks.
* Transactions.

Look for missing cleanup, incorrect ownership, excessive creation/destruction, and unclear lifetimes.

### 15. Database and performance smells

Look for:

* N+1 queries.
* Repeated database access inside loops.
* Unnecessary serialization/deserialization.
* Loading entire datasets when only subsets are needed.
* Missing pagination.
* Repeated expensive computation.
* Obvious sequential I/O.

Do not optimize speculative micro-performance. Focus on measurable or structurally obvious costs.

### 16. Magic numbers and unexplained policy

Flag unexplained values such as:

```ts
retryCount > 3
sleep(5000)
tokens > 8192
timeout = 30000
```

Determine whether the value represents:

* A protocol constraint.
* A product requirement.
* An empirical limit.
* An arbitrary default.

The problem is not the number itself; it is unexplained policy.

### 17. Excessive configuration

Look for configuration options that:

* Are never changed.
* Duplicate constants.
* Exist only for hypothetical future requirements.
* Create combinations the system cannot meaningfully support.

Avoid configurable-everything designs.

### 18. Mini-framework syndrome

Look for custom infrastructure unnecessarily recreating established mechanisms:

```text
CustomEventBus
CustomRepositoryFramework
CustomValidationFramework
CustomHTTPClient
CustomDependencyContainer
CustomStateManager
```

Determine whether the project actually needs the abstraction or whether it adds complexity without providing meaningful capabilities.

### 19. Poor architectural "center of gravity"

Determine whether the codebase has clear ownership for:

* Business/domain logic.
* Persistence.
* External integrations.
* Configuration.
* Validation.
* Application orchestration.
* Shared types.

Flag circular dependencies and situations where equivalent responsibilities are scattered across unrelated modules.

### 20. Difficult change propagation

Choose a representative feature and trace it end-to-end.

Ask:

> If this requirement changes, where must I modify the code?

Flag cases where one conceptual change requires modifying many duplicated representations, wrappers, transformations, or special cases.

### 21. Excessive future-proofing

Be suspicious of comments or abstractions justified primarily by:

> "This will make it easy to add X later."

Require a concrete present-day requirement or architectural constraint before accepting the complexity.

## Deep review procedure

For a meaningful review:

1. Inspect repository structure and existing conventions.
2. Identify the main architectural boundaries.
3. Pick one or two representative features.
4. Trace them end-to-end.
5. Inspect dependencies and data flow.
6. Search for duplicate implementations of important concepts.
7. Inspect error handling and type boundaries.
8. Inspect tests for behavioral coverage.
9. Check resource and database lifecycles.
10. Identify systemic inconsistencies.
11. Separate intentional complexity from accidental complexity.
12. Report only findings supported by concrete evidence.

Use language-server capabilities where available to understand:

* Definitions.
* References.
* Implementations.
* Types.
* Call sites.
* Diagnostics.
* Symbol relationships.

Use text search for complementary evidence, not as a substitute for semantic understanding.

## Severity

Classify findings:

**Critical**

* Correctness or data-integrity risk.
* Security boundary failure.
* Severe resource leak.
* Broken concurrency/transaction behavior.

**High**

* Significant architectural inconsistency.
* Silent error handling that can hide failures.
* Major duplication of business logic.
* Serious performance problem such as an N+1 query.
* Type-system bypass creating meaningful runtime risk.

**Medium**

* Unnecessary abstraction.
* Poor module boundaries.
* Weak tests.
* Excessive configuration.
* Repeated defensive code.
* Maintainability problems.

**Low**

* Naming.
* Redundant comments.
* Minor duplication.
* Local style inconsistencies.

Do not inflate severity merely because code is ugly.

## Finding format

For every finding, report:

```text
[SEVERITY] Short title

Evidence:
<file/path and relevant symbol or line range>

Problem:
<what is concretely wrong>

Why it matters:
<correctness, maintainability, performance, or architectural consequence>

Recommendation:
<smallest reasonable fix>
```

Group related instances into one systemic finding rather than reporting every occurrence individually.

## Final assessment

End with:

### Systemic patterns

List the 3–7 most important recurring patterns.

### Architecture

Describe whether responsibilities and dependencies have clear ownership.

### Testing

Describe whether tests provide meaningful behavioral confidence.

### Maintainability

Describe the main sources of accidental complexity.

### Refactoring priorities

Provide an ordered list of concrete improvements based on engineering impact.

Do **not** provide an "AI-generated score," "slop percentage," or authorship conclusion. The goal is to identify engineering problems, not determine who or what produced the code.
