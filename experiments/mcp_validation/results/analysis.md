# MCP Validation Experiments — Unit Test Results

Date: 2026-02-09

## Overall: 73/74 (99%)

## Experiment Results Summary

### Exp1: Verified Info Cache (16/17)

**Problem confirmed: YES**

| Finding | Evidence |
|---|---|
| pool.py functions work correctly | 14/15 unit tests pass |
| Worker imports but NEVER calls verified info functions | `add_verified_info`, `get_verified_info`, `is_topic_verified` imported but 0 calls in worker logic |
| Worker has NO custom MCP tools | Only standard tools: Read, Glob, Grep, WebSearch, etc. |
| Prompts ASK workers to use verified info | Both EXPLORE and IMPLEMENT prompts mention it |
| Minor bug: `(none yet)` in HTML comment not removed | Non-blocking |

**Design gap**: Complete. pool.py has the cache functions. Prompts tell workers to use them. But workers have NO mechanism to call them — they can only read pool.md manually.

**MCP value**: HIGH. `ralph_check_verified(topic)` and `ralph_add_verified(topic, finding, url)` would close this loop.

---

### Exp2: Atomic Task Creation (20/20)

**Problem status: MITIGATED but not eliminated**

| Finding | Evidence |
|---|---|
| pool.py task functions work correctly | 6/6 unit tests |
| Inconsistency detection works | Both directions: pool-without-file and file-without-pool detected |
| Table format parsing is robust | 6/6 format variants handled (bold, links, spaces, etc.) |
| Task ID format is robust | T001, T999, T1000, T00001 all work |
| Planner prompt has explicit sync rule | "NEVER add a task to pool.md without creating its task file" |
| Orchestrator has `ensure_task_files_exist` safety net | Code-level auto-fix for missing files |

**Design gap**: The orchestrator's `ensure_task_files_exist()` is a safety net that creates *minimal* task files when Planner misses them. This means:
1. Task files exist but have placeholder content ("Auto-created - details to be filled")
2. Extra I/O to detect and fix inconsistencies
3. The real task description is only in pool.md, not the task file

**MCP value**: MEDIUM. `ralph_create_task(id, type, title, desc)` would be atomic, but the safety net already prevents data loss. Integration tests needed to measure how often Planner actually misses.

---

### Exp3: Pivot Signal Detection (24/24)

**Problem status: Well-defended but fragile**

| Finding | Evidence |
|---|---|
| pool.py pivot functions all work | 13/13 unit tests |
| All format variants detected | 7/7: plain, bold, no-space, timestamp, extra spaces, multiline, unicode |
| clear_pivot_recommendation is task-specific | Only clears the target task, preserves others |
| Planner prompt has detailed pivot instructions | `<reading_evaluator_signals>` section with examples |
| Orchestrator blocks DONE on pending pivots | Code-level check in orchestrator.py |

**Design gap**: Two layers of defense (prompt + code), but:
1. Planner must manually parse markdown to find `[PIVOT_RECOMMENDED]`
2. Planner must use Edit tool to mark as `[PIVOT_PROCESSED]`
3. Both operations are error-prone for an LLM
4. The orchestrator's code-level DONE block is a safety net, not a solution

**MCP value**: MEDIUM. `ralph_get_pivot_signals()` and `ralph_mark_pivot_processed(task_id)` would be cleaner, but the prompt instructions already work for the basic case. Integration tests needed to measure detection rate under realistic conditions.

---

### Exp4: Concurrent Writes (13/13)

**Problem confirmed: YES (for agent-level, not pool.py-level)**

| Finding | Evidence |
|---|---|
| pool.py file locking works perfectly | 50 concurrent writes, 0 data loss |
| Mixed operations are safe | 30 concurrent mixed writes (findings + progress + verified info) |
| Stress test passes | 50 rapid-fire writes, large content, all preserved |
| File lock timeout works | Correctly throws TimeoutError |
| **BUT agents bypass pool.py** | Agents use Edit/Write tools directly on pool.md |
| **Simulated Edit race loses data** | Agent2's write overwrites Agent1's (proven) |
| pool.py locked writes preserve both | Same scenario with pool.py: both preserved |

**Design gap**: CRITICAL. pool.py's locking is excellent but **completely bypassed** by agents. Agents:
1. Read pool.md with the Read tool
2. Compute changes
3. Write back with Edit/Write tool

If two agents do this concurrently, the last write wins and the first write's changes are lost.

**MCP value**: HIGH. `ralph_append_finding(finding)` would route all writes through pool.py's locked functions, eliminating the Edit tool bypass. This is the most impactful MCP tool because it addresses a proven data loss scenario.

---

## Decision Matrix

| MCP Tool Candidate | Problem Real? | Severity | pool.py Ready? | MCP Value |
|---|---|---|---|---|
| Verified Info Cache (3 tools) | YES | Medium | YES | **HIGH** |
| Atomic Task Creation (1 tool) | Mitigated | Low | YES | MEDIUM |
| Pivot Signals (2 tools) | Defended | Low-Med | YES | MEDIUM |
| Findings Append (1 tool) | YES | **HIGH** | YES | **HIGH** |

## Recommendation

### Must implement (confirmed problems):
1. **`ralph_append_finding(finding)`** — Eliminates proven data loss in concurrent writes
2. **`ralph_check_verified(topic)` / `ralph_add_verified(...)` / `ralph_list_verified()`** — Closes the verified info loop

### Should implement (defense in depth):
3. **`ralph_create_task(id, type, title, desc)`** — Atomic task creation (safety net exists but MCP is cleaner)
4. **`ralph_get_pivot_signals()` / `ralph_mark_pivot_processed(task_id)`** — Cleaner than markdown parsing

### Integration tests still needed for:
- Exp1: Do workers actually duplicate WebSearch in practice?
- Exp2: How often does Planner actually miss task file creation?
- Exp3: How reliably does Planner detect buried pivot signals?
- Exp4: Do parallel workers actually lose findings in practice?
