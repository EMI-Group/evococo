"""Performance-stats aggregation and report formatting for EvoCoCo.

Extracted from backend/engine.py. Behavior is identical to the original
inline logic: aggregate_stage_metrics resets totals to 0 before recomputing
them, and format_stats_header builds the standalone human-readable report.
"""

import json

from .config import STRATEGIES_SHORT


def aggregate_stage_metrics(perf_stats):
    """Reset and recompute total token/LLM-time metrics across all stages.

    Accumulates metrics from the analyst/rag/architect/judge stages plus each
    branch's coder/static_fixes/runtime_fixes entries, writing the totals into
    perf_stats["total_tokens"] and perf_stats["total_llm_time"].
    """
    perf_stats["total_tokens"] = {"prompt": 0, "completion": 0, "total": 0}
    perf_stats["total_llm_time"] = 0.0
    input_cache = {
        "cached_input_tokens": None,
        "uncached_input_tokens": None,
        "reported_prompt_tokens": 0,
        "unreported_prompt_tokens": 0,
        "hit_rate": None,
    }
    perf_stats["input_cache"] = input_cache

    def add_metrics(m):
        if m:
            perf_stats["total_tokens"]["prompt"] += m.get("prompt_tokens", 0)
            perf_stats["total_tokens"]["completion"] += m.get("completion_tokens", 0)
            perf_stats["total_tokens"]["total"] += m.get("total_tokens", 0)
            perf_stats["total_llm_time"] += m.get("latency", 0.0)
            cached = m.get("cached_input_tokens")
            uncached = m.get("uncached_input_tokens")
            if cached is not None and uncached is not None:
                input_cache["cached_input_tokens"] = (
                    input_cache["cached_input_tokens"] or 0
                ) + cached
                input_cache["uncached_input_tokens"] = (
                    input_cache["uncached_input_tokens"] or 0
                ) + uncached
                input_cache["reported_prompt_tokens"] += cached + uncached
            else:
                input_cache["unreported_prompt_tokens"] += m.get("prompt_tokens", 0)

    add_metrics(perf_stats["stages"].get("analyst"))
    add_metrics(perf_stats["stages"].get("rag"))
    add_metrics(perf_stats["stages"].get("architect"))
    add_metrics(perf_stats["stages"].get("judge"))

    for b in perf_stats["stages"].get("branches", []):
        b_m = b.get("metrics", {})
        add_metrics(b_m.get("coder"))
        for s_fix in b_m.get("static_fixes", []):
            add_metrics(s_fix)
        for r_fix in b_m.get("runtime_fixes", []):
            add_metrics(r_fix)

    if input_cache["reported_prompt_tokens"]:
        input_cache["hit_rate"] = (
            input_cache["cached_input_tokens"] / input_cache["reported_prompt_tokens"]
        )


def format_stats_header(perf_stats, judge_result=None):
    """Build the performance report saved separately from the final code.

    `judge_result` is the JudgeResult model when the judge step succeeded,
    or None otherwise. The winning branch is read from perf_stats, including
    when the pipeline selected it through the deterministic fallback.
    """
    stats_header = (
        "# ==============================================================================\n"
        f"# EvoCoCo Performance Statistics\n"
        f"# Total LLM Time: {perf_stats['total_llm_time']:.2f}s\n"
        f"# Total Tokens: {perf_stats['total_tokens']['total']:,} "
        f"(Prompt: {perf_stats['total_tokens']['prompt']:,}, Completion: {perf_stats['total_tokens']['completion']:,})\n"
    )
    input_cache = perf_stats.get("input_cache", {})
    cached = input_cache.get("cached_input_tokens")
    if cached is None:
        stats_header += "# Input Cache: Unknown (provider did not report cache usage)\n"
    else:
        hit_rate = input_cache.get("hit_rate")
        rate_text = f"{hit_rate:.1%}" if hit_rate is not None else "N/A"
        stats_header += (
            f"# Input Cache: {cached:,} cached / "
            f"{input_cache['reported_prompt_tokens']:,} reported input tokens "
            f"({rate_text}); unreported input: "
            f"{input_cache['unreported_prompt_tokens']:,}\n"
        )
    stats_header += "#\n# Operators & Stages Status:\n"
    if "analyst" in perf_stats["stages"]:
        a = perf_stats["stages"]["analyst"]
        analyst_status = "CACHED (local)" if a.get("local_cache_hit") else "SUCCESS"
        stats_header += f"#   - Analyst: {analyst_status} | Latency: {a.get('latency', 0.0):.2f}s | Tokens: {a.get('total_tokens', 0):,} (Prompt: {a.get('prompt_tokens', 0):,}, Completion: {a.get('completion_tokens', 0):,})\n"
    if "rag" in perf_stats["stages"]:
        r = perf_stats["stages"]["rag"]
        stats_header += f"#   - RAG: SUCCESS | Latency: {r.get('latency', 0.0):.2f}s | Tokens: {r.get('total_tokens', 0):,} (Prompt: {r.get('prompt_tokens', 0):,}, Completion: {r.get('completion_tokens', 0):,})\n"
    if "architect" in perf_stats["stages"]:
        arc = perf_stats["stages"]["architect"]
        stats_header += f"#   - Architect: SUCCESS | Latency: {arc.get('latency', 0.0):.2f}s | Tokens: {arc.get('total_tokens', 0):,} (Prompt: {arc.get('prompt_tokens', 0):,}, Completion: {arc.get('completion_tokens', 0):,})\n"

    if "branches" in perf_stats["stages"]:
        for b in perf_stats["stages"]["branches"]:
            b_idx = b.get("branch_idx", -1)
            strat_name = b.get("strategy", "Unknown Strategy")
            success_str = "SUCCESS" if b.get("success") else "FAILED"
            igd_val = b.get("igd", float("inf"))
            if isinstance(igd_val, float) and igd_val != float("inf"):
                igd_str = f"{igd_val:.5f}"
            else:
                igd_str = str(igd_val)
            stats_header += f"#   - Branch {b_idx} ({strat_name}): {success_str} | Final IGD: {igd_str}\n"
            constraint_history = b.get("constraint_history", [])
            if constraint_history:
                last = constraint_history[-1]
                stats_header += (
                    f"#     * {last['problem']}: feasible {last['feasible_count']}/{last['population']} "
                    f"| Mean CV: {last['mean_cv']:.5g} | Feasible IGD: {last['feasible_igd']}\n"
                )

            b_m = b.get("metrics", {})
            if b_m.get("coder"):
                c = b_m["coder"]
                stats_header += f"#     * Coder: SUCCESS | Latency: {c.get('latency', 0.0):.2f}s | Tokens: {c.get('total_tokens', 0):,} (Prompt: {c.get('prompt_tokens', 0):,}, Completion: {c.get('completion_tokens', 0):,})\n"
            else:
                stats_header += "#     * Coder: FAILED or SKIPPED\n"

            s_fixes = b_m.get("static_fixes", [])
            if s_fixes:
                s_latency = sum(x.get("latency", 0.0) for x in s_fixes)
                s_tokens = sum(x.get("total_tokens", 0) for x in s_fixes)
                s_prompt = sum(x.get("prompt_tokens", 0) for x in s_fixes)
                s_completion = sum(x.get("completion_tokens", 0) for x in s_fixes)
                stats_header += f"#     * Static Fixer: {len(s_fixes)} attempts | Total Latency: {s_latency:.2f}s | Total Tokens: {s_tokens:,} (Prompt: {s_prompt:,}, Completion: {s_completion:,})\n"
            else:
                stats_header += "#     * Static Fixer: 0 attempts\n"

            r_fixes = b_m.get("runtime_fixes", [])
            if r_fixes:
                r_latency = sum(x.get("latency", 0.0) for x in r_fixes)
                r_tokens = sum(x.get("total_tokens", 0) for x in r_fixes)
                r_prompt = sum(x.get("prompt_tokens", 0) for x in r_fixes)
                r_completion = sum(x.get("completion_tokens", 0) for x in r_fixes)
                stats_header += f"#     * Runtime Fixer: {len(r_fixes)} attempts | Total Latency: {r_latency:.2f}s | Total Tokens: {r_tokens:,} (Prompt: {r_prompt:,}, Completion: {r_completion:,})\n"
            else:
                stats_header += "#     * Runtime Fixer: 0 attempts\n"

    if "judge" in perf_stats["stages"]:
        j = perf_stats["stages"]["judge"]
        judge_status = "SUCCESS" if judge_result is not None else "FAILED (Fallback)"
        stats_header += f"#   - Judge/Selector: {judge_status} | Latency: {j.get('latency', 0.0):.2f}s | Tokens: {j.get('total_tokens', 0):,} (Prompt: {j.get('prompt_tokens', 0):,}, Completion: {j.get('completion_tokens', 0):,})\n"

    winner_id = perf_stats.get("winning_branch_id")
    if winner_id is not None:
        strat_name = STRATEGIES_SHORT[winner_id % len(STRATEGIES_SHORT)]
        stats_header += f"# Winning Branch: {winner_id} (Strategy: {strat_name})\n"

    stats_header += "#\n# Raw Operator Stats JSON:\n"
    perf_stats_json_str = json.dumps(perf_stats, indent=2)
    for line in perf_stats_json_str.split("\n"):
        stats_header += f"# {line}\n"

    stats_header += "# ==============================================================================\n\n"

    return stats_header
