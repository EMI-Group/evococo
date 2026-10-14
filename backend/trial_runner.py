"""Run an algorithm trial with checks independent of its generated demo code."""

import sys
import builtins
import json
import math
from pathlib import Path
from types import ModuleType


def _state_nan_counts(algorithm):
    """Inspect registered state, without guessing algorithm-specific field names."""
    import torch

    return {
        name: int(torch.isnan(value).sum())
        for name, value in algorithm.named_buffers()
        if value.is_floating_point() or value.is_complex()
    }


def run_trial(
    filename: str,
    *,
    supports_constraints: bool = False,
    objective_mode: str = "multi",
    objective_direction: str = "min",
) -> None:
    import torch
    from evox import metrics
    from evox.workflows import StdWorkflow

    workflows = []
    initial_nan_counts = {}
    evaluation_modes = {}
    initialized = set()
    workflow_types = [StdWorkflow]
    if supports_constraints:
        from evomo.workflows import UnifiedWorkflow

        workflow_types.append(UnifiedWorkflow)
    original_igd = metrics.igd
    original_print = builtins.print
    originals = []

    def validate_workflow(workflow):
        fit = getattr(workflow.algorithm, "fit", None)
        if not isinstance(fit, torch.Tensor) or fit.ndim != 2 or fit.numel() == 0:
            raise ValueError(
                "Runtime validation failed: missing or empty algorithm.fit"
            )
        if not torch.isfinite(fit).all():
            raise ValueError(
                "Runtime validation failed: algorithm.fit contains NaN or infinite values"
            )
        if objective_mode == "single":
            pop = getattr(workflow.algorithm, "pop", None)
            if (
                fit.shape[1] != 1
                or not isinstance(pop, torch.Tensor)
                or pop.ndim != 2
                or pop.shape != (fit.shape[0], 12)
                or not torch.isfinite(pop).all()
            ):
                raise ValueError(
                    "Runtime validation failed: invalid single-objective state shape"
                )
            evaluated = workflow.problem.evaluate(pop)
            actual = workflow.fitness_transform(
                evaluated[0] if isinstance(evaluated, tuple) else evaluated
            )
            if actual.shape != fit.shape or not torch.allclose(
                fit, actual, rtol=1e-4, atol=1e-5
            ):
                raise ValueError(
                    "Runtime validation failed: algorithm.fit does not match selected population"
                )
        if evaluation_modes.get(id(workflow)) == "constrained":
            cv = getattr(workflow.algorithm, "cv", None)
            pop = getattr(workflow.algorithm, "pop", None)
            if (
                not isinstance(cv, torch.Tensor)
                or cv.ndim not in (1, 2)
                or cv.numel() == 0
                or cv.shape[0] != fit.shape[0]
                or not torch.isfinite(cv).all()
                or (cv < 0).any()
            ):
                raise ValueError(
                    "Runtime validation failed: missing, misaligned or invalid algorithm.cv"
                )
            # The constrained profile uses deterministic numerical benchmarks.
            # Re-evaluate selected rows so zero-filled/stale CV cannot fake feasibility.
            actual_fit, actual_cv = workflow.problem.evaluate(pop)
            if objective_mode == "single":
                actual_fit = workflow.fitness_transform(actual_fit)
            if actual_cv.shape != cv.shape or not torch.allclose(
                cv, actual_cv, rtol=1e-4, atol=1e-5
            ):
                raise ValueError(
                    "Runtime validation failed: algorithm.cv does not match selected population"
                )
            if actual_fit.shape != fit.shape or not torch.allclose(
                fit, actual_fit, rtol=1e-4, atol=1e-5
            ):
                raise ValueError(
                    "Runtime validation failed: algorithm.fit does not match selected population"
                )
        baseline = initial_nan_counts[id(workflow)]
        for name, count in _state_nan_counts(workflow.algorithm).items():
            if count > baseline.get(name, 0):
                raise ValueError(
                    f"Runtime validation failed: algorithm state '{name}' contains "
                    f"new NaN values ({count}; constructor baseline {baseline.get(name, 0)}). "
                    "Check the producing operation and masked updates; preserve legitimate sentinels."
                )

    def validate_workflows():
        if not workflows:
            raise ValueError(
                "Runtime validation failed: no supported workflow was executed"
            )
        if not initialized:
            raise ValueError("Runtime validation failed: no workflow was initialized")
        for workflow in workflows:
            if id(workflow) in initialized:
                validate_workflow(workflow)

    for workflow_type in workflow_types:
        original_init = workflow_type.__init__
        original_init_step = workflow_type.init_step
        original_evaluate = workflow_type._evaluate
        originals.append(
            (workflow_type, original_init, original_init_step, original_evaluate)
        )

        def track_workflow(workflow, *args, _original=original_init, **kwargs):
            _original(workflow, *args, **kwargs)
            workflows.append(workflow)
            initial_nan_counts[id(workflow)] = _state_nan_counts(workflow.algorithm)

        def track_evaluate(workflow, *args, _original=original_evaluate, **kwargs):
            result = _original(workflow, *args, **kwargs)
            evaluation_modes[id(workflow)] = (
                "constrained"
                if isinstance(result, tuple) and result[1] is not None
                else "unconstrained"
            )
            return result

        def validated_init_step(
            workflow, *args, _original=original_init_step, **kwargs
        ):
            result = _original(workflow, *args, **kwargs)
            initialized.add(id(workflow))
            validate_workflow(workflow)
            return result

        workflow_type.__init__ = track_workflow
        workflow_type.init_step = validated_init_step
        if supports_constraints:
            workflow_type._evaluate = track_evaluate

    def validated_igd(*args, **kwargs):
        # Inspect the original state even when the demo filtered its local `fit`.
        # This runs only at metric checkpoints, outside the compiled step.
        validate_workflows()
        return original_igd(*args, **kwargs)

    def validated_print(*args, **kwargs):
        if (
            args
            and isinstance(args[0], str)
            and args[0].startswith("SINGLE_OBJECTIVE_METRICS: ")
        ):
            validate_workflows()
            row = json.loads(args[0].removeprefix("SINGLE_OBJECTIVE_METRICS: "))
            workflow = workflows[-1]
            fit = workflow.algorithm.fit
            best = (fit.min() if objective_direction == "min" else fit.max()).item()
            if row.get("direction") != objective_direction or not math.isclose(
                row["best_fitness"], best, rel_tol=1e-5, abs_tol=1e-8
            ):
                raise ValueError(
                    "Runtime validation failed: reported best fitness does not match algorithm state"
                )
            if supports_constraints:
                if row.get("problem") != type(workflow.problem).__name__:
                    raise ValueError(
                        "Runtime validation failed: reported problem does not match workflow"
                    )
                if evaluation_modes[id(workflow)] == "constrained":
                    cv = workflow.algorithm.cv
                    total_cv = cv.sum(dim=1) if cv.ndim == 2 else cv
                    feasible = total_cv <= 0
                    feasible_best = (
                        (
                            fit[feasible].min()
                            if objective_direction == "min"
                            else fit[feasible].max()
                        ).item()
                        if feasible.any()
                        else None
                    )
                    reported = row.get("feasible_best_fitness")
                    if (
                        row.get("population") != fit.shape[0]
                        or row.get("feasible_count") != feasible.sum().item()
                        or not math.isclose(
                            row.get("mean_cv", float("nan")),
                            total_cv.mean().item(),
                            rel_tol=1e-5,
                            abs_tol=1e-8,
                        )
                        or (reported is None) != (feasible_best is None)
                        or (
                            feasible_best is not None
                            and not math.isclose(
                                reported, feasible_best, rel_tol=1e-5, abs_tol=1e-8
                            )
                        )
                    ):
                        raise ValueError(
                            "Runtime validation failed: reported constrained fitness does not match algorithm state"
                        )
        return original_print(*args, **kwargs)

    original_argv = sys.argv
    original_path = sys.path[:]
    original_main = sys.modules["__main__"]
    script = Path(filename).resolve()
    main_module = ModuleType("__main__")
    main_module.__file__ = str(script)
    main_module.__cached__ = None
    metrics.igd = validated_igd
    if objective_mode == "single":
        builtins.print = validated_print
    sys.argv = [filename]
    sys.path[0:1] = [str(script.parent)]
    sys.modules["__main__"] = main_module
    try:
        exec(compile(script.read_bytes(), str(script), "exec"), main_module.__dict__)  # noqa: S102 - intentional: execute the trial script in its child process
        validate_workflows()
        if supports_constraints and set(evaluation_modes.values()) != {
            "unconstrained",
            "constrained",
        }:
            raise ValueError(
                "Runtime validation failed: source-supported constraints require both unconstrained and constrained trials"
            )
    finally:
        # Keep the original traceback on failure and restore state for callers.
        for (
            workflow_type,
            original_init,
            original_init_step,
            original_evaluate,
        ) in originals:
            workflow_type.__init__ = original_init
            workflow_type.init_step = original_init_step
            workflow_type._evaluate = original_evaluate
        metrics.igd = original_igd
        builtins.print = original_print
        sys.argv = original_argv
        sys.path[:] = original_path
        sys.modules["__main__"] = original_main


if __name__ == "__main__":
    run_trial(
        sys.argv[1],
        supports_constraints="--constraints" in sys.argv[2:],
        objective_mode="single" if "--single-objective" in sys.argv[2:] else "multi",
        objective_direction="max" if "--maximize" in sys.argv[2:] else "min",
    )
