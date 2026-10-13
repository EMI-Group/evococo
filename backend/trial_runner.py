"""Run an algorithm trial with checks independent of its generated demo code."""

import sys
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


def run_trial(filename: str, *, supports_constraints: bool = False) -> None:
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

    original_argv = sys.argv
    original_path = sys.path[:]
    original_main = sys.modules["__main__"]
    script = Path(filename).resolve()
    main_module = ModuleType("__main__")
    main_module.__file__ = str(script)
    main_module.__cached__ = None
    metrics.igd = validated_igd
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
        sys.argv = original_argv
        sys.path[:] = original_path
        sys.modules["__main__"] = original_main


if __name__ == "__main__":
    run_trial(sys.argv[1], supports_constraints="--constraints" in sys.argv[2:])
