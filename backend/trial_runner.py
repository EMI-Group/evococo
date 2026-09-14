"""Run an algorithm trial with checks independent of its generated demo code."""

import sys
from pathlib import Path
from types import ModuleType


def run_trial(filename: str) -> None:
    import torch
    from evox import metrics
    from evox.workflows import StdWorkflow

    workflows = []
    original_init = StdWorkflow.__init__
    original_igd = metrics.igd

    def track_workflow(workflow, *args, **kwargs):
        original_init(workflow, *args, **kwargs)
        workflows.append(workflow)

    def validate_workflows():
        if not workflows:
            raise ValueError("Runtime validation failed: no StdWorkflow was executed")
        for workflow in workflows:
            fit = getattr(workflow.algorithm, "fit", None)
            if not isinstance(fit, torch.Tensor) or fit.ndim != 2 or fit.numel() == 0:
                raise ValueError(
                    "Runtime validation failed: missing or empty algorithm.fit"
                )
            if not torch.isfinite(fit).all():
                raise ValueError(
                    "Runtime validation failed: algorithm.fit contains NaN or infinite values"
                )

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
    StdWorkflow.__init__ = track_workflow
    metrics.igd = validated_igd
    sys.argv = [filename]
    sys.path[0:1] = [str(script.parent)]
    sys.modules["__main__"] = main_module
    try:
        exec(compile(script.read_bytes(), str(script), "exec"), main_module.__dict__)  # noqa: S102 - intentional: execute the trial script in its child process
        validate_workflows()
    finally:
        # Keep the original traceback on failure and restore state for callers.
        StdWorkflow.__init__ = original_init
        metrics.igd = original_igd
        sys.argv = original_argv
        sys.path[:] = original_path
        sys.modules["__main__"] = original_main


if __name__ == "__main__":
    run_trial(sys.argv[1])
