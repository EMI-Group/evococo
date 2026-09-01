"""Problem-suite definitions shared by the fidelity runner and worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import torch

PROBLEM_SUITES: Final[dict[str, tuple[str, ...]]] = {
    "DTLZ": tuple(f"DTLZ{index}" for index in range(1, 8)),
    "WFG": tuple(f"WFG{index}" for index in range(1, 10)),
    "LSMOP": tuple(f"LSMOP{index}" for index in range(1, 10)),
    "MaF": tuple(f"MaF{index}" for index in range(1, 16)),
}
ALL_PROBLEMS: Final[tuple[str, ...]] = tuple(
    problem for suite in PROBLEM_SUITES.values() for problem in suite
)

PROBLEM_DIMENSIONS: Final[dict[str, int]] = {
    "DTLZ1": 7,
    "DTLZ2": 12,
    "DTLZ3": 12,
    "DTLZ4": 12,
    "DTLZ5": 12,
    "DTLZ6": 12,
    "DTLZ7": 22,
    **{f"WFG{index}": 12 for index in range(1, 10)},
    **{f"LSMOP{index}": 300 for index in range(1, 10)},
    **{f"MaF{index}": 12 for index in range(1, 7)},
    "MaF7": 22,
    "MaF8": 2,
    "MaF9": 2,
    "MaF10": 12,
    "MaF11": 12,
    "MaF12": 12,
    "MaF13": 5,
    "MaF14": 60,
    "MaF15": 60,
}


def suite_problems(suite: str) -> tuple[str, ...]:
    if suite == "all":
        return ALL_PROBLEMS
    return PROBLEM_SUITES[suite]


def problem_suite(problem: str) -> str:
    for suite, problems in PROBLEM_SUITES.items():
        if problem in problems:
            return suite
    raise ValueError(f"Unknown benchmark problem: {problem}")


def create_problem(
    problem_name: str, objectives: int, device: str
) -> tuple[object, int, torch.Tensor, torch.Tensor]:
    """Return ``(problem, dimension, lower_bound, upper_bound)``."""
    import torch

    dimension = PROBLEM_DIMENSIONS[problem_name]
    suite = problem_suite(problem_name)
    from evomo.problems import numerical as evomo_numerical

    class_name = problem_name if suite != "MaF" else f"MAF{problem_name[3:]}"
    problem_class = getattr(evomo_numerical, class_name)
    problem = problem_class(d=dimension, m=objectives)
    lower_bound = torch.zeros(dimension, device=device)
    if suite == "WFG":
        lower_bound = problem.lower.to(device=device)
        upper_bound = problem.upper.to(device=device)
    elif suite == "LSMOP" or problem_name in {"MaF14", "MaF15"}:
        upper_bound = torch.cat(
            [
                torch.ones(objectives - 1, device=device),
                torch.full(
                    (dimension - objectives + 1,),
                    10.0,
                    device=device,
                ),
            ]
        )
    elif problem_name in {"MaF8", "MaF9"}:
        lower_bound = torch.full((dimension,), -10000.0, device=device)
        upper_bound = torch.full((dimension,), 10000.0, device=device)
    elif problem_name in {"MaF10", "MaF11", "MaF12"}:
        upper_bound = torch.arange(
            2,
            2 * dimension + 1,
            2,
            dtype=torch.get_default_dtype(),
            device=device,
        )
    elif problem_name == "MaF13":
        lower_bound[2:] = -2
        upper_bound = torch.cat(
            [
                torch.ones(2, device=device),
                torch.full((dimension - 2,), 2.0, device=device),
            ]
        )
    else:
        upper_bound = torch.ones(dimension, device=device)
    return problem, dimension, lower_bound, upper_bound
