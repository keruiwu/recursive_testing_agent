from pathlib import Path
import argparse
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm

from _strategy import (
    Strategy,
    MetricResult,
    get_strategy,
    get_heuristic_strategies,
    STRATEGIES,
    HEURISTIC_STRATEGIES,
)


def find_leaf_directories(root: Path) -> list[Path]:
    """
    Find all leaf directories (directories with no subdirectories).
    """
    leaves = []
    for entry in os.scandir(root):
        if entry.is_dir():
            subdirs = [e for e in os.scandir(entry.path) if e.is_dir()]
            if subdirs:
                leaves.extend(find_leaf_directories(Path(entry.path)))
            else:
                leaves.append(Path(entry.path))
    return leaves


def load_results(directories: list[Path]) -> tuple[dict[str, list[dict]], int]:
    """
    Load results from N directories.

    Returns:
        Tuple of (dict mapping problem_id to list of N result dicts, N)
    """
    results = defaultdict(list)
    n = len(directories)

    for dir_path in tqdm(directories):
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {dir_path}")

        json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        dir_results = {}

        for filename in json_files:
            filepath = os.path.join(dir_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            problem_id = data.get("question") or data.get("instance_id") or filename
            dir_results[problem_id] = data

        for problem_id, data in dir_results.items():
            results[problem_id].append(data)

    # Verify all problems have N results
    for problem_id, res_list in results.items():
        if len(res_list) != n:
            print(f"Warning: Problem {problem_id} has {len(res_list)} results, expected {n}")

    return results, n


_LLM_STRATEGY_NAMES = [k for k in STRATEGIES if k not in HEURISTIC_STRATEGIES]
_SEPARATOR = "  " + "─" * 44


def _sanitize_path_tag(raw: str) -> str:
    """
    Convert arbitrary identifiers (e.g. hf:/path/to/model) into a safe path tag.
    """
    tag = re.sub(r"[^A-Za-z0-9._-]+", "__", raw.strip())
    tag = tag.strip("._-")
    return tag or "model"


def run_strategy(
    strategy: Strategy, results: dict[str, list[dict]], n: int,
    k_values: list[int] | None = None,
) -> dict[int, MetricResult]:
    """Run a single strategy and return its scores."""
    return strategy.run(results, n, k_values=k_values)


def run_heuristic_strategies(
    results: dict[str, list[dict]], n: int, k_values: list[int] | None = None,
    **kwargs
) -> dict[str, dict[int, MetricResult]]:
    """Run all heuristic strategies and return their scores."""
    all_scores = {}
    strategies = get_heuristic_strategies(**kwargs)

    for i, strategy in enumerate(strategies):
        all_scores[strategy.name] = strategy.run(results, n, k_values=k_values)
        if i < len(strategies) - 1:
            print(_SEPARATOR)

    return all_scores


def run_all_strategies(
    results: dict[str, list[dict]], n: int, k_values: list[int] | None = None,
    **kwargs
) -> dict[str, dict[int, MetricResult]]:
    """Run all strategies: heuristic first, then LLM-based."""
    all_scores = {}
    all_scores.update(run_heuristic_strategies(results, n, k_values=k_values, **kwargs))
    for name in _LLM_STRATEGY_NAMES:
        print(_SEPARATOR)
        strategy = get_strategy(name, **kwargs)
        all_scores[name] = strategy.run(results, n, k_values=k_values)
    return all_scores


def main():
    strategy_choices = list(STRATEGIES.keys()) + ["heuristic", "all"]

    parser = argparse.ArgumentParser(description="Test-Time Scaling evaluation (modular)")
    ### Aggregation Strategy
    parser.add_argument("--strategy", type=str, default="heuristic", choices=strategy_choices,
                        help="Evaluation strategy (default: heuristic)")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="k values to evaluate (e.g. --k 1 2 4). If not set, runs all k=1,2,4,...,N")
    ### LLM-based Aggregation
    parser.add_argument("--model", type=str, default=None,
                        help="Model for LLM-based aggregation")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Base URL for vLLM deployment (optional)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for logs")
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="Comma-separated GPU ids to expose (e.g. 0,1,2,3)")
    parser.add_argument("--hf_device_map", type=str, default="auto",
                        help="Transformers device_map for local hf:* models (default: auto)")
    parser.add_argument("--hf_torch_dtype", type=str, default=None,
                        help="Torch dtype for local hf:* models (e.g. bfloat16, float16)")
    parser.add_argument("--hf_max_new_tokens", type=int, default=4096,
                        help="Max new tokens for local hf:* generation (default: 4096)")
    parser.add_argument("--hf_temperature", type=float, default=0.2,
                        help="Sampling temperature for local hf:* generation (default: 0.2)")
    parser.add_argument("--hf_top_p", type=float, default=0.95,
                        help="Top-p for local hf:* generation (default: 0.95)")
    ### Evaluation
    parser.add_argument("--task", type=str, default="browsecomp", choices=["browsecomp", "browsecomp-plus", "deepsearchqa", "hle", "researchrubrics", "healthbench"],
                        help="Task type for judging (default: browsecomp)")
    parser.add_argument("--skip_score", action="store_true",
                        help="Skip compute_score during aggregation; score posthoc from logs")

    parser.add_argument("directories", nargs="+",
                        help="Parent directories to search for leaf directories (json)")

    args = parser.parse_args()

    if args.strategy in set(_LLM_STRATEGY_NAMES) | {"all"} and args.model is None:
        parser.error(f"--model is required for --strategy {args.strategy}")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Find all leaf directories from the given parent directories
    all_leaves = []
    for d in args.directories:
        all_leaves.extend(find_leaf_directories(Path(d)))
    all_leaves = sorted(all_leaves)

    n = len(all_leaves)
    if n == 0: parser.error("No leaf directories found")

    print(f"Found {n} leaf directories")
    results, n = load_results(all_leaves)

    n_problems = len(results)
    header_parts = [f"task: {args.task}", f"N: {n}", f"problems: {n_problems}"]
    if args.model:
        header_parts.append(f"model: {args.model}")
    header = "  " + "   ".join(header_parts) + "  "
    bar = "─" * len(header)
    print(f"\n{bar}\n{header}\n{bar}\n")

    # Extra kwargs forwarded to strategy constructors
    output_dir = args.output_dir
    if args.strategy in _LLM_STRATEGY_NAMES + ["all"]:
        _parts = [p for p in args.directories[0].split("/") if p]
        _dir_tag = "-".join(_parts[-4:]) if _parts else args.directories[0]
        _model_tag = _sanitize_path_tag(args.model or "model")
        _dir_tag = _sanitize_path_tag(_dir_tag)
        output_dir = args.output_dir or f"output/aggregation/{_model_tag}/{_dir_tag}"
        os.makedirs(output_dir, exist_ok=True)
    strategy_kwargs = {
        "model": args.model,
        "api_base": args.api_base,
        "task": args.task,
        "max_workers": args.max_workers,
        "output_dir": output_dir,
        "resume": True,
        "skip_score": args.skip_score,
        "hf_device_map": args.hf_device_map,
        "hf_torch_dtype": args.hf_torch_dtype,
        "hf_max_new_tokens": args.hf_max_new_tokens,
        "hf_temperature": args.hf_temperature,
        "hf_top_p": args.hf_top_p,
    }

    k_values = args.k

    if args.strategy == "heuristic":
        run_heuristic_strategies(results, n, k_values=k_values, **strategy_kwargs)
    elif args.strategy == "all":
        run_all_strategies(results, n, k_values=k_values, **strategy_kwargs)
    else:
        strategy = get_strategy(args.strategy, **strategy_kwargs)
        run_strategy(strategy, results, n, k_values=k_values)


if __name__ == "__main__":
    main()
