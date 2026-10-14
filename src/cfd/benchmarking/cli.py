# benchmarking/cli.py
import argparse
import traceback
import os

from cfd.benchmarking.config import SuiteConfig, BenchmarkConfig
from cfd.benchmarking.suite_runner import SuiteRunner
from cfd.benchmarking.writer import ResultWriter

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            prog="cfd-bench",
            description="CFD Benchmarking CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single benchmark config")
    run_parser.add_argument("config", type=str, help="Path to the benchmark config file")

    validate_parser = subparsers.add_parser("validate", help="Validate a benchmark config or suite config")
    validate_parser.add_argument("config", type=str, help="Path to the benchmark config or suite config file")

    list_parser = subparsers.add_parser("list", help="List registered benchmark components")
    list_parser.add_argument("resource", type=str, choices=["solvers", "operators", "backends", "timings", "metrics"])

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        match args.command:
            case "run":
                return run_command(args)
            case "validate":
                return validate_command(args)
            case "list":
                return list_command(args)

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    except KeyboardInterrupt:
        print("Execution interrupted by user.")
        return 1

    return 0

def run_command(args) -> int:
    config_path = args.config
    try:
        config = SuiteConfig.from_json(config_path)
        writer_root = os.path.join(os.getcwd(), "benchmarks", "results")
        writer = ResultWriter(root=writer_root, write_raw=True)
        runner = SuiteRunner(writer=writer)
        result = runner.run(config)
        print(result.to_summary())

    except ValueError as e:
        tb = traceback.extract_tb(e.__traceback__)
        last = tb[-1] if tb else None

        if last:
            print(f"{last.filename}:{last.lineno} - {last.name}: {e}")
        else:
            print(f"Invalid suite config: {e}")
        return 1

    return 0

def validate_command(args) -> int:
    config_path = args.config
    try:
        config = SuiteConfig.from_json(config_path)
        config.validate()
        print(f"Valid suite config: {config.name}")
    except ValueError as e:
        tb = traceback.extract_tb(e.__traceback__)
        last = tb[-1] if tb else None

        if last:
            print(f"{last.filename}:{last.lineno} - {last.name}: {e}")
        else:
            print(f"Invalid suite config: {e}")

        return 1

    return 0

def list_command(args) -> int:
    resource = args.resource
    from cfd.solvers.registry import SOLVER_REGISTRY
    from cfd.operators.registry import OPERATOR_REGISTRY
    from cfd.common.config import Backend
    from cfd.profiling.config import TIMINGS, SCALAR_METRICS, SOURCE_STAGES, REDUCERS, SCOPES
    try:
        match resource:
            case "solvers":
                print("Registered solvers:")
                for solver, backend in SOLVER_REGISTRY:
                    print(f"  - {solver} (backend: {backend})")
            case "operators":
                print("Registered operators:")
                for operator, backend in OPERATOR_REGISTRY:
                    print(f"  - {operator} (backend: {backend})")
            case "backends":
                print("Registered backends:")
                for backend in Backend:
                    print(f"  - {backend.value}")
            case "timings":
                print("Registered timings:")
                for timing in TIMINGS:
                    print(f"  - {timing}")
            case "metrics":
                print("Registered scalar metrics:")
                for metric in SCALAR_METRICS:
                    print(f"  - {metric}")
                print("\nRegistered field sources:")
                for source, stages in SOURCE_STAGES.items():
                    print(f"  - {source} (stages: {', '.join(stages)})")
                print("Registered reducers:")
                for reducer in REDUCERS:
                    print(f"  - {reducer}")
                print("Registered scopes:")
                for scope in SCOPES:
                    print(f"  - {scope}")
    except Exception as e:
        print(f"Error listing {resource}: {e}")
        return 1

    return 0

