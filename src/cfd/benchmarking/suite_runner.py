from .writer import ResultWriter
from .runner import BenchmarkRunner
from .config import SuiteConfig
from .result import SuiteResult

class SuiteRunner:
    def __init__(self, writer: ResultWriter):
        self.writer = writer
        self.runner = BenchmarkRunner(writer)

    def _make_execution_id(self, suite: SuiteConfig) -> str:
        # Generate a unique execution ID based on the suite name and current timestamp
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        suite_name_clean = suite.name.replace(" ", "_").replace("/", "_").lower()
        return f"{suite_name_clean}_{ts}"

    def run(self, suite: SuiteConfig) -> SuiteResult:
        execution_id = self._make_execution_id(suite)
        self.writer.set_execution_id(execution_id)
        self.writer.initialize_execution(suite.name)

        benchmarks = suite.expand_suite()

        self.writer.write_config(suite)

        benchmark_results = []
        benchmark_runner = BenchmarkRunner(self.writer)

        for benchmark in benchmarks:
            result = benchmark_runner.run(benchmark)
            benchmark_results.append(result)

        suite_result = SuiteResult(
                execution_id = execution_id,
                name = suite.name,
                benchmark_results = benchmark_results,
        )

        self.writer.write_suite_result(suite_result)

        return suite_result




