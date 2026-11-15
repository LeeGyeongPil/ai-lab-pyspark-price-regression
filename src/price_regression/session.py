"""One local SparkSession, quiet by default, with a writable temp folder."""
from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession

_spark: SparkSession | None = None


def default_tmp_dir() -> Path:
    """``SPARK_LOCAL_DIRS`` if the user set it, otherwise ``.spark-tmp/`` in the current folder."""
    env = os.environ.get("SPARK_LOCAL_DIRS")
    return Path(env.split(",")[0]) if env else Path.cwd() / ".spark-tmp"


def _prepare_tmp(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log4j = tmp_dir / "log4j2.properties"
    log4j.write_text(
        "rootLogger.level = error\n"
        "rootLogger.appenderRef.stdout.ref = console\n"
        "appender.console.type = Console\n"
        "appender.console.name = console\n"
        "appender.console.target = SYSTEM_ERR\n"
        "appender.console.layout.type = PatternLayout\n"
        "appender.console.layout.pattern = %p %c: %m%n\n"
    )
    # The JVM itself needs a writable java.io.tmpdir; in restricted environments the default
    # folder is not writable and Spark dies with JAVA_GATEWAY_EXITED. JAVA_TOOL_OPTIONS reaches
    # every JVM that PySpark starts.
    flags = f"-Djava.io.tmpdir={tmp_dir} -Dlog4j2.configurationFile={log4j}"
    existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "java.io.tmpdir" not in existing:
        os.environ["JAVA_TOOL_OPTIONS"] = f"{existing} {flags}".strip()
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(tmp_dir))


def get_spark(master: str = "local[*]", tmp_dir: Path | None = None) -> SparkSession:
    """Create (or reuse) the session. Call :func:`stop_spark` when done."""
    global _spark
    if _spark is not None:
        return _spark
    tmp = Path(tmp_dir) if tmp_dir else default_tmp_dir()
    _prepare_tmp(tmp)
    spark = (
        SparkSession.builder.master(master)
        .appName("price-regression")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.local.dir", str(tmp))
        .config("spark.sql.warehouse.dir", str(tmp / "warehouse"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    _spark = spark
    return spark


def stop_spark() -> None:
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
