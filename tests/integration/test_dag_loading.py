import sys
from pathlib import Path

import pytest

# Add dags directory to path so we can import the pipeline
DAGS_DIR = Path(__file__).parent.parent.parent / "dags"
sys.path.insert(0, str(DAGS_DIR))


@pytest.fixture(scope="module")
def dag_object():
    """Load the DAG object once for all tests."""
    try:
        import qe_pipeline

        # Access the DAG object directly
        # If it's not named 'dag', find it in the module
        dag = getattr(qe_pipeline, "dag", None)
        if dag is None:
            for _, obj in vars(qe_pipeline).items():
                if hasattr(obj, "task_dict") and hasattr(obj, "dag_id"):
                    dag = obj
                    break
        return dag
    except ImportError as e:
        pytest.fail(f"Failed to import DAG: {e}")
    finally:
        # Cleanup path is handled by module scope behavior roughly,
        # but sys.path persists. That's fine for test process.
        pass


class TestDAGIntegrity:
    """Tests to ensure DAG structure and integrity."""

    def test_dag_loaded_successfully(self, dag_object):
        """Verify DAG object was found and loaded."""
        assert dag_object is not None
        assert dag_object.dag_id == "asr_qe_pipeline"

    def test_dag_has_no_cycles(self, dag_object):
        """Check for cycle errors in the DAG."""
        # Airflow's DAG.validate() checks for cycles
        dag_object.validate()

    @pytest.mark.parametrize(
        "expected_task_id",
        [
            "wait_for_incoming_data",
            "ingest_data",
            "archive_processed_files",
            "train_model",
            "validate_model",
            "deploy_model",
        ],
    )
    def test_dag_contains_task(self, dag_object, expected_task_id):
        """Verify specific critical tasks exist."""
        assert expected_task_id in dag_object.task_dict

    def test_task_dependencies(self, dag_object):
        """Verify the order of operations (dependencies)."""
        tasks = dag_object.task_dict

        # Ingest depends on Wait
        assert "wait_for_incoming_data" in tasks["ingest_data"].upstream_task_ids

        # Archive depends on Ingest
        assert "ingest_data" in tasks["archive_processed_files"].upstream_task_ids

        # Train depends on Archive (ensures data is moved only after features extracted)
        assert "archive_processed_files" in tasks["train_model"].upstream_task_ids

        # Validate depends on Train
        assert "train_model" in tasks["validate_model"].upstream_task_ids

        # Deploy depends on Validate
        assert "validate_model" in tasks["deploy_model"].upstream_task_ids
