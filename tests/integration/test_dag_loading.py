"""
Integration test for Airflow DAG loading.

This test verifies that the DAG file can be parsed and loaded without
ImportErrors or syntax issues. This is critical for CI/CD pipelines.
"""
import pytest
from pathlib import Path


class TestDAGIntegrity:
    """Tests to ensure DAG files are valid and loadable."""
    
    def test_dag_file_exists(self):
        """Verify the DAG file exists in the expected location."""
        dag_path = Path(__file__).parent.parent.parent / "dags" / "qe_pipeline.py"
        assert dag_path.exists(), f"DAG file not found at {dag_path}"
    
    def test_dag_imports_successfully(self):
        """Test that the DAG module can be imported without errors."""
        import sys
        from pathlib import Path
        
        # Add dags directory to path for import
        dags_dir = Path(__file__).parent.parent.parent / "dags"
        sys.path.insert(0, str(dags_dir))
        
        try:
            # This will fail if there are import errors in the DAG
            import qe_pipeline
            
            # Verify the DAG object exists
            assert hasattr(qe_pipeline, "dag") or hasattr(qe_pipeline, "asr_qe_cumulative_retraining"), \
                "DAG object not found in module"
        finally:
            # Clean up path
            sys.path.remove(str(dags_dir))
    
    def test_dag_has_required_tasks(self):
        """Test that the DAG contains the expected tasks."""
        import sys
        from pathlib import Path
        
        dags_dir = Path(__file__).parent.parent.parent / "dags"
        sys.path.insert(0, str(dags_dir))
        
        try:
            import qe_pipeline
            
            # Get the DAG - it might be defined differently
            dag = getattr(qe_pipeline, "dag", None)
            if dag is None:
                # Check for the DAG in globals
                for name, obj in vars(qe_pipeline).items():
                    if hasattr(obj, "task_dict"):
                        dag = obj
                        break
            
            if dag is not None:
                task_ids = list(dag.task_dict.keys())
                
                # Check for expected tasks (must match actual task_ids in qe_pipeline.py)
                expected_tasks = ["wait_for_incoming_data", "ingest_data", "train_model", "validate_model", "deploy_model"]
                for task in expected_tasks:
                    assert task in task_ids, f"Expected task '{task}' not found in DAG. Found: {task_ids}"
        finally:
            sys.path.remove(str(dags_dir))
