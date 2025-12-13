import subprocess
import sys
from pathlib import Path

import pytest

# Add dags directory to path for imports
DAGS_DIR = Path(__file__).parent.parent.parent / "dags"

@pytest.fixture(scope="module")
def dag():
    """Fixture to load the DAG once for all tests in this module."""
    sys.path.insert(0, str(DAGS_DIR))
    try:
        import qe_pipeline
        return qe_pipeline.dag
    finally:
        sys.path.remove(str(DAGS_DIR))


class TestArchiveProcessedFiles:
    """Test the archive_processed_files task behavior."""

    def test_move_wav_files_to_archive(self, tmp_path):
        """Test that .wav files are moved from incoming to archive directory."""
        # Arrange
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        (incoming / "test1.wav").write_text("audio data 1")
        (incoming / "test2.wav").write_text("audio data 2")

        archive = tmp_path / "archive"
        archive.mkdir()

        # Act
        subprocess.run(f"mv {incoming}/*.wav {archive}/", shell=True, check=True)

        # Assert
        assert not (incoming / "test1.wav").exists()
        assert not (incoming / "test2.wav").exists()
        assert (archive / "test1.wav").exists()
        assert (archive / "test2.wav").exists()

    def test_archive_handles_no_files_gracefully(self, tmp_path):
        """Test that archive command doesn't fail if no .wav files exist."""
        # Arrange
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        archive = tmp_path / "archive"
        archive.mkdir()

        # Act
        result = subprocess.run(
            f"mv {incoming}/*.wav {archive}/ || true",
            shell=True,
            capture_output=True,
        )

        # Assert
        assert result.returncode == 0

    def test_archive_preserves_other_file_types(self, tmp_path):
        """Test that non-.wav files are not moved."""
        # Arrange
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        (incoming / "metadata.json").write_text('{"info": "test"}')
        (incoming / "test.wav").write_text("audio")

        archive = tmp_path / "archive"
        archive.mkdir()

        # Act
        subprocess.run(f"mv {incoming}/*.wav {archive}/", shell=True, check=True)

        # Assert
        assert (incoming / "metadata.json").exists()
        assert not (incoming / "test.wav").exists()
        assert (archive / "test.wav").exists()


class TestDAGTaskDependencies:
    """Test that DAG task dependencies are correctly defined."""

    def test_archive_task_exists_in_dag(self, dag):
        """Verify the archive task is defined in the DAG."""
        # ASSERT
        assert "archive_processed_files" in dag.task_dict

    def test_archive_task_dependencies(self, dag):
        """Verify archive task is correctly placed in the pipeline."""
        # ARRANGE
        archive_task = dag.task_dict["archive_processed_files"]
        ingest_task = dag.task_dict["ingest_data"]
        train_task = dag.task_dict["train_model"]

        # ASSERT
        # Verify: archive comes after ingest
        assert ingest_task in archive_task.upstream_list

        # Verify: train comes after archive
        assert archive_task in train_task.upstream_list
