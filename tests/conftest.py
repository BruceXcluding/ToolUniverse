import os
import pytest
import warnings


@pytest.fixture(scope="session", autouse=True)
def _set_test_env() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    # Avoid accidental network in unit tests unless explicitly marked
    os.environ.setdefault("TOOLUNIVERSE_TESTING", "1")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: slow tests (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "require_api_keys: tests requiring API keys")
    config.addinivalue_line("markers", "manual: manual tests (not run in CI)")
    config.addinivalue_line("markers", "unit: unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: integration tests (may use network)")


def pytest_collection_modifyitems(items):
    """Ensure test quality by checking for required elements."""
    for item in items:
        # Check for docstring
        if not item.function.__doc__:
            warnings.warn(
                f"Test {item.nodeid} missing docstring - consider adding one",
                category=UserWarning
            )
        
        # Check for appropriate markers
        marks = [m.name for m in item.iter_markers()]
        if not any(m in marks for m in ['unit', 'integration', 'slow', 'manual']):
            warnings.warn(
                f"Test {item.nodeid} missing category marker (unit/integration/slow/manual)",
                category=UserWarning
            )
        
        # Check for meaningful test names
        if not any(keyword in item.name.lower() for keyword in ['test_', 'check_', 'verify_']):
            warnings.warn(
                f"Test {item.nodeid} may not follow naming convention (should start with test_)",
                category=UserWarning
            )


@pytest.fixture(scope="session")
def tools_generated():
    """Ensure tools are generated before running tests."""
    from pathlib import Path
    tools_dir = Path("src/tooluniverse/tools")
    if not tools_dir.exists() or not any(tools_dir.glob("*.py")):
        pytest.fail("Tools not generated. Run: python scripts/build_tools.py")
    return tools_dir


@pytest.fixture(scope="session")
def tooluniverse_instance():
    """Session-scoped ToolUniverse instance for better performance."""
    from tooluniverse import ToolUniverse
    tu = ToolUniverse()
    tu.load_tools()
    return tu


@pytest.fixture
def disable_network(monkeypatch: pytest.MonkeyPatch):
    """Disable network by patching requests' adapters. Use for unit tests."""
    import requests

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Network disabled in unit test. Use @pytest.mark.integration for network tests.")

    monkeypatch.setattr(requests.sessions.Session, "request", _raise)
    return None


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


