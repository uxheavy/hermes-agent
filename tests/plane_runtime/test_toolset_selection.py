from types import SimpleNamespace

from plane_runtime.hermes_adapter import _plane_model_toolsets
from plane_runtime.host_port import (
    PLANE_CODE_MODE_TOOL,
    PLANE_CODE_MODE_TOOLSET,
    PLANE_OPERATION_TOOL,
    PLANE_OPERATION_TOOLSET,
    PLANE_PUBLISH_TOOL,
    PLANE_PUBLICATION_TOOLSET,
    install_plane_tools,
)
from tools.registry import registry
from toolsets import resolve_multiple_toolsets


def test_code_mode_snapshot_selects_only_code_and_publication_tools():
    standard = SimpleNamespace(model_toolset="standard")
    code_mode = SimpleNamespace(model_toolset="code_mode_only")
    assert _plane_model_toolsets(standard) == (PLANE_OPERATION_TOOLSET, PLANE_PUBLICATION_TOOLSET, PLANE_CODE_MODE_TOOLSET)
    assert _plane_model_toolsets(code_mode) == (PLANE_PUBLICATION_TOOLSET, PLANE_CODE_MODE_TOOLSET)

    install_plane_tools()
    standard_tools = set(resolve_multiple_toolsets(list(_plane_model_toolsets(standard))))
    code_mode_tools = set(resolve_multiple_toolsets(list(_plane_model_toolsets(code_mode))))
    assert {PLANE_OPERATION_TOOL, PLANE_PUBLISH_TOOL}.issubset(standard_tools)
    assert code_mode_tools == {PLANE_CODE_MODE_TOOL, PLANE_PUBLISH_TOOL}
    assert PLANE_OPERATION_TOOL not in code_mode_tools
    assert registry.get_entry(PLANE_OPERATION_TOOL).toolset == PLANE_OPERATION_TOOLSET
    assert registry.get_entry(PLANE_PUBLISH_TOOL).toolset == PLANE_PUBLICATION_TOOLSET
