"""Tool registry — smoke tests."""

import pytest
from pydantic import BaseModel

from samhita.core.tools import (
    Tool,
    clear_registry,
    get_tool,
    list_tools,
    register_tool,
)


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    r: str


async def _echo(payload: _In) -> _Out:
    return _Out(r=payload.q)


def test_register_and_get_tool() -> None:
    clear_registry()
    tool = Tool(
        name="echo",
        description="round-trip a string",
        input_schema=_In,
        output_schema=_Out,
        func=_echo,
        tags=["test"],
    )
    register_tool(tool)
    fetched = get_tool("echo")
    assert fetched.name == "echo"
    assert "test" in fetched.tags


def test_duplicate_registration_raises() -> None:
    clear_registry()
    tool = Tool(
        name="echo",
        description="",
        input_schema=_In,
        output_schema=_Out,
        func=_echo,
    )
    register_tool(tool)
    with pytest.raises(ValueError):
        register_tool(tool)


def test_unknown_tool_raises() -> None:
    clear_registry()
    with pytest.raises(KeyError):
        get_tool("does_not_exist")


def test_list_tools_by_tag() -> None:
    clear_registry()
    register_tool(
        Tool(
            name="a",
            description="",
            input_schema=_In,
            output_schema=_Out,
            func=_echo,
            tags=["fetch"],
        )
    )
    register_tool(
        Tool(
            name="b",
            description="",
            input_schema=_In,
            output_schema=_Out,
            func=_echo,
            tags=["extract"],
        )
    )
    fetch_tools = list_tools(tag="fetch")
    assert [t.name for t in fetch_tools] == ["a"]
