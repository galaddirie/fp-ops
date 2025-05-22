import pytest
from typing import Optional, Dict, List, Any
import inspect
import asyncio

from fp_ops.primitives import (
    Placeholder, 
    Template, 
    OpSpec, 
    EdgeType, 
    PortType, 
    Port, 
    Edge,
    _
)
from fp_ops.context import BaseContext

class TestPlaceholder:
    
    def test_singleton(self):
        # Test that Placeholder is a singleton
        p1 = Placeholder()
        p2 = Placeholder()
        assert p1 is p2
        assert _ is p1
    
    def test_repr(self):
        # Test the string representation
        assert repr(_) == "_"

class TestContext(BaseContext):
    value: str = ""

class TestTemplate:
    
    def test_empty_template(self):
        # Test an empty template
        t = Template()
        assert t.args == ()
        assert t.kwargs == {}
        assert not t.has_placeholders()
        assert not t.is_identity()
        
        # Rendering an empty template should return empty args and kwargs
        args, kwargs = t.render("value")
        assert args == ()
        assert kwargs == {}
    
    def test_identity_template(self):
        # Test an identity template (single _ placeholder)
        t = Template(args=(_,))
        assert t.has_placeholders()
        assert t.is_identity()
        
        # Rendering an identity template should return the value
        args, kwargs = t.render("value")
        assert args == ("value",)
        assert kwargs == {}
    
    def test_complex_args(self):
        # Test template with complex args
        t = Template(args=(_, "static", 123))
        assert t.has_placeholders()
        assert not t.is_identity()
        
        # Rendering should replace _ with the value
        args, kwargs = t.render("dynamic")
        assert args == ("dynamic", "static", 123)
        assert kwargs == {}
    
    def test_kwargs(self):
        # Test template with kwargs
        t = Template(kwargs={"dynamic": _, "static": "value"})
        assert t.has_placeholders()
        assert not t.is_identity()
        
        # Rendering should replace _ with the value
        args, kwargs = t.render(42)
        assert args == ()
        assert kwargs == {"dynamic": 42, "static": "value"}
    
    def test_mixed_args_kwargs(self):
        # Test template with both args and kwargs
        t = Template(args=(_, "second"), kwargs={"key": _})
        assert t.has_placeholders()
        assert not t.is_identity()
        
        # Rendering should replace _ with the value in both places
        args, kwargs = t.render("value")
        assert args == ("value", "second")
        assert kwargs == {"key": "value"}
    
    def test_nested_placeholders(self):
        # Test template with nested placeholders in data structures
        t = Template(
            args=([1, _, 3], {"a": _}),
            kwargs={"list": [_, 2, 3], "dict": {"nested": _}}
        )
        assert t.has_placeholders()
        assert t._deep  # Should detect nested placeholders
        
        # Rendering should replace all _ instances
        args, kwargs = t.render("X")
        assert args == ([1, "X", 3], {"a": "X"})
        assert kwargs == {"list": ["X", 2, 3], "dict": {"nested": "X"}}
    
    def test_no_placeholders(self):
        # Test template with no placeholders
        t = Template(args=(1, 2, 3), kwargs={"a": "b"})
        assert not t.has_placeholders()
        assert not t.is_identity()
        
        # Rendering should return the original args and kwargs
        args, kwargs = t.render("ignored")
        assert args == (1, 2, 3)
        assert kwargs == {"a": "b"}

class TestOpSpec:
    
    @pytest.fixture
    def async_func(self):
        async def func(a: int, b: str, context: Optional[BaseContext] = None) -> str:
            return f"{a}-{b}"
        return func
    
    def test_create_opspec(self, async_func):
        # Create an OpSpec
        sig = inspect.signature(async_func)
        spec = OpSpec(
            id="test_op",
            func=async_func,
            signature=sig,
            ctx_type=TestContext,
            require_ctx=True,
            template=Template(args=(1, "test"))
        )
        
        # Check properties
        assert spec.id == "test_op"
        assert spec.func is async_func
        assert spec.signature is sig
        assert spec.ctx_type is TestContext
        assert spec.require_ctx is True
        assert isinstance(spec.template, Template)
    
    def test_params_property(self, async_func):
        # Test params property
        sig = inspect.signature(async_func)
        spec = OpSpec(
            id="test_op",
            func=async_func,
            signature=sig,
            ctx_type=None
        )
        
        # Should include all parameters including context
        assert spec.params == ["a", "b", "context"]
    
    def test_non_context_params_property(self, async_func):
        # Test non_context_params property
        sig = inspect.signature(async_func)
        spec = OpSpec(
            id="test_op",
            func=async_func,
            signature=sig,
            ctx_type=None
        )
        
        # Should exclude context parameter
        assert spec.non_context_params == ["a", "b"]

class TestEdgeAndPort:
    
    def test_edge_types(self):
        # Test EdgeType enum
        assert EdgeType.RESULT.value == "result"
        assert EdgeType.ERROR.value == "error"
        assert EdgeType.CONTEXT.value == "context"
    
    def test_port_types(self):
        # Test PortType enum
        assert PortType.TARGET.value == "target"
        assert PortType.SOURCE.value == "source"
    
    def test_port(self):
        # Test Port dataclass
        port = Port(
            node_id="node1",
            port_type=PortType.SOURCE,
            name="output",
            optional=True,
            default="default"
        )
        
        assert port.node_id == "node1"
        assert port.port_type == PortType.SOURCE
        assert port.name == "output"
        assert port.optional is True
        assert port.default == "default"
        
        # Test with defaults
        minimal_port = Port(
            node_id="node2",
            port_type=PortType.TARGET
        )
        
        assert minimal_port.node_id == "node2"
        assert minimal_port.port_type == PortType.TARGET
        assert minimal_port.name is None
        assert minimal_port.optional is False
        assert minimal_port.default is None
    
    def test_edge(self):
        # Test Edge dataclass
        source = Port(node_id="node1", port_type=PortType.SOURCE)
        target = Port(node_id="node2", port_type=PortType.TARGET)
        
        async def transform(x):
            return x * 2
        
        edge = Edge(
            source=source,
            target=target,
            type=EdgeType.RESULT,
            transform=transform
        )
        
        assert edge.source is source
        assert edge.target is target
        assert edge.type == EdgeType.RESULT
        assert edge.transform is transform
        
        # Test with default type
        edge_default = Edge(source=source, target=target)
        assert edge_default.type == EdgeType.RESULT
        assert edge_default.transform is None 