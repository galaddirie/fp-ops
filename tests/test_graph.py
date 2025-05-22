import pytest
from typing import List

from fp_ops.graph import OpGraph
from fp_ops.primitives import OpSpec, Edge, Port, PortType
from fp_ops.primitives import Template


class TestOpGraph:
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph with two nodes and one edge for testing."""
        from inspect import signature
        
        async def dummy_func(x):
            return x
        
        graph = OpGraph()
        
        # Create two nodes
        spec1 = OpSpec(
            id="node1",
            func=dummy_func,
            signature=signature(dummy_func),
            template=Template(),
            ctx_type=None,
            require_ctx=False
        )
        
        spec2 = OpSpec(
            id="node2",
            func=dummy_func,
            signature=signature(dummy_func),
            template=Template(),
            ctx_type=None,
            require_ctx=False
        )
        
        graph.add_node(spec1)
        graph.add_node(spec2)
        
        # Create an edge from node1 to node2
        edge = Edge(
            source=Port(node_id="node1", port_type=PortType.SOURCE, name="result"),
            target=Port(node_id="node2", port_type=PortType.TARGET, name=None)
        )
        
        graph.add_edge(edge)
        
        return graph
    
    def test_outgoing_with_edges(self, simple_graph):
        """Test the outgoing method when edges exist."""
        edges = simple_graph.outgoing("node1")
        
        # Check that we get a tuple
        assert isinstance(edges, tuple)
        
        # Check that we have exactly one edge
        assert len(edges) == 1
        
        # Check the edge properties
        edge = edges[0]
        assert edge.source.node_id == "node1"
        assert edge.target.node_id == "node2"
    
    def test_outgoing_no_edges(self, simple_graph):
        """Test the outgoing method when no edges exist for the node."""
        # node2 has no outgoing edges
        edges = simple_graph.outgoing("node2")
        
        # Check that we get an empty tuple
        assert isinstance(edges, tuple)
        assert len(edges) == 0
    
    def test_outgoing_nonexistent_node(self, simple_graph):
        """Test the outgoing method for a node that doesn't exist."""
        edges = simple_graph.outgoing("nonexistent_node")
        
        # Should return an empty tuple, not raise an exception
        assert isinstance(edges, tuple)
        assert len(edges) == 0 