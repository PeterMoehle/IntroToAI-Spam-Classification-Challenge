import numpy as np

from typing import List, Optional

from graph import Node
from graph.nodes import FunctionNode


class DigraphGenerator:
    """Handles all digraph generation and formatting for computational graphs."""

    def __init__(self):
        pass

    def _format_value_summary(self, node: Node) -> str:
        """Return a formatted summary of the node's value."""
        if node.value is None:
            return "None"
        if node.value.size == 1:
            return f"{node.value.item():.4f}"
        elif node.value.size <= 4:
            formatted_values = [f"{val:.4f}" for val in node.value.flatten()]
            return f"[{', '.join(formatted_values)}]"
        else:
            return f"Array{node.value.shape}"

    def _format_gradient_summary(self, node: Node) -> str:
        """Return a formatted summary of the node's gradient."""
        if node.gradient is None:
            return "None"
        if node.gradient.size == 1:
            return f"{node.gradient.item():.4f}"
        elif node.gradient.size <= 4:
            formatted_gradients = [f"{grad:.4f}" for grad in node.gradient.flatten()]
            return f"[{', '.join(formatted_gradients)}]"
        else:
            return f"Grad{node.gradient.shape}"

    def _format_edge_label(self, from_node: Node, to_node: Node) -> str:
        """Format edge label with value and gradient information."""
        label_parts = []

        # Add value information
        if from_node.value is not None:
            value_str = self._format_value_summary(from_node)
            label_parts.append(f"val (alpha{from_node.id}): {value_str}")

        # Add gradient information (only if gradient exists and is not None/zero)
        if (from_node.gradient is not None and
                not np.allclose(from_node.gradient, 0) and
                not np.all(from_node.gradient == 0)):
            grad_str = self._format_gradient_summary(from_node)
            label_parts.append(f"grad (beta{from_node.id}): {grad_str}")

        if label_parts:
            return f'label="{" | ".join(label_parts)}"'
        else:
            return ""

    def generate_digraph(self, nodes: List[Node], title: Optional[str] = None) -> str:
        """Generate DOT format digraph with values and gradients in edges."""
        dot_lines = []
        dot_lines.append("digraph ComputationalGraph {")
        dot_lines.append("    rankdir=LR;")
        dot_lines.append("    edge [fontsize=10];")

        if title:
            dot_lines.append(f'    labelloc="t";')
            dot_lines.append(f'    label="{title}";')

        # Add edges with value and gradient information
        for node in nodes:
            if isinstance(node, FunctionNode):
                for input_node in node.input_nodes:
                    edge_label = self._format_edge_label(input_node, node)
                    if edge_label:
                        dot_lines.append(f'    {input_node.name} -> {node.name} [{edge_label}];')
                    else:
                        dot_lines.append(f'    {input_node.name} -> {node.name} [];')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def save_digraph(self, nodes: List[Node], filename: str, title: Optional[str] = None) -> None:
        """Save the digraph to a file."""
        dot_content = self.generate_digraph(nodes, title)

        with open(f"{filename}.dot", "w", encoding='utf-8') as f:
            f.write(dot_content)

        print(f"Digraph saved to {filename}.dot")

    def print_digraph(self, nodes: List[Node], title: Optional[str] = None) -> None:
        """Print the digraph to console."""
        dot_content = self.generate_digraph(nodes, title)
        print("\n--- Computational Graph with Values and Gradients ---")
        print(dot_content)
        print("--- End of Graph ---\n")
