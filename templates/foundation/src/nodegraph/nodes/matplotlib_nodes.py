# -*- coding: utf-8 -*-
"""
Matplotlib Nodes - Data visualization with Matplotlib.

Provides nodes for:
- Creating figures
- Various plot types (line, bar, scatter, pie, histogram)
- Customizing axes and labels
- Displaying and saving figures
"""
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class CreateFigureNode(BaseNode):
    """Create a new matplotlib figure."""
    node_type = "CreateFigure"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Create Figure",
        description="Create new figure with size",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("width", PinType.FLOAT, default_value=10.0))
        self.add_input_pin(DataPin("height", PinType.FLOAT, default_value=6.0))
        self.add_input_pin(DataPin("title", PinType.STRING, default_value=""))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("figure", PinType.ANY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class PlotLineNode(BaseNode):
    """Plot a line chart."""
    node_type = "PlotLine"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Plot Line",
        description="Plot line chart",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("x_data", PinType.ARRAY))
        self.add_input_pin(DataPin("y_data", PinType.ARRAY))
        self.add_input_pin(DataPin("label", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("color", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("line_style", PinType.STRING, default_value="-"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class PlotBarNode(BaseNode):
    """Plot a bar chart."""
    node_type = "PlotBar"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Plot Bar",
        description="Plot bar chart",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("x_data", PinType.ARRAY))
        self.add_input_pin(DataPin("y_data", PinType.ARRAY))
        self.add_input_pin(DataPin("label", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("color", PinType.STRING, default_value=""))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class PlotScatterNode(BaseNode):
    """Plot a scatter chart."""
    node_type = "PlotScatter"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Plot Scatter",
        description="Plot scatter chart",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("x_data", PinType.ARRAY))
        self.add_input_pin(DataPin("y_data", PinType.ARRAY))
        self.add_input_pin(DataPin("label", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("color", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("size", PinType.FLOAT, default_value=20.0))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class PlotHistogramNode(BaseNode):
    """Plot a histogram."""
    node_type = "PlotHistogram"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Plot Histogram",
        description="Plot histogram",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("data", PinType.ARRAY))
        self.add_input_pin(DataPin("bins", PinType.INTEGER, default_value=10))
        self.add_input_pin(DataPin("label", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("color", PinType.STRING, default_value=""))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class PlotPieNode(BaseNode):
    """Plot a pie chart."""
    node_type = "PlotPie"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Plot Pie",
        description="Plot pie chart",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("values", PinType.ARRAY))
        self.add_input_pin(DataPin("labels", PinType.ARRAY))
        self.add_input_pin(DataPin("autopct", PinType.STRING, default_value="%1.1f%%"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class SetAxisLabelsNode(BaseNode):
    """Set axis labels and title."""
    node_type = "SetAxisLabels"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Set Axis Labels",
        description="Set x/y labels and title",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("x_label", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("y_label", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("title", PinType.STRING, default_value=""))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class ShowFigureNode(BaseNode):
    """Display figure in popup window."""
    node_type = "ShowFigure"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Show Figure",
        description="Display figure in popup",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("figure", PinType.ANY))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))


class SaveFigureNode(BaseNode):
    """Save figure to file."""
    node_type = "SaveFigure"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Save Figure",
        description="Save figure as PNG/PDF/SVG",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("figure", PinType.ANY))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("dpi", PinType.INTEGER, default_value=150))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class AddLegendNode(BaseNode):
    """Add legend to figure."""
    node_type = "AddLegend"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Add Legend",
        description="Add legend to axes",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("location", PinType.STRING, default_value="best"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


class SetGridNode(BaseNode):
    """Enable/disable grid."""
    node_type = "SetGrid"
    metadata = NodeMetadata(
        category="Matplotlib",
        display_name="Set Grid",
        description="Enable or disable grid",
        color="#11557C"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("axes", PinType.ANY))
        self.add_input_pin(DataPin("enabled", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("axes", PinType.ANY, PinDirection.OUTPUT))


# Export all nodes
ALL_NODES = [
    CreateFigureNode,
    PlotLineNode,
    PlotBarNode,
    PlotScatterNode,
    PlotHistogramNode,
    PlotPieNode,
    SetAxisLabelsNode,
    ShowFigureNode,
    SaveFigureNode,
    AddLegendNode,
    SetGridNode,
]
