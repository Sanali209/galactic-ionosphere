# -*- coding: utf-8 -*-
"""
Matplotlib Node Executors - Execution logic for data visualization.
"""
from .node_executor import BaseNodeExecutor, register_executor

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CreateFigureExecutor(BaseNodeExecutor):
    """Create matplotlib figure."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            context.log(node, "matplotlib not installed", "ERROR")
            return
        
        width = executor.evaluate_input(node, "width") or 10.0
        height = executor.evaluate_input(node, "height") or 6.0
        title = executor.evaluate_input(node, "title") or ""
        
        fig, ax = plt.subplots(figsize=(float(width), float(height)))
        
        if title:
            fig.suptitle(title)
        
        node.set_output("figure", fig)
        node.set_output("axes", ax)
        context.log(node, f"Created {width}x{height} figure")
        
        await executor.execute_output_pin(node, "exec_out")


class PlotLineExecutor(BaseNodeExecutor):
    """Plot line chart."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        x_data = executor.evaluate_input(node, "x_data") or []
        y_data = executor.evaluate_input(node, "y_data") or []
        label = executor.evaluate_input(node, "label") or ""
        color = executor.evaluate_input(node, "color") or None
        line_style = executor.evaluate_input(node, "line_style") or "-"
        
        if ax is None:
            node.set_output("axes", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        kwargs = {"linestyle": line_style}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        ax.plot(x_data, y_data, **kwargs)
        
        node.set_output("axes", ax)
        context.log(node, f"Plotted line with {len(y_data)} points")
        
        await executor.execute_output_pin(node, "exec_out")


class PlotBarExecutor(BaseNodeExecutor):
    """Plot bar chart."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        x_data = executor.evaluate_input(node, "x_data") or []
        y_data = executor.evaluate_input(node, "y_data") or []
        label = executor.evaluate_input(node, "label") or ""
        color = executor.evaluate_input(node, "color") or None
        
        if ax is None:
            node.set_output("axes", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        kwargs = {}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        ax.bar(x_data, y_data, **kwargs)
        
        node.set_output("axes", ax)
        context.log(node, f"Plotted bar chart with {len(y_data)} bars")
        
        await executor.execute_output_pin(node, "exec_out")


class PlotScatterExecutor(BaseNodeExecutor):
    """Plot scatter chart."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        x_data = executor.evaluate_input(node, "x_data") or []
        y_data = executor.evaluate_input(node, "y_data") or []
        label = executor.evaluate_input(node, "label") or ""
        color = executor.evaluate_input(node, "color") or None
        size = executor.evaluate_input(node, "size") or 20.0
        
        if ax is None:
            node.set_output("axes", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        kwargs = {"s": float(size)}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["c"] = color
        
        ax.scatter(x_data, y_data, **kwargs)
        
        node.set_output("axes", ax)
        context.log(node, f"Plotted scatter with {len(y_data)} points")
        
        await executor.execute_output_pin(node, "exec_out")


class PlotHistogramExecutor(BaseNodeExecutor):
    """Plot histogram."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        data = executor.evaluate_input(node, "data") or []
        bins = executor.evaluate_input(node, "bins") or 10
        label = executor.evaluate_input(node, "label") or ""
        color = executor.evaluate_input(node, "color") or None
        
        if ax is None:
            node.set_output("axes", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        kwargs = {"bins": int(bins)}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        ax.hist(data, **kwargs)
        
        node.set_output("axes", ax)
        context.log(node, f"Plotted histogram with {len(data)} values")
        
        await executor.execute_output_pin(node, "exec_out")


class PlotPieExecutor(BaseNodeExecutor):
    """Plot pie chart."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        values = executor.evaluate_input(node, "values") or []
        labels = executor.evaluate_input(node, "labels") or []
        autopct = executor.evaluate_input(node, "autopct") or "%1.1f%%"
        
        if ax is None:
            node.set_output("axes", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        ax.pie(values, labels=labels if labels else None, autopct=autopct)
        
        node.set_output("axes", ax)
        context.log(node, f"Plotted pie chart with {len(values)} slices")
        
        await executor.execute_output_pin(node, "exec_out")


class SetAxisLabelsExecutor(BaseNodeExecutor):
    """Set axis labels."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        x_label = executor.evaluate_input(node, "x_label") or ""
        y_label = executor.evaluate_input(node, "y_label") or ""
        title = executor.evaluate_input(node, "title") or ""
        
        if ax is None:
            node.set_output("axes", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        
        node.set_output("axes", ax)
        
        await executor.execute_output_pin(node, "exec_out")


class ShowFigureExecutor(BaseNodeExecutor):
    """Show figure (for interactive mode)."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig = executor.evaluate_input(node, "figure")
        
        if fig is not None:
            # In interactive mode, this would show the figure
            # For now, just log
            context.log(node, "Figure ready (use SaveFigure to export)")
        
        await executor.execute_output_pin(node, "exec_out")


class SaveFigureExecutor(BaseNodeExecutor):
    """Save figure to file."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            node.set_output("error", "matplotlib not installed")
            await executor.execute_output_pin(node, "failed")
            return
        
        fig = executor.evaluate_input(node, "figure")
        path = executor.evaluate_input(node, "path") or ""
        dpi = executor.evaluate_input(node, "dpi") or 150
        
        try:
            if fig is None:
                raise ValueError("No figure provided")
            
            fig.savefig(path, dpi=int(dpi), bbox_inches='tight')
            node.set_output("error", "")
            context.log(node, f"Saved figure to {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class AddLegendExecutor(BaseNodeExecutor):
    """Add legend."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        location = executor.evaluate_input(node, "location") or "best"
        
        if ax is not None:
            ax.legend(loc=location)
        
        node.set_output("axes", ax)
        
        await executor.execute_output_pin(node, "exec_out")


class SetGridExecutor(BaseNodeExecutor):
    """Set grid."""
    
    async def execute(self, node, context, executor):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = executor.evaluate_input(node, "axes")
        enabled = executor.evaluate_input(node, "enabled")
        if enabled is None:
            enabled = True
        
        if ax is not None:
            ax.grid(enabled)
        
        node.set_output("axes", ax)
        
        await executor.execute_output_pin(node, "exec_out")


def register_matplotlib_executors():
    """Register all matplotlib executors."""
    register_executor("CreateFigure", CreateFigureExecutor())
    register_executor("PlotLine", PlotLineExecutor())
    register_executor("PlotBar", PlotBarExecutor())
    register_executor("PlotScatter", PlotScatterExecutor())
    register_executor("PlotHistogram", PlotHistogramExecutor())
    register_executor("PlotPie", PlotPieExecutor())
    register_executor("SetAxisLabels", SetAxisLabelsExecutor())
    register_executor("ShowFigure", ShowFigureExecutor())
    register_executor("SaveFigure", SaveFigureExecutor())
    register_executor("AddLegend", AddLegendExecutor())
    register_executor("SetGrid", SetGridExecutor())
