# -*- coding: utf-8 -*-
"""
Image Nodes - Image manipulation using Pillow.

Provides nodes for:
- Loading/saving images
- Resize, crop, rotate
- Format conversion
- Image info
"""
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class LoadImageNode(BaseNode):
    """Load image from file."""
    node_type = "LoadImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Load Image",
        description="Load image from file",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("image", PinType.ANY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("width", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("height", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class SaveImageNode(BaseNode):
    """Save image to file."""
    node_type = "SaveImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Save Image",
        description="Save image to file",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("quality", PinType.INTEGER, default_value=95))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class ResizeImageNode(BaseNode):
    """Resize image to new dimensions."""
    node_type = "ResizeImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Resize Image",
        description="Resize image to width x height",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_input_pin(DataPin("width", PinType.INTEGER, default_value=800))
        self.add_input_pin(DataPin("height", PinType.INTEGER, default_value=600))
        self.add_input_pin(DataPin("keep_aspect", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("result", PinType.ANY, PinDirection.OUTPUT))


class CropImageNode(BaseNode):
    """Crop image to rectangle."""
    node_type = "CropImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Crop Image",
        description="Crop image to rectangle (x, y, width, height)",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_input_pin(DataPin("x", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("y", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("width", PinType.INTEGER, default_value=100))
        self.add_input_pin(DataPin("height", PinType.INTEGER, default_value=100))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("result", PinType.ANY, PinDirection.OUTPUT))


class RotateImageNode(BaseNode):
    """Rotate image by angle."""
    node_type = "RotateImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Rotate Image",
        description="Rotate image by degrees",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_input_pin(DataPin("angle", PinType.FLOAT, default_value=90.0))
        self.add_input_pin(DataPin("expand", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("result", PinType.ANY, PinDirection.OUTPUT))


class ConvertImageNode(BaseNode):
    """Convert image format/mode."""
    node_type = "ConvertImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Convert Image",
        description="Convert image mode (RGB, RGBA, L, etc.)",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_input_pin(DataPin("mode", PinType.STRING, default_value="RGB"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("result", PinType.ANY, PinDirection.OUTPUT))


class ImageInfoNode(BaseNode):
    """Get image information."""
    node_type = "ImageInfo"
    metadata = NodeMetadata(
        category="Image",
        display_name="Image Info",
        description="Get image dimensions and format",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_output_pin(DataPin("width", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("height", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("mode", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("format", PinType.STRING, PinDirection.OUTPUT))


class FlipImageNode(BaseNode):
    """Flip image horizontally or vertically."""
    node_type = "FlipImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Flip Image",
        description="Flip image horizontally or vertically",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("image", PinType.ANY))
        self.add_input_pin(DataPin("horizontal", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("result", PinType.ANY, PinDirection.OUTPUT))


class CreateImageNode(BaseNode):
    """Create new blank image."""
    node_type = "CreateImage"
    metadata = NodeMetadata(
        category="Image",
        display_name="Create Image",
        description="Create blank image with color",
        color="#4169E1"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("width", PinType.INTEGER, default_value=800))
        self.add_input_pin(DataPin("height", PinType.INTEGER, default_value=600))
        self.add_input_pin(DataPin("color", PinType.STRING, default_value="#FFFFFF"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("image", PinType.ANY, PinDirection.OUTPUT))


# Export all nodes
ALL_NODES = [
    LoadImageNode,
    SaveImageNode,
    ResizeImageNode,
    CropImageNode,
    RotateImageNode,
    ConvertImageNode,
    ImageInfoNode,
    FlipImageNode,
    CreateImageNode,
]
