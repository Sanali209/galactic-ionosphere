# -*- coding: utf-8 -*-
"""
Image Node Executors - Execution logic for image operations using Pillow.
"""
from .node_executor import BaseNodeExecutor, register_executor

# Check if Pillow is available
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


class LoadImageExecutor(BaseNodeExecutor):
    """Load image from file."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            node.set_output("error", "Pillow not installed")
            await executor.execute_output_pin(node, "failed")
            return
        
        path = executor.evaluate_input(node, "path") or ""
        
        try:
            img = Image.open(path)
            node.set_output("image", img)
            node.set_output("width", img.width)
            node.set_output("height", img.height)
            node.set_output("error", "")
            context.log(node, f"Loaded {img.width}x{img.height} from {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("image", None)
            node.set_output("width", 0)
            node.set_output("height", 0)
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class SaveImageExecutor(BaseNodeExecutor):
    """Save image to file."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            node.set_output("error", "Pillow not installed")
            await executor.execute_output_pin(node, "failed")
            return
        
        img = executor.evaluate_input(node, "image")
        path = executor.evaluate_input(node, "path") or ""
        quality = executor.evaluate_input(node, "quality") or 95
        
        try:
            if img is None:
                raise ValueError("No image provided")
            
            img.save(path, quality=int(quality))
            node.set_output("error", "")
            context.log(node, f"Saved image to {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class ResizeImageExecutor(BaseNodeExecutor):
    """Resize image."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            return
        
        img = executor.evaluate_input(node, "image")
        width = executor.evaluate_input(node, "width") or 800
        height = executor.evaluate_input(node, "height") or 600
        keep_aspect = executor.evaluate_input(node, "keep_aspect")
        if keep_aspect is None:
            keep_aspect = True
        
        if img is None:
            node.set_output("result", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        if keep_aspect:
            img.thumbnail((int(width), int(height)), Image.Resampling.LANCZOS)
            result = img
        else:
            result = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        
        node.set_output("result", result)
        context.log(node, f"Resized to {result.width}x{result.height}")
        
        await executor.execute_output_pin(node, "exec_out")


class CropImageExecutor(BaseNodeExecutor):
    """Crop image."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            return
        
        img = executor.evaluate_input(node, "image")
        x = executor.evaluate_input(node, "x") or 0
        y = executor.evaluate_input(node, "y") or 0
        width = executor.evaluate_input(node, "width") or 100
        height = executor.evaluate_input(node, "height") or 100
        
        if img is None:
            node.set_output("result", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        box = (int(x), int(y), int(x + width), int(y + height))
        result = img.crop(box)
        
        node.set_output("result", result)
        context.log(node, f"Cropped to {result.width}x{result.height}")
        
        await executor.execute_output_pin(node, "exec_out")


class RotateImageExecutor(BaseNodeExecutor):
    """Rotate image."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            return
        
        img = executor.evaluate_input(node, "image")
        angle = executor.evaluate_input(node, "angle") or 90
        expand = executor.evaluate_input(node, "expand")
        if expand is None:
            expand = True
        
        if img is None:
            node.set_output("result", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        result = img.rotate(float(angle), expand=expand, resample=Image.Resampling.BICUBIC)
        
        node.set_output("result", result)
        context.log(node, f"Rotated {angle}°")
        
        await executor.execute_output_pin(node, "exec_out")


class ConvertImageExecutor(BaseNodeExecutor):
    """Convert image mode."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            return
        
        img = executor.evaluate_input(node, "image")
        mode = executor.evaluate_input(node, "mode") or "RGB"
        
        if img is None:
            node.set_output("result", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        result = img.convert(mode)
        
        node.set_output("result", result)
        context.log(node, f"Converted to {mode}")
        
        await executor.execute_output_pin(node, "exec_out")


class ImageInfoExecutor(BaseNodeExecutor):
    """Get image info."""
    
    async def execute(self, node, context, executor):
        img = executor.evaluate_input(node, "image")
        
        if img is None:
            node.set_output("width", 0)
            node.set_output("height", 0)
            node.set_output("mode", "")
            node.set_output("format", "")
            return
        
        node.set_output("width", img.width)
        node.set_output("height", img.height)
        node.set_output("mode", img.mode)
        node.set_output("format", img.format or "")


class FlipImageExecutor(BaseNodeExecutor):
    """Flip image."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            return
        
        img = executor.evaluate_input(node, "image")
        horizontal = executor.evaluate_input(node, "horizontal")
        if horizontal is None:
            horizontal = True
        
        if img is None:
            node.set_output("result", None)
            await executor.execute_output_pin(node, "exec_out")
            return
        
        if horizontal:
            result = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            result = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        node.set_output("result", result)
        context.log(node, f"Flipped {'horizontal' if horizontal else 'vertical'}")
        
        await executor.execute_output_pin(node, "exec_out")


class CreateImageExecutor(BaseNodeExecutor):
    """Create blank image."""
    
    async def execute(self, node, context, executor):
        if not PILLOW_AVAILABLE:
            return
        
        width = executor.evaluate_input(node, "width") or 800
        height = executor.evaluate_input(node, "height") or 600
        color = executor.evaluate_input(node, "color") or "#FFFFFF"
        
        img = Image.new("RGB", (int(width), int(height)), color)
        
        node.set_output("image", img)
        context.log(node, f"Created {width}x{height} image")
        
        await executor.execute_output_pin(node, "exec_out")


def register_image_executors():
    """Register all image executors."""
    register_executor("LoadImage", LoadImageExecutor())
    register_executor("SaveImage", SaveImageExecutor())
    register_executor("ResizeImage", ResizeImageExecutor())
    register_executor("CropImage", CropImageExecutor())
    register_executor("RotateImage", RotateImageExecutor())
    register_executor("ConvertImage", ConvertImageExecutor())
    register_executor("ImageInfo", ImageInfoExecutor())
    register_executor("FlipImage", FlipImageExecutor())
    register_executor("CreateImage", CreateImageExecutor())
