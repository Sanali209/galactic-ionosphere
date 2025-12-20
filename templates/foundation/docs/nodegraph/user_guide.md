# NodeGraph User Guide

Visual programming for the Foundation template.

## Getting Started

Launch the Node Editor:
```bash
python samples/node_editor/main.py
```

## Interface

### Canvas (Center)
- **Pan**: Middle-mouse drag or Space+drag
- **Zoom**: Mouse wheel
- **Select**: Click nodes
- **Multi-select**: Ctrl+Click or box select

### Node Palette (Left)
Browse nodes by category. Double-click or drag to add.

### Properties (Right)
Edit values for selected node.

### Execution Log (Bottom)  
View output and errors when running.

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New | Ctrl+N |
| Open | Ctrl+O |
| Save | Ctrl+S |
| Copy | Ctrl+C |
| Cut | Ctrl+X |
| Paste | Ctrl+V |
| Duplicate | Ctrl+D |
| Delete | Delete |
| Select All | Ctrl+A |
| Fit View | F |
| Run | F5 |
| Stop | Escape |

## Creating Connections

1. Click on an output pin (right side)
2. Drag to an input pin (left side)
3. Release to connect

Only compatible pins can be connected:
- Execution → Execution (white)
- Same data types (matching colors)

## Execution Flow

Graphs run from **Start** nodes, following white execution wires.

### Example: Hello World
```
Start → Print("Hello") → Print("World")
```

### Example: Loop Counter
```
Start → Set counter=0 → ForLoop(1,5) → Increment counter → Print
                                    ↓
                              (on completed) → Print("Done!")
```

## Node Categories

### Events (2)
- **Start**: Entry point
- **Update**: Tick handler

### Flow Control (11)
- **Branch**: If/else
- **For Loop**: Count-based iteration
- **For Each Loop**: Array iteration
- **While Loop**: Condition-based loop
- **Sequence**: Multi-output chain
- **Gate**: Open/close flow
- **Do Once**: Execute once per run
- **Flip Flop**: Toggle between outputs

### Variables (4)
- **Get Variable**: Read value
- **Set Variable**: Write value
- **Increment Variable**: Add to value
- **Is Valid**: Check if set

### Utilities (8)
- **Print**: Output to log
- **Comment**: Notes (not executed)
- **MakeArray**: Create array
- **ToString/ToInteger/ToFloat/ToBoolean**: Type conversions

### File (11)
- **Read/Write File**: Text file I/O
- **List Directory**: Get file list
- **Copy/Move/Delete File**: File operations
- **Create Directory**: Make folder
- **Get File Info**: Size, date, name

### String (11)
- **Concat**: Join strings
- **Split**: Break apart
- **Replace**: Find and replace
- **Format**: Template substitution
- **Length/Contains/StartsWith/EndsWith**: Queries
- **Trim/Upper/Lower**: Transformations

### Array (11)
- **Join**: Array to string
- **Get/Set**: Index access
- **Append/Merge**: Add elements
- **Filter**: Pattern matching
- **Sort/Reverse/Unique**: Transformations
- **Slice**: Subset

### Image (9)
Requires Pillow library.
- **Load/Save Image**: File I/O
- **Resize/Crop/Rotate/Flip**: Transforms
- **Image Info**: Get dimensions

### Matplotlib (11)
Requires Matplotlib library.
- **Create Figure**: New chart
- **Plot Line/Bar/Scatter/Histogram/Pie**: Chart types
- **Set Axis Labels/Legend/Grid**: Styling
- **Save Figure**: Export to file

## Tips

1. **Use Comments**: Add Comment nodes to document sections
2. **Name Variables**: Use descriptive names
3. **Check Log**: Errors show in Execution Log
4. **Save Often**: Use Ctrl+S frequently
5. **Test Incrementally**: Run after adding each feature
