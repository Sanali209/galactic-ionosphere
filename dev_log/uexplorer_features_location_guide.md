# UExplorer - File Count Features Location Guide

**Feature Implementation Locations in UExplorer UI**

---

## 1. Maintenance Menu (Menu Bar)

**Location**: Top menu bar, between "Tools" and "Help"

```
┌─────────────────────────────────────────────────┐
│ File  Edit  View  Tools  [Maintenance]  Help   │
└─────────────────────────────────────────────────┘
                      ↑
                      └── NEW MENU HERE
```

**Menu Structure**:
```
Maintenance
├── 🔄 Rebuild All Counts...      ← Recalculates all counts
├── ─────────────────────
├── 🔍 Verify References...       ← Checks data integrity
└── 🧹 Cleanup Orphaned Records   ← Removes invalid refs
```

**How to access**:
1. Click "Maintenance" in menu bar
2. Select any of the 3 actions
3. Progress dialog will appear
4. Results shown in message box

---

## 2. Tag Context Menu (Right-Click)

**Location**: Tags panel (left sidebar)

```
┌─────────────────┐
│ Tags Panel      │
├─────────────────┤
│ ✓ Nature (42)   │ ← RIGHT-CLICK HERE
│   └─ Animals    │
│   └─ Plants     │
│ ✓ People        │
│ □ Places        │
└─────────────────┘
```

**Context Menu**:
```
Right-click on any tag →
├── ✓ Include in Filter (I)
├── ✗ Exclude from Filter (E)
├── ─────────────────────
├── Add Child Tag
├── ─────────────────────
├── Rename
├── Delete
├── ─────────────────────
└── 🔄 Recalculate Count      ← NEW ACTION
```

**What happens**:
1. Right-click any tag
2. Select "🔄 Recalculate Count"
3. Count recalculated from database
4. Display updates immediately
5. Toast notification (optional)

---

## 3. Album Context Menu (Right-Click)

**Location**: Albums panel (left sidebar tabs)

```
┌─────────────────────┐
│ Albums Panel        │
├─────────────────────┤
│ Favorites (28)      │ ← RIGHT-CLICK HERE
│ 📊 Recent (123)     │ ← Smart album
│ Vacation 2023 (56)  │
└─────────────────────┘
```

**Context Menu**:
```
Right-click on any album →
├── ✓ Include in Filter (I)
├── ✗ Exclude from Filter (E)
├── ─────────────────────
├── Create Album
├── Create Smart Album...
├── ─────────────────────
├── Rename
├── Delete
├── ─────────────────────
└── 🔄 Recalculate Count      ← NEW ACTION
```

**What happens**:
1. Right-click any album
2. Select "🔄 Recalculate Count"
3. For manual albums: counts file_ids
4. For smart albums: executes query
5. Display updates with new count

---

## 4. Real-Time Counts Display

### Tags Panel
```
┌─────────────────────────┐
│ Tags                    │
├─────────────────────────┤
│ Nature (42)             │ ← Count shown here
│   └─ Animals (18)       │
│   └─ Plants (24)        │
│ People (156)            │
│ Urban (89)              │
└─────────────────────────┘
```

### Albums Panel
```
┌─────────────────────────┐
│ Albums                  │
├─────────────────────────┤
│ Favorites (28)          │ ← Manual album count
│ 📊 Untagged (1,234)     │ ← Smart album (real-time)
│ 📊 5-Star (45)          │
│ Vacation 2023 (56)      │
└─────────────────────────┘
```

**Smart Albums** (📊 icon):
- Count calculated on-the-fly
- Always accurate
- Executes MongoDB query

**Manual Albums**:
- Count cached in database
- Fast display
- Use recalculate if drift suspected

### Directories Panel
```
┌─────────────────────────────┐
│ Directories                 │
├─────────────────────────────┤
│ 📁 D:/Photos (12,345 files) │ ← ROOT: file count
│   └─ 📂 2023 (156)          │ ← SUBDIR: child count
│   └─ 📂 2024 (89)           │
│ 📁 E:/Documents (5,678 files)│
└─────────────────────────────┘
```

**Roots** (📁):
- Show total file count recursively
- Example: "Photos (12,345 files)"

**Subdirectories** (📂):
- Show immediate child count (files + subdirs)
- Example: "2023 (156)"

---

## 5. Progress Dialogs

### Rebuild All Counts Dialog
```
┌───────────────────────────────────┐
│ Rebuilding file counts...         │
├───────────────────────────────────┤
│ [████████████████░░░░░] 75%       │
│                                   │
│ Recalculating file counts         │
│ across all systems...             │
│                                   │
│              [Cancel]             │
└───────────────────────────────────┘
```

### Results Message Box
```
┌───────────────────────────────────┐
│          Rebuild Complete         │
├───────────────────────────────────┤
│ Count rebuild complete!           │
│                                   │
│ Tags updated: 42                  │
│ Albums updated: 15                │
│ Directories updated: 128          │
│ Duration: 2.34s                   │
│                                   │
│              [  OK  ]             │
└───────────────────────────────────┘
```

---

## Quick Access Guide

### To Rebuild All Counts:
```
Menu Bar → Maintenance → Rebuild All Counts
```

### To Verify Data Integrity:
```
Menu Bar → Maintenance → Verify References
```

### To Cleanup Orphaned Records:
```
Menu Bar → Maintenance → Cleanup Orphaned Records
```

### To Recalculate Single Tag:
```
Tags Panel → Right-click tag → Recalculate Count
```

### To Recalculate Single Album:
```
Albums Panel → Right-click album → Recalculate Count
```

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────┐
│              UExplorer Main Window              │
├─────────────────────────────────────────────────┤
│ File Edit View Tools [Maintenance] Help        │ ← Menu Bar
├────────┬────────────────────────────────────────┤
│ TAGS   │                                        │
│ ─────  │         File Browser Area              │
│ Nature │                                        │
│ (42) ← │         (Cards/List/Grid View)         │
│   └──  │                                        │
│ Albums │                                        │
│ ─────  │                                        │
│ Fav ←  │                                        │
│ (28)   │                                        │
│        │                                        │
│ Dirs   │                                        │
│ ─────  │                                        │
│ 📁 Lib │                                        │
│ (1.2k) │                                        │
└────────┴────────────────────────────────────────┘
   ↑           
   └── Left sidebar with counts displayed
```

---

## File Locations (For Reference)

### Menu Implementation:
- **Menu Manager**: `samples/uexplorer/uexplorer_src/ui/managers/menu_manager.py`
- **Action Definitions**: `samples/uexplorer/uexplorer_src/ui/actions/action_definitions.py`
- **Main Window Handlers**: `samples/uexplorer/uexplorer_src/ui/main_window.py`

### Context Menus:
- **Tag Tree**: `samples/uexplorer/uexplorer_src/ui/widgets/tag_tree.py` (line ~280)
- **Album Tree**: `samples/uexplorer/uexplorer_src/ui/widgets/album_tree.py` (line ~193)

### Count Display:
- **Tag Display**: `samples/uexplorer/uexplorer_src/ui/widgets/tag_tree.py` (line ~123)
- **Album Display**: `samples/uexplorer/uexplorer_src/ui/widgets/album_tree.py` (line ~73)
- **Directory Display**: `samples/uexplorer/uexplorer_src/ui/docking/directory_panel.py` (line ~166)

---

## Testing the Features

### Step 1: Check Menu Exists
1. Launch UExplorer
2. Look at menu bar
3. Find "Maintenance" between "Tools" and "Help"
4. Click to see 3 menu items

### Step 2: Test Rebuild
1. Menu → Maintenance → Rebuild All Counts
2. Wait for progress dialog
3. See results message

### Step 3: Test Tag Context Menu
1. Go to Tags panel (left sidebar)
2. Right-click any tag
3. See "🔄 Recalculate Count" at bottom
4. Click to recalculate

### Step 4: Verify Counts Display
1. Tags panel: See "(N)" after tag names
2. Albums panel: See counts for both regular and 📊 smart albums
3. Directories panel: See file counts for roots, child counts for subdirs

---

## Troubleshooting

### Menu Not Appearing?
- **Check**: MaintenanceService registered in `main.py`
- **Check**: ActionDefinitions loaded correctly
- **Fix**: Restart UExplorer after code changes

### Counts Not Showing?
- **Check**: Database has file_count fields
- **Run**: Menu → Maintenance → Rebuild All Counts
- **Verify**: Right-click item → Recalculate Count

### Progress Dialog Not Appearing?
- **Check**: Qt event loop running
- **Check**: async/await properly configured
- **Fix**: Check console for Python errors

---

## Expected Behavior Summary

✅ **Maintenance Menu**: Visible in menu bar  
✅ **3 Menu Items**: Rebuild, Verify, Cleanup  
✅ **Context Menus**: Recalculate on tags & albums  
✅ **Real-Time Counts**: Displayed everywhere  
✅ **Progress Dialogs**: Show during operations  
✅ **Result Messages**: Confirm completion  
✅ **Auto-Refresh**: UI updates after rebuild  

All features are implemented and should be visible immediately in UExplorer!
