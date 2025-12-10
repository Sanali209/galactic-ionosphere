import QtQuick 2.15

QtObject {
    id: dockManager

    // Current drag state
    property var draggedPanel: null
    property bool isDragging: draggedPanel !== null

    // Registry of dock zones (Layouts register themselves here)
    // We might not need an explicit list if we use DropArea signals.

    // Action: Start Drag
    function startDrag(panel) {
        draggedPanel = panel;
    }

    // Action: End Drag
    function endDrag() {
        draggedPanel = null;
    }

    // Action: Dock Panel
    function dock(panel, targetParent, properties) {
        if (!panel || !targetParent)
            return;

        console.log("Docking " + panel + " to " + targetParent);

        // Reparent
        panel.parent = targetParent;

        // Apply layout properties if needed (e.g. Layout.fillWidth)
        // Since we can't easily set attached properties from JS on an object that already exists
        // without some hacks or careful setup, we assume the target layout handles it
        // or the panel has bindings.

        // For SplitView/ColumnLayout, items usually just need to be children.
    }

    // Action: Float Panel (Create Window)
    function floatPanel(panel) {
        // This requires creating a purely dynamic Window and reparenting the content.
        // QML 'Window' cannot easily accept an existing Item as a child of its 'contentItem'
        // if that Item was created elsewhere, UNLESS we just reparent the visual item.
        // It's possible.
        console.log("Float requested (Not yet implemented)");
    }
}
