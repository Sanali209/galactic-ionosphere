import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root

    default property alias content: contentArea.data
    property string title: "Panel"

    // The Manager instance must be passed or available in scope
    property var manager: null

    // Layout properties (when docked)
    Layout.fillWidth: true
    Layout.fillHeight: true

    // Visuals
    Rectangle {
        id: bg
        anchors.fill: parent
        color: "#252526"
        border.color: "#3e3e42"
        border.width: 1

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // Title Bar (Draggable)
            Rectangle {
                id: titleBar
                Layout.fillWidth: true
                height: 28
                color: activeFocus || mouseArea.pressed ? "#007acc" : "#2d2d30"

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 5
                    anchors.rightMargin: 5

                    Label {
                        text: root.title
                        color: "white"
                        font.bold: true
                        Layout.fillWidth: true
                    }

                    ToolButton {
                        text: "X"
                        background: Rectangle {
                            color: "transparent"
                        }
                        contentItem: Text {
                            text: "✕"
                            color: "white"
                        }
                        onClicked: root.visible = false // Standard hide behavior
                    }
                }

                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    drag.target: draggableGhost // Drag a ghost item

                    property var ghost: null

                    onPressed: {
                        if (root.manager) {
                            root.manager.startDrag(root);
                        }
                        // Position ghost
                        draggableGhost.x = mouseArea.mouseX;
                        draggableGhost.y = mouseArea.mouseY;
                    }

                    onReleased: {
                        if (root.manager) {
                            root.manager.endDrag();
                        }
                        draggableGhost.Drag.drop();
                    }
                }
            }

            // Content
            Item {
                id: contentArea
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }

    // Ghost Item for Dragging
    // We use a simple rectangle that follows the mouse
    // In a real system, this should probably be outside the panel, in an overlay layer.
    // But Drag.active requires an Item.
    Item {
        id: draggableGhost
        width: root.width
        height: 28
        visible: mouseArea.drag.active

        Drag.active: mouseArea.drag.active
        Drag.hotSpot.x: 10
        Drag.hotSpot.y: 10
        Drag.keys: ["dockable"]

        Rectangle {
            anchors.fill: parent
            color: "#007acc"
            opacity: 0.5
            border.color: "white"

            Text {
                anchors.centerIn: parent
                text: root.title
                color: "white"
            }
        }
    }
}
