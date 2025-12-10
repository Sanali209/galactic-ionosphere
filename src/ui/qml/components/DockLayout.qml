import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../docking"

Item {
    id: root

    // Manager Injection
    property var manager: null

    // Areas (We use aliases so parent can inject content)
    property alias leftPanel: leftArea.data
    property alias rightPanel: rightArea.data
    property alias bottomPanel: bottomArea.data
    property alias centerArea: centerContent.data

    // State for forcing layout updates
    property int layoutRevision: 0
    function updateLayout() {
        layoutRevision++;
    }

    // Helper to check if any child is visible
    function hasVisibleChildren(item) {
        var _ = layoutRevision; // Dependency injection
        if (!item || !item.children)
            return false;

        for (var i = 0; i < item.children.length; i++) {
            if (item.children[i].visible)
                return true;
        }
        return false;
    }

    property bool leftDockVisible: hasVisibleChildren(leftArea)
    property bool rightDockVisible: hasVisibleChildren(rightArea)
    property bool bottomDockVisible: hasVisibleChildren(bottomArea)

    // Auto-update when children are added/removed (Drag & Drop)
    Connections {
        target: leftArea
        function onChildrenChanged() {
            root.updateLayout();
        }
    }
    Connections {
        target: rightArea
        function onChildrenChanged() {
            root.updateLayout();
        }
    }
    Connections {
        target: bottomArea
        function onChildrenChanged() {
            root.updateLayout();
        }
    }

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal
        clip: false

        // LEFT DOCK
        Item {
            id: leftDock
            SplitView.preferredWidth: 250
            SplitView.minimumWidth: 50
            visible: root.leftDockVisible

            ColumnLayout {
                anchors.fill: parent
                spacing: 2
                Item {
                    id: leftArea
                    Layout.fillHeight: true
                    Layout.fillWidth: true
                }
            }
        }

        // CENTER + RIGHT + BOTTOM
        // Visual Studio Layout: (Left | (Center | Right)) then Bottom?
        // Simplest VS-like: (Left | Center | Right)

        SplitView {
            SplitView.fillWidth: true
            orientation: Qt.Vertical

            SplitView {
                SplitView.fillHeight: true
                orientation: Qt.Horizontal

                // CENTER DOCUMENT AREA
                Rectangle {
                    color: "#252526" // Dark Editor BG
                    SplitView.fillWidth: true
                    Item {
                        id: centerContent
                        anchors.fill: parent
                    }
                }

                // RIGHT DOCK
                Item {
                    id: rightDock
                    SplitView.preferredWidth: 250
                    // visible controlled by alias

                    Item {
                        id: rightArea
                        anchors.fill: parent
                    }
                }
            }

            // BOTTOM DOCK
            Item {
                id: bottomDock
                SplitView.preferredHeight: 150
                visible: root.bottomDockVisible
                // visible controlled by alias

                Item {
                    id: bottomArea
                    anchors.fill: parent
                }
            }
        }
    }

    // DOCK ZONES OVERLAY
    Item {
        anchors.fill: parent
        // Visible only when dragging
        visible: manager ? manager.isDragging : false
        z: 100 // Topmost

        // Left Zone
        DockZone {
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 50
            zoneName: "Left"
            manager: root.manager
            targetParent: leftArea
        }

        // Right Zone
        DockZone {
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 50
            zoneName: "Right"
            manager: root.manager
            targetParent: rightArea
        }

        // Bottom Zone
        DockZone {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 50
            zoneName: "Bottom"
            manager: root.manager
            targetParent: bottomArea
        }
    }
}
