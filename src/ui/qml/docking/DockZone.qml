import QtQuick 2.15

DropArea {
    id: root

    property var manager: null
    property string zoneName: "zone"
    property Item targetParent: null // Where to dock children

    keys: ["dockable"]

    anchors.fill: parent

    Rectangle {
        anchors.fill: parent
        color: "#007acc"
        opacity: root.containsDrag ? 0.3 : 0.0
        visible: root.containsDrag

        Text {
            anchors.centerIn: parent
            text: "Dock " + root.zoneName
            color: "white"
            font.bold: true
            font.pixelSize: 20
        }
    }

    onDropped: drop => {
        console.log("Dropped in " + zoneName);
        if (manager && manager.draggedPanel && targetParent) {
            manager.dock(manager.draggedPanel, targetParent);
            drop.accept();
        }
    }
}
