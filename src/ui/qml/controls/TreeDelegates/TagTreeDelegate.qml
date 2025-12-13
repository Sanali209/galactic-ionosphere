import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

BaseTreeDelegate {
    id: tagDelegate

    // Tag specific icon or styling
    Rectangle {
        Layout.preferredWidth: 16
        Layout.preferredHeight: 16
        color: "transparent"

        // Simple circle for tag
        Rectangle {
            anchors.centerIn: parent
            width: 10
            height: 10
            radius: 5
            color: "#6a9fb5" // Muted blue
        }
    }

    Text {
        Layout.fillWidth: true
        text: name || display
        color: "#e0e0e0"
        elide: Text.ElideRight
        verticalAlignment: Text.AlignVCenter
    }
}
