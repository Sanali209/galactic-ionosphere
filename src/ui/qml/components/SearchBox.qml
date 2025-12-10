import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    height: 40
    Layout.fillWidth: true

    // Signals
    signal searchRequested(string query)

    RowLayout {
        anchors.fill: parent
        spacing: 5

        TextField {
            id: searchField
            Layout.fillWidth: true
            placeholderText: "Search images..."
            onAccepted: root.searchRequested(text)
            selectByMouse: true
            font.pixelSize: 14
            background: Rectangle {
                color: "#252526"
                border.color: "#3e3e42"
                radius: 2
            }
            color: "#cccccc"
        }

        Button {
            text: "Search"
            onClicked: root.searchRequested(searchField.text)
            background: Rectangle {
                color: "#0e639c"
                radius: 2
            }
            contentItem: Text {
                text: parent.text
                color: "white"
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }
    }
}
