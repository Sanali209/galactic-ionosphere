import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components"

Item {
    // title handled by parent DockablePanel

    property string activeId: ""
    property string activePath: "-"
    property string activeDims: "-"
    property string activeSize: "-"
    property string activeMeta: "-"

    Connections {
        target: backendBridge
        function onImageSelected(id, path, dims, size, meta) {
            activeId = id;
            activePath = path;
            activeDims = dims;
            activeSize = size;
            activeMeta = meta;
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 5

        Label {
            text: "Image Details"
            font.bold: true
            font.pixelSize: 14
        }

        Rectangle {
            height: 1
            Layout.fillWidth: true
            color: "#ccc"
        }

        Label {
            text: "ID:"
            color: "#666"
        }
        TextField {
            text: activeId
            readOnly: true
            Layout.fillWidth: true
            background: Rectangle {
                color: "transparent"
            }
        }

        Label {
            text: "Path:"
            color: "#666"
        }
        Label {
            text: activePath
            Layout.fillWidth: true
            elide: Text.ElideMiddle
        }

        Label {
            text: "Dimensions:"
            color: "#666"
        }
        Label {
            text: activeDims
        }

        Label {
            text: "Size:"
            color: "#666"
        }
        Label {
            text: activeSize
        }

        Label {
            text: "Metadata:"
            color: "#666"
        }
        TextArea {
            text: activeMeta
            readOnly: true
            Layout.fillWidth: true
            Layout.fillHeight: true
            wrapMode: Text.Wrap
            background: Rectangle {
                color: "#eee"
            }
        }

        Item {
            Layout.fillHeight: true
        } // Spacer
    }
}
