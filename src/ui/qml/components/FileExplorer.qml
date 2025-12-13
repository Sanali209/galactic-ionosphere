import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../controls"
import "../controls/TreeDelegates"

Item {
    id: root

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header
        Rectangle {
            Layout.fillWidth: true
            height: 30
            color: "#252526"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 2

                Label {
                    text: "Folders"
                    color: "white"
                    font.bold: true
                    Layout.leftMargin: 5
                }

                Item {
                    Layout.fillWidth: true
                }

                ToolButton {
                    text: "\u21BB" // Refresh
                    onClicked: backendBridge.refreshGallery()
                    // Ideally we refresh folders via fs_model.load_roots() if exposed
                }
            }
        }

        // Tree View
        TreeListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            model: fileSystemModel

            itemDelegate: FileTreeDelegate {
                // We can add actions here if needed or modify FileTreeDelegate
                // FileTreeDelegate handles renaming/deleting via context menu

                // Double click to import? Or Context menu 'Import'?
                // FileTreeDelegate has a context menu.
            }
        }
    }
}
