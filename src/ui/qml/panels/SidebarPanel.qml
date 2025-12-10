import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components"

Item {
    id: root

    // Explicit size hints for QQuickWidget
    width: 250
    height: 600

    property var tagModel: null
    property var folderModel: null
    // Data models injected via context properties usually, but explicit properties allow cleaner interface if needed.
    // For now we rely on Global Context 'folderModel' and 'tagModel' set in Python.

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        TabBar {
            id: sidebarTabs
            Layout.fillWidth: true
            TabButton {
                text: "Tags"
            }
            TabButton {
                text: "Files"
            }
        }

        StackLayout {
            currentIndex: sidebarTabs.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            // index 0: Tags
            ListView {
                clip: true
                model: tagModel // Context Property
                delegate: ItemDelegate {
                    text: display
                    width: parent.width
                    onClicked: console.log("Clicked tag: " + tagId)
                }
            }

            // index 1: Files
            FileExplorer {
                // components/FileExplorer.qml
            }
        }
    }
}
