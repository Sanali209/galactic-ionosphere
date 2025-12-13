import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components"

Item {
    id: root

    // Explicit size hints for QQuickWidget
    width: 250
    height: 600

    // Properties removed to prevent shadowing global context properties
    // property var tagModel: null
    // property var folderModel: null

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
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                // Tag Header
                Rectangle {
                    Layout.fillWidth: true
                    height: 30
                    color: "#252526"

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 2

                        Label {
                            text: "Tags"
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
                        }
                    }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: tagModel // Context Property
                    delegate: ItemDelegate {
                        text: model.display
                        width: parent.width
                        onClicked: console.log("Clicked tag: " + (model.name || "")) // 'tagId' might not be in roleNames
                    }
                }
            }

            // index 1: Files
            FileExplorer {
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }
}
