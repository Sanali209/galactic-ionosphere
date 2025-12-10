import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Qt.labs.folderlistmodel 2.15

Item {
    id: root

    // Start at a sensible default (User Home or C:/)
    property string currentPath: "file:///C:/"

    function navigate(path) {
        if (path.toString().startsWith("file:///")) {
            currentPath = path;
        } else {
            currentPath = "file:///" + path;
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header
        Rectangle {
            Layout.fillWidth: true
            height: 30
            color: "#333"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 2

                ToolButton {
                    text: "\u2B06" // Up Arrow
                    onClicked: {
                        console.log("Up pressed. Current parent:", folderModel.parentFolder);
                        root.currentPath = folderModel.parentFolder;
                    }
                }

                TextField {
                    text: root.currentPath
                    Layout.fillWidth: true
                    readOnly: true
                    background: Rectangle {
                        color: "transparent"
                    }
                    color: "white"
                }
            }
        }

        // List
        ListView {
            id: fileList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            model: FolderListModel {
                id: folderModel
                folder: root.currentPath
                showDirs: true
                showFiles: true
                showDotAndDotDot: false
                nameFilters: ["*"]
            }

            delegate: ItemDelegate {
                width: parent ? parent.width : 200
                text: fileName
                icon.name: fileIsDir ? "folder" : "image-x-generic"

                onClicked: {
                    if (fileIsDir) {
                        // Single click: Filter Gallery to this folder (Non-Recursive)
                        console.log("Filtering by folder:", fileUrl);
                        backendBridge.filterByFolder(fileUrl, false);

                        // Also navigate? Usually single click selects, double click navigates.
                        // User said "on pres folder faind images onli in this folder" -> Single Click.
                        // But also need to navigate? Let's assume single click just filters content,
                        // DOUBLE click enters folder.
                    }
                }

                onDoubleClicked: {
                    if (fileIsDir) {
                        root.currentPath = fileUrl;
                        backendBridge.filterByFolder(fileUrl, false);
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    acceptedButtons: Qt.RightButton
                    onClicked: {
                        if (fileIsDir) {
                            contextMenu.open();
                            // Set context for menu actions if needed (e.g. current path)
                            fileList.currentIndex = index;
                        }
                    }
                }

                Menu {
                    id: contextMenu
                    title: "Folder Actions"
                    MenuItem {
                        text: "Show Recursive"
                        onTriggered: {
                            console.log("Recursive show for: " + fileUrl);
                            backendBridge.filterByFolder(fileUrl, true);
                        }
                    }
                    MenuItem {
                        text: "Import This Folder"
                        onTriggered: backendBridge.importFolder(fileUrl)
                    }
                }

                // Highlight styles
                highlighted: ListView.isCurrentItem
            }

            ScrollBar.vertical: ScrollBar {}
        }
    }
}
