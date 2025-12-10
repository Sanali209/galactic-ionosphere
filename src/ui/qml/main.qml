import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "components"
import "panels"
import "docking"

ApplicationWindow {
    id: appWindow
    visible: true
    width: 1400
    height: 900
    title: "Galactic Ionosphere (Pro)"

    // Core Managers
    DockManager {
        id: dockManager
    }

    FolderDialog {
        id: importDialog
        title: "Select Folder"
        onAccepted: backendBridge.importFolder(selectedFolder)
    }

    menuBar: MenuBar {
        Menu {
            title: "&File"
            MenuItem {
                text: "Import Folder..."
                onTriggered: importDialog.open()
            }
            MenuSeparator {}
            MenuItem {
                text: "Exit"
                onTriggered: Qt.quit()
            }
        }
        Menu {
            title: "&View"
            MenuItem {
                text: "Solution Explorer"
                checkable: true
                checked: solutionExplorerPanel.visible
                onTriggered: {
                    solutionExplorerPanel.visible = checked;
                    mainDockLayout.updateLayout();
                }
            }
            MenuItem {
                text: "Properties"
                checkable: true
                checked: propertiesPanel.visible
                onTriggered: {
                    propertiesPanel.visible = checked;
                    mainDockLayout.updateLayout();
                }
            }
            MenuItem {
                text: "Output"
                checkable: true
                checked: outputPanel.visible
                onTriggered: {
                    outputPanel.visible = checked;
                    mainDockLayout.updateLayout();
                }
            }
        }
    }

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            ToolButton {
                text: "Import Folder"
                onClicked: importDialog.open()
            }
            TextField {
                id: searchInput
                placeholderText: "Search..."
                Layout.fillWidth: true
                onAccepted: backendBridge.search(text)
            }
        }
    }

    DockLayout {
        id: mainDockLayout
        anchors.fill: parent

        // Docks auto-manage visibility based on content
        // leftDockVisible, rightDockVisible, etc. are now read-only derived properties in DockLayout
        // We just toggle the visibility of the panels themselves.

        // Note: When we "close" a docked panel, we set its visible to false.
        // DockLayout needs to know to hide the dock if all children are hidden.

        // Wait, DockLayout's new logic uses `children.length > 0`.
        // If a child is hidden but still parented, length is still > 0.
        // We need `visible` check too.

        // But for now, let's remove these explicit aliases which were forcing the dock visible/hidden
        // independent of content location.

        // LEFT: Sidebar
        leftPanel: DockablePanel {
            id: solutionExplorerPanel
            title: "Solution Explorer"
            anchors.fill: parent
            manager: dockManager

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
                        model: tagModel
                        delegate: ItemDelegate {
                            text: display
                            width: parent.width
                            onClicked: console.log("Clicked tag: " + tagId)
                        }
                    }

                    // index 1: Files
                    FileExplorer {
                        // anchors.fill not needed in StackLayout usually, but StackLayout manages visibility
                    }
                }
            }
        }

        // CENTER: Gallery (Document Tab)
        centerArea: TabManager {
            anchors.fill: parent
        }

        // RIGHT: Properties
        rightPanel: DockablePanel {
            id: propertiesPanel
            title: "Properties"
            anchors.fill: parent
            manager: dockManager
            // Because PropertiesPanel is a type, we wrap it or put it inside content
            // Assuming PropertiesPanel is an Item based panel

            PropertiesPanel {
                anchors.fill: parent
            }
        }

        // BOTTOM: Output
        bottomPanel: DockablePanel {
            id: outputPanel
            title: "Output"
            anchors.fill: parent
            manager: dockManager

            OutputPanel {
                anchors.fill: parent
            }
        }
    }
}
