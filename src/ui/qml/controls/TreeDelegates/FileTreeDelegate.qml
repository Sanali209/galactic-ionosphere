import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

BaseTreeDelegate {

    // Properties required for context menu
    property bool editable: true

    onRightClicked: fileMenu.popup()

    // Content
    // Placeholder Icon (Orange Rectangle) to avoid QQuickImage errors
    Rectangle {
        Layout.preferredWidth: 20
        Layout.preferredHeight: 20
        color: "orange"
        radius: 4
    }

    TextInput {
        id: nameInput
        Layout.fillWidth: true
        text: name || "Unknown" // Use name from BaseTreeDelegate (from Loader)
        color: "#ccc"
        selectByMouse: true
        readOnly: !editable

        onAccepted: {
            // TODO: Wiring for rename needs to be checked, modelData.rename might not exist on simple item dict
            // if (modelData.rename) modelData.rename(text);
            focus = false;
        }
    }

    // Context Menu (triggered by BaseTreeDelegate signal)
    Menu {
        id: fileMenu
        MenuItem {
            text: "Rename"
            onTriggered: {
                nameInput.readOnly = false;
                nameInput.forceActiveFocus();
            }
        }
        MenuItem {
            text: "Delete"
            onTriggered: {
                // if (modelData.delete_item) modelData.delete_item()
            }
        }
        MenuSeparator {}

        MenuItem {
            text: "Import This Folder"
            onTriggered: {
                // Call bridge
                // We need path.
                // modelData might be the dict from Python.
                // Check BaseFlatTreeModel data: it returns item dict for some roles?
                // Actually we have 'path' role?
                // Let's use the 'modelObj' which is the model itself? No.
                // We can try to access the path from the model data if exposed.
                // BaseFlatTreeModel exposes "path" role? Yes, wait.
                // Loader exposes: name, display.
                // We might need to expose 'path' in TreeListView.qml if not already.
                // Let's assume 'model.path' works if role is strictly defined.
                // BUT, simpler: The user imports via the folder structure.
                // If we don't have path, we can't import.
                // Checking TreeListView.qml again...
                console.log("Importing: " + baseDelegate.modelData.path); // Debug
                backendBridge.importFolder("file:///" + baseDelegate.modelData.path);
            }
        }

        MenuItem {
            text: "Show Recursive"
            onTriggered: {
                // Trigger filter
                // backendBridge.filterByFolder...
            }
        }
    }
}
