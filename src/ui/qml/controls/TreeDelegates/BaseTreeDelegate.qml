import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// Abstract base for tree items.
// Usage: Inherit/Use this and put your content in 'content' property (alias to innerLayout.data)
Item {
    id: baseDelegate

    property int indentation: 20
    implicitHeight: 30

    // Properties injected by Loader (bound to parent)
    property var modelData: parent ? parent.modelData : null
    property int depth: parent ? parent.depth : 0
    property bool isExpanded: parent ? parent.isExpanded : false
    property bool hasChildren: parent ? parent.hasChildren : false
    property int row: parent ? parent.row : 0
    property var modelObj: parent ? parent.modelObj : null

    // Data Roles exposed by Loader
    property string name: parent ? parent.name : ""
    property string display: parent ? parent.display : ""

    default property alias content: innerLayout.data

    RowLayout {
        anchors.fill: parent
        spacing: 4

        // Indentation
        Item {
            Layout.preferredWidth: (baseDelegate.depth || 0) * baseDelegate.indentation
            Layout.fillHeight: true
        }

        // Expander
        Button {
            Layout.preferredWidth: 20
            Layout.preferredHeight: 20
            flat: true
            text: baseDelegate.isExpanded ? "v" : ">"
            visible: baseDelegate.hasChildren
            onClicked: {
                if (baseDelegate.modelObj) {
                    baseDelegate.modelObj.toggle(baseDelegate.row);
                }
            }
        }

        // Your content goes here
        RowLayout {
            id: innerLayout
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }

    // Expanded signal
    signal rightClicked

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.RightButton
        onClicked: baseDelegate.rightClicked()
        z: -1 // Ensure it is behind interactive content if any
    }
}
