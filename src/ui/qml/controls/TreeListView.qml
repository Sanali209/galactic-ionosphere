import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ListView {
    id: root

    // The component to use for each row
    property Component itemDelegate: null

    // Indentation per depth level
    property int indentation: 20

    clip: true
    boundsBehavior: Flickable.StopAtBounds

    // Default delegate if none provided
    Component {
        id: defaultDelegate
        Item {
            width: root.width
            height: 30
            Text {
                text: "No Delegate Provided"
                color: "red"
                anchors.centerIn: parent
            }
        }
    }

    delegate: Loader {
        width: root.width
        // Allow the delegate to calculate its own height, or default to loaded item height
        // Allow the delegate to calculate its own height, or default to loaded item height
        height: item ? (item.implicitHeight || item.height || 30) : 30

        sourceComponent: root.itemDelegate || defaultDelegate

        // Pass model data to the loaded item
        property var modelData: model

        // Pass view properties
        property int depth: model.depth !== undefined ? model.depth : 0
        property bool isExpanded: model.isExpanded !== undefined ? model.isExpanded : false
        property bool hasChildren: model.hasChildren !== undefined ? model.hasChildren : false

        // Expose common data roles
        property string display: model.display !== undefined ? model.display : ""
        property string name: model.name !== undefined ? model.name : ""

        property int row: index
        property var view: ListView.view
        property var modelObj: ListView.view.model

        onLoaded: {
            // Optional: Connect signals if needed, though direct binding in delegate is better
        }
    }
}
