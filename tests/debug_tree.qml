import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../src/ui/qml/controls"

ApplicationWindow {
    visible: true
    width: 600
    height: 800
    title: "Tree Debugger"

    ColumnLayout {
        anchors.fill: parent

        RowLayout {
            Button {
                text: "Add Root"
                onClicked: testModel.add_random_item()
            }
            Button {
                text: "Remove Random"
                onClicked: testModel.remove_random_item()
            }
            Button {
                text: "Reset Info"
                onClicked: testModel.reset_data()
            }
        }

        TreeListView {
            id: tree
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Explicitly set model from context property
            model: testModel

            // Inline Delegate for isolation
            itemDelegate: Rectangle {
                implicitHeight: 30
                height: 30
                width: view ? view.width : 200 // Safety fallback
                color: row % 2 == 0 ? "#f0f0f0" : "#ffffff"

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: (depth || 0) * 20

                    Text {
                        text: (isExpanded ? "[-]" : "[+]")
                        visible: hasChildren
                        MouseArea {
                            anchors.fill: parent
                            onClicked: testModel.toggle(row)
                        }
                    }

                    Text {
                        // Use exposed 'display' property from Loader
                        text: display || "N/A"
                    }

                    Text {
                        text: "(Row: " + row + ")"
                        color: "grey"
                        font.pixelSize: 10
                    }
                }
            }
        }
    }
}
