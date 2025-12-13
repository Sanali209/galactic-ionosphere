import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../controls"

Item {
    id: root

    // Bridge to Python Model via Context Property "journalModel"
    // property var journalModel: null  <-- Removed to prevent shadowing

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Toolbar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "#333"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 5

                TextField {
                    placeholderText: "Search..."
                    Layout.fillWidth: true
                    onTextEdited: journalModel.set_filter(text)
                }

                CheckBox {
                    text: "Info"
                    checked: true
                    onCheckedChanged: journalModel.toggle_level("INFO", checked)
                }
                CheckBox {
                    text: "Warn"
                    checked: true
                    onCheckedChanged: journalModel.toggle_level("WARNING", checked)
                }
                CheckBox {
                    text: "Error"
                    checked: true
                    onCheckedChanged: journalModel.toggle_level("ERROR", checked)
                }

                Button {
                    text: "Clear"
                    onClicked: journalModel.clear()
                }
            }
        }

        // Logs List
        ListView {
            id: logView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: journalModel

            delegate: Rectangle {
                width: ListView.view.width
                height: 25
                color: (model.level === "ERROR") ? "#442222" : (model.level === "WARNING") ? "#443300" : "transparent"

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 5
                    spacing: 10

                    Text {
                        text: model.timestamp
                        color: "#888"
                        font.family: "Consolas"
                    }
                    Text {
                        text: "[" + model.level + "]"
                        color: (model.level === "ERROR") ? "#ff5555" : (model.level === "WARNING") ? "#ffaa00" : "#aaaaaa"
                        font.bold: true
                        Layout.preferredWidth: 60
                    }
                    Text {
                        text: model.message
                        color: "#ddd"
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    onDoubleClicked: console.log("Show details for: " + model.details)
                }
            }

            // Auto Scroll
            onCountChanged: {
                if (contentY >= contentHeight - height - 40) { // If near bottom
                    positionViewAtEnd();
                }
            }
        }
    }
}
