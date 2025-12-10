import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    width: 600
    height: 400

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Left vertical tabs
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 150
            color: "#252526" // Visual Studio Code Sidebar Color

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                Repeater {
                    model: ["General", "AI"]
                    delegate: Button {
                        text: modelData
                        checkable: true
                        checked: index === stack.currentIndex
                        onClicked: stack.currentIndex = index
                        Layout.fillWidth: true
                        Layout.preferredHeight: 40

                        background: Rectangle {
                            color: parent.checked ? "#37373d" : "transparent"
                        }
                        contentItem: Text {
                            text: parent.text
                            color: parent.checked ? "white" : "#cccccc"
                            font.pixelSize: 14
                            leftPadding: 15
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
                // Spacer to push items up
                Item {
                    Layout.fillHeight: true
                }
            }
        }

        // Right side pages
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#1e1e1e" // Editor background

            StackLayout {
                id: stack
                anchors.fill: parent
                currentIndex: 0

                // General Settings Page
                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 20
                        Text {
                            text: "General Settings"
                            color: "white"
                            font.bold: true
                            font.pixelSize: 18
                        }
                        Item {
                            Layout.fillHeight: true
                        } // Spacer
                    }
                }

                // AI Settings Page
                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 20
                        spacing: 15

                        Text {
                            text: "AI Configuration"
                            color: "white"
                            font.bold: true
                            font.pixelSize: 18
                        }

                        RowLayout {
                            spacing: 10
                            Text {
                                text: "Search Result Limit:"
                                color: "#cccccc"
                            }
                            TextField {
                                id: resultCountField
                                text: backendBridge.aiResultLimit ? backendBridge.aiResultLimit.toString() : "20"
                                color: "white"
                                background: Rectangle {
                                    color: "#3c3c3c"
                                    border.color: "#3c3c3c"
                                }
                                inputMethodHints: Qt.ImhDigitsOnly
                                validator: IntValidator {
                                    bottom: 1
                                    top: 100
                                }
                                onEditingFinished: {
                                    var val = parseInt(text);
                                    if (!isNaN(val))
                                        backendBridge.setAiResultLimit(val);
                                }
                            }
                        }

                        Text {
                            text: "Note: Changes are saved automatically."
                            color: "#808080"
                            font.italic: true
                        }

                        Item {
                            Layout.fillHeight: true
                        } // Spacer
                    }
                }
            }
        }
    }
}
