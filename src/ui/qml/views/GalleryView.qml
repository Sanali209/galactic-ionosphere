import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components"

Item {
    id: root

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Search Box Area
        Rectangle {
            Layout.fillWidth: true
            height: 50
            color: "#252526"

            SearchBox {
                anchors.centerIn: parent
                width: parent.width - 20
                onSearchRequested: backendBridge.search(query)
            }
        }

        // Grid View
        GridView {
            id: grid
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            cellWidth: 160
            cellHeight: 160

            model: galleryModel

            delegate: Item {
                width: 150
                height: 150

                Rectangle {
                    anchors.fill: parent
                    color: "#333"
                    border.color: selected ? "#007acc" : "#555"
                    border.width: selected ? 2 : 1

                    property bool selected: false // TODO: Bind to selection model
                }

                Image {
                    anchors.fill: parent
                    anchors.margins: 2
                    source: "file:///" + imagePath
                    asynchronous: true
                    fillMode: Image.PreserveAspectCrop
                }

                Text {
                    text: imagePath
                    color: "white"
                    font.pixelSize: 10
                    anchors.bottom: parent.bottom
                    anchors.margins: 2
                    width: parent.width
                    elide: Text.ElideMiddle
                    horizontalAlignment: Text.AlignHCenter
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        backendBridge.selectImage(imageId);
                        grid.currentIndex = index;
                    }
                }
            }

            ScrollBar.vertical: ScrollBar {}
        }
    }
}
