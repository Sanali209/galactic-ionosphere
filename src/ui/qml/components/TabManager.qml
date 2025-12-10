import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    
    // Model for tabs: List of { title, type, data }
    property var tabs: [
        { title: "Gallery.view", type: "gallery" },
        { title: "Search Results [Cat]", type: "search", query: "cat" }
    ]
    
    property int currentIndex: 0
    
    // Top Tab Bar
    Rectangle {
        id: tabBar
        height: 32
        width: parent.width
        color: "#2d2d30"
        z: 10
        
        ListView {
            anchors.fill: parent
            orientation: ListView.Horizontal
            model: tabs
            
            delegate: Rectangle {
                width: 150
                height: 32
                color: index === root.currentIndex ? "#007acc" : "#3e3e42"
                border.color: "#252526"
                
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 5
                    Text { 
                        text: modelData.title
                        color: index === root.currentIndex ? "white" : "#ccc"
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                    Text { 
                        text: "x"
                        color: "#ccc"
                        MouseArea {
                            anchors.fill: parent
                            onClicked: { 
                                // Close logic (stub)
                                console.log("Close tab " + index)
                            }
                        }
                    }
                }
                
                MouseArea { 
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton
                    z: -1
                    onClicked: root.currentIndex = index
                }
            }
        }
    }
    
    // Content Area
    Rectangle {
        anchors.top: tabBar.bottom
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        color: "#1e1e1e"
        
        // Loader based on current tab type
        Loader {
            anchors.fill: parent
            sourceComponent: switch(tabs[root.currentIndex].type) {
                case "gallery": return galleryComp
                case "search": return searchComp
                default: return null
            }
        }
    }
    
    Component {
        id: galleryComp
        GridView {
            model: galleryModel
            cellWidth: 160; cellHeight: 160
            clip: true
            delegate: Item { 
                width: 150; height: 150
                Rectangle { anchors.fill: parent; color: "#333"; border.color: "#555" } 
                Image { anchors.fill:parent; anchors.margins: 1; source: "file:///" + imagePath; asynchronous: true; fillMode: Image.PreserveAspectCrop }
                Text { text: imagePath; color: "white"; font.pixelSize: 10; anchors.bottom: parent.bottom }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: backendBridge.selectImage(imageId)
                }
            }
        }
    }
    
    Component {
        id: searchComp
        Rectangle {
            color: "#222"
            Text { text: "Search Results Placeholder"; color: "white"; anchors.centerIn: parent }
        }
    }
}
