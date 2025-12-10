import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    property string title: "Panel"
    default property alias content: contentArea.data
    
    color: "#e8e8e8" // VS Panel background
    border.color: "#ccc"
    border.width: 1
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        // Title Bar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 28
            color: "#007acc" // Visual Studio Blue
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 4
                
                Text {
                    text: root.title
                    color: "white"
                    font.pixelSize: 12
                    font.bold: true
                    Layout.fillWidth: true
                }
                
                // Docking Controls (Stub)
                Text { text: "📌"; color: "#ddd"; font.pixelSize: 12 }
                Text { text: "❌"; color: "#ddd"; font.pixelSize: 12 }
            }
        }
        
        // Content
        Item {
            id: contentArea
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
        }
    }
}
