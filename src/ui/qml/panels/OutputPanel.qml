import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components"

Item {
    // title handled by parent DockablePanel

    ScrollView {
        id: scrollView
        anchors.fill: parent
        
        TextArea {
            id: logArea
            readOnly: true
            font.family: "Consolas"
            font.pixelSize: 11
            background: Rectangle {
                color: "#1e1e1e"
            }
            color: "#d4d4d4"
            text: "System Initialized.\n"
            
            // Auto-scroll
            onTextChanged: {
                logArea.cursorPosition = logArea.length
            }

            Connections {
                target: backendBridge
                function onLogMessage(msg) {
                    logArea.append(msg);
                }
            }
        }
    }
}
