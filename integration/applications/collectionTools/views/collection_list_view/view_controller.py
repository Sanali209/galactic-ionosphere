from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem


class ViewController:
    @staticmethod
    def delete_Detection(detection):
        detection.delete()
        MessageSystem.SendMessage('on_collection_record_deleted', detection)

