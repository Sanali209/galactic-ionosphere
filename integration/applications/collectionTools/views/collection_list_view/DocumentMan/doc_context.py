from SLM.appGlue.DesignPaterns.SingletonAtr import singleton


@singleton
class DocumentContext:
    #todo temporary class
    def __init__(self):
        self.document = None