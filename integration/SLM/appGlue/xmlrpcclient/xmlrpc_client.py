import xmlrpc.client


class XMLRPCClient:
    uri = "http://localhost:8000"

    @staticmethod
    def getproxy():
        return xmlrpc.client.ServerProxy(XMLRPCClient.uri + "/RPC2")


