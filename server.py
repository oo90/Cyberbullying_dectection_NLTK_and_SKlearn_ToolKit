
from autobahn.twisted.websocket import WebSocketServerProtocol, WebSocketServerFactory
import sys
from twisted.python import log
from twisted.internet import reactor

import psutil
import joblib
# from sklearn.externalsjoblib import joblib


# import trained classifiers from external files
save_Classifiers = "pickles/Classifier.pickle"
classifier = joblib.load(save_Classifiers)

save_featureGetters = "pickles/featureGetters.pickle"
featureGetter = joblib.load(save_featureGetters)


class MyServerProtocol(WebSocketServerProtocol):

    def onConnect(self, request):
        print("Client connecting: {}".format(request.peer))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        if isBinary:
            print("Binary message received: {} bytes".format(len(payload)))
        else:
            # take input from client
            print("Text message received: {}".format(payload.decode('utf8')))
            txt = payload.decode('utf8')
            example = []
            example.append(txt)
            # getting features with feature_Getter
            example_features = featureGetter.transform(example)
            # classify sentence using imported classifier
            result = classifier.predict(example_features)

            if result == 1:
                res = "Cyberbullying detected!"
            else:
                res = "Non Cyberbullying."

        # echo back message verbatim
        # self.sendMessage(payload, isBinary)
        self.sendMessage(res.encode('utf8'), isBinary)

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {}".format(reason))

        PROC_NAME = "QtWebEngineProcess.exe"

        for proc in psutil.process_iter():
            # check whether the process to kill name matches
            if proc.name() == PROC_NAME:
                proc.kill()
        print("WebEngineProcess also killed.")


if __name__ == '__main__':

    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory(u"ws://127.0.0.1:50007")
    factory.protocol = MyServerProtocol
    # factory.setProtocolOptions(maxConnections=2)

    # note to self: if using putChild, the child must be bytes...

    reactor.listenTCP(50007, factory)

    reactor.run()
