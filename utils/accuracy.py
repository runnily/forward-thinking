class Measure:
    def __init__(self, time_elapsed=[], epoch=[], loss=[], test_accuracy=[], train_accuracy=[]):
        self.time_elapsed = time_elapsed
        self.epoch = epoch
        self.loss = loss
        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
        self.__vals = [epoch, time_elapsed, loss, train_accuracy, test_accuracy]
        self.__dict_vals = {
            "Epoch": epoch,
            "Time Elapsed": self.time_elapsed,
            "Loss": self.loss,
            "Train accuracy": self.train_accuracy,
            "Test accuracy": self.test_accuracy,
        }
        self.init = False
        self.CONVERT_TO_SECONDS_METRIC = 1000

    def recordAccuracy(self, time_elapsed, epoch, loss, test_accuracy, train_accuracy):
        self.time_elapsed.append(time_elapsed / self.CONVERT_TO_SECONDS_METRIC)
        self.epoch.append(epoch)
        self.loss.append(loss)
        self.test_accuracy.append(test_accuracy)
        self.train_accuracy.append(train_accuracy)
        self.init = True
        return self.getDict()

    def __str__(self):
        title = ["Epoch", "Time Elapsed", "Loss", "Train accuracy", "Test accuracy"]
        string = ""
        # row_format = "{:15}" * (len(title) + 1)
        # string += row_format.format("", *title)
        for num, row in zip(title, self.__vals):
            string += "\n |{:>3}|{:>20}|".format(num, *row)
        return string

    def __clear(self):
        self.__vals = [[] for array in self.__vals]

    def __call__(self, time_elapsed, epoch, loss, test_accuracy, train_accuracy):
        if self.init:
            self.__clear()
        return self.recordAccuracy(time_elapsed, epoch, loss, test_accuracy, train_accuracy)

    def getDict(self):
        return self.__dict_vals

    def save(self, filename="accuracy.csv"):
        try:
            csv_text = "epoch,time_elapsed,loss,train_accuracy,test_accuracy\n"
            csv_rows = [
                "{},{},{},{},{}".format(x, y, z, i, k) for x, y, z, i, k in zip(*self.__vals)
            ]
            csv_text += "\n".join(csv_rows)
            with open("utils/recorded-accuracy/" + filename, "w") as f:
                f.write(csv_text)
            return True
        except Exception:
            return False
