
class Measure():

  def __init__(self,time_elapsed=[], epoch=[], loss=[], test_accuracy=[], train_accuracy=[]):
      self.time_elapsed = time_elapsed
      self.epoch = epoch
      self.loss = loss
      self.test_accuracy = test_accuracy
      self.train_accuracy = train_accuracy
      self.__vals = [epoch, time_elapsed, loss, train_accuracy, test_accuracy]

  def recordAccuracy(self, time_elapsed, epoch, loss, test_accuracy, train_accuracy):
      self.time_elapsed.append(time_elapsed)
      self.epoch.append(epoch)
      self.loss.append(loss)
      self.test_accuracy.append(test_accuracy)
      self.train_accuracy.append(train_accuracy)

  def __str__(self):
    title = ["Epoch", "Time Elapsed", "Loss", "Train accuracy", "Test accuracy"]
    string = ""
    #row_format = "{:15}" * (len(title) + 1) 
    #string += row_format.format("", *title) 
    for num, row in zip(title, self.__vals):
        string += "\n |{:>3}|{:>20}|".format(num, *row)
    return string

  def __call__(self, time_elapsed, epoch, loss, test_accuracy, train_accuracy):
      self.recordAccuracy(time_elapsed, epoch, loss, test_accuracy, train_accuracy)

  