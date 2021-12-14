
class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None


class LinkedList:
    def __init__(self):
        self.headval = None

    def FindVal(self, name, year, gender):
        printval = self.headval
        while printval is not None:
            if printval.dataval.name == name and printval.dataval.year == year and printval.dataval.gender == gender:
                return True
            printval = printval.nextval
        return False

    def CountNodes(self):
        val = self.headval
        counter = 0
        while val is not None:
            counter += 1
            val = val.nextval
        return counter

    def AtBegining(self, newdata):
        NewNode = Node(newdata)
        NewNode.nextval = self.headval
        self.headval = NewNode

    def AtEnd(self, newdata):
        NewNode = Node(newdata)
        if self.headval is None:
            self.headval = NewNode
            return
        laste = self.headval
        while laste.nextval:
            laste = laste.nextval
        laste.nextval = NewNode

    def __len__(self):
        if self.headval:
            return True
        else:
            return False

