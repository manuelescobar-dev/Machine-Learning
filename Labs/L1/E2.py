import math
import sys

#from snakeML.loads import read_txt

class Pt():
    buses={}
    lines={}
        
    def addBus(self,bus,location,time):
        if bus.lineID in self.lines:
            if bus.busID in self.lines[bus.lineID]:
                current_loc=self.lines[bus.lineID][bus.busID].initial_location
                current_time=self.lines[bus.lineID][bus.busID].initial_time
                self.lines[bus.lineID][bus.busID].update_Ttime(time-current_time)
                self.lines[bus.lineID][bus.busID].update_Tdis(euclidean(current_loc,location))
                self.lines[bus.lineID][bus.busID].update_time(time)
                self.lines[bus.lineID][bus.busID].update_loc(location)
            else:
                bus.update_time(time)
                bus.update_loc(location)
                self.lines[bus.lineID][bus.busID]=bus
        else:
            bus.update_time(time)
            bus.update_loc(location)
            self.lines[bus.lineID]={bus.busID:bus}

    def getDistance(self,busID):
        for line in self.lines:
            if busID in self.lines[line]:
                print(busID,' - Total Distance: ',self.lines[line][busID].total_distance)
    
    def getLineAvgSpeed(self,lineID):
        distance=0
        time=0
        for i in self.lines[lineID]:
            distance+=self.lines[lineID][i].total_distance
            time+=self.lines[lineID][i].total_time
        print(lineID,' - Avg Speed: ', distance/time)

def euclidean(ini,fin):
    return math.sqrt((ini[0]-fin[0])**2+(ini[1]-fin[1])**2)

class Bus():
    initial_time=0
    intial_location=()
    total_time=0
    total_distance=0
    def __init__(self, busID, lineID):
        self.busID=busID
        self.lineID=lineID

    def update_time(self,time):
        self.initial_time=time

    def update_loc(self,location):
        self.initial_location=location

    def update_Ttime(self,time):
        self.total_time+=time

    def update_Tdis(self,location):
        self.total_distance+=location

    def getAvgSpeed(self):
        return self.total_distance/self.total_time

def pt_from_file(filename):
    lines = read_txt(filename)
    gtt=Pt()
    for line in lines:
        new_line=line.split(' ')
        busID=new_line[0]
        lineID=new_line[1]
        location=(int(new_line[2]),int(new_line[3]))
        time=int(new_line[4])
        gtt.addBus(Bus(busID,lineID),location,time)
    return gtt

def read_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
        return map(lambda x:x.strip(),lines)

if __name__ == '__main__': 
    filename=sys.argv[1]
    flag=sys.argv[2]
    id=sys.argv[3]
    gtt=pt_from_file(filename)
    if flag=='-b':
        gtt.getDistance(id)
    if flag=='-l':
        gtt.getLineAvgSpeed(id)