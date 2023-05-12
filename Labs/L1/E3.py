import sys

class Birth():
    def __init__(self, name, surname, birthplace, birthdate):
        self.name=name 
        self.surname=surname 
        self.birthplace=birthplace 
        self.birthdate=(birthdate) 

def births_from_file(filename):
    births=[]
    lines = read_txt(filename)
    for line in lines:
        name, surname, birthplace, birthdate=line.split(' ')
        births.append(Birth(name, surname, birthplace, birthdate))
    return births

def read_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
        return map(lambda x:x.strip(),lines)
    
def births_per_city(births):
    cities={}
    for i in births:
        if i.birthplace in cities:
            cities[i.birthplace]+=1
        else:
            cities[i.birthplace]=1
    
    return cities

def births_per_month(births):
    months={}
    for i in births:
        month=i.birthdate.split('/')[1]
        if month in months:
            months[month]+=1
        else:
            months[month]=1
    return months
    

def avgBirths_per_city(births):
    cities=births_per_city(births)
    return(sum(cities.values())/len(cities))

if __name__ == '__main__': 
    filename=sys.argv[1]
    births=births_from_file(filename=filename)
    print("Births per city:", births_per_city(births=births))
    print("Births per month:", births_per_month(births))
    print("Avg number of births:",avgBirths_per_city(births))