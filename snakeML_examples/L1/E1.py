#LABORATORY 1 - EXERCISE 1 - MANUEL ESCOBAR
import sys
from snakeML.loads import read_txt

class competitor:
    def __init__(self, name,country,scores):
        self.name=name
        self.country=country
        self.scores=scores
        self.scores.sort()
        self.final_score=sum(scores[1:4])

class event:
    competitors=[]
    countries={}

    def add_competitor(self, competitor):
        self.competitors.append(competitor)
        if competitor.country in self.countries:
            self.countries[competitor.country]+=competitor.final_score
        else:
            self.countries[competitor.country]=competitor.final_score
    
    def rank_competitors(self,n):
        self.competitors.sort(key=lambda x: x.final_score, reverse=True)
        print("Final ranking:")
        for i in range(n):
            print(self.competitors[i].name,"| Score: ",self.competitors[i].final_score)

    def best_country(self):
        print("Best Country:")
        max_value = max(self.countries.values () )
        max_key = max (self.countries, key=self.countries.get)
        print (max_key,"| Score: ",max_value)

def event_from_file(filename):
    the_event=event()
    lines=read_txt(filename)
    for line in lines:
        new_line=line.split(' ')
        name=new_line[0]+' '+new_line[1]
        country=new_line[2]
        scores=[]
        for i in new_line[3:]:
            scores.append(float(i))
        the_event.add_competitor(competitor=competitor(name,country,scores))
    return the_event

if __name__ == '__main__': 
    filename=sys.argv[1]
    gymnastic=event_from_file(filename)
    gymnastic.rank_competitors(3)
    gymnastic.best_country()

