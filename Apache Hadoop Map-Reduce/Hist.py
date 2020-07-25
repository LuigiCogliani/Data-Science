"""Task 01 - histogram
"""
from mrjob.job import MRJob
import re
import time

#this is a regular expression that finds all the words inside a String
WORD_REGEX = re.compile(r"\b\w+\b")

#This line declares the class Lab3, that extends the MRJob format.
class Task01(MRJob):

# this class will define two additional methods: the mapper method goes here
    def mapper(self, _, line):
        fields = line.split(",")
        try:
            if (len(fields)==9):
            #access the fields you want, assuming the format is correct now
                time_epoch = int(fields[7])
                month = time.strftime("%m",time.gmtime(time_epoch)) #returns month
                year = time.strftime("%y", time.gmtime(time_epoch))  # returns year
                day = time.strftime("%d", time.gmtime(time_epoch))  # returns day
                gas = (int(fields[6]),1)
                key = (year, month, day)
                yield(key, gas)

        except:
            pass
            #no need to do anything, just ignore the line, as it was malformed

    def combiner(self, feature, values):
        count = 0
        total = 0
        for value in values:
            count += value[1]
            total += value[0]
        yield (feature, (total, count) )


        #and the reducer method goes after this line
    def reducer(self, feature, values):
        count = 0
        total = 0
        for value in values:
            count += value[1]
            total += value[0]
        yield (feature, total/count)

#this part of the python script tells to actually run the defined MapReduce job. Note that Lab1 is the name of the class
if __name__ == '__main__':
    Task01.JOBCONF= { 'mapreduce.job.reduces': '3' }
    Task01.run()
