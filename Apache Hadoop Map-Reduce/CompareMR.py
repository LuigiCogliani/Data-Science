"""Task 01 - histogram
for this task we will need to create a histogram where each bin will group the transactions happened each month,
throughout the entire dataset.
"""
from mrjob.job import MRJob
import re
import time

#this is a regular expression that finds all the words inside a String
WORD_REGEX = re.compile(r"\b\w+\b")

#This line declares the class Task01, that extends the MRJob format.
class Task01(MRJob):

# this class will define the mapper method
    def mapper(self, _, line):
        #the mapper receives in input each line of the datasets, one line at the time. This command will split the line into a list, using coma as separator
        fields = line.split(",")
        try:
            #This line will filter the good lines (only the lines that returns 9 fields once split, as we are dealing with a dataset with 9 columns)
            if (len(fields)==9):
                #convert the time epoch into year and month
                time_epoch = int(fields[7])
                month = time.strftime("%m",time.gmtime(time_epoch)) #returns month
                year = time.strftime("%y", time.gmtime(time_epoch))  # returns year
                N_trans = int(fields[8])
                key = (year, month)
                #The mapper output will be a key value pair where the key is a year-month tuple and the value is the number of transactions
                yield(key, N_trans)

        except:
            pass

# this class will define the combiner method. As we are doing a basic count, this program will benefit greatly of a combiner
    def combiner(self, key, N_trans):
        # the combiner will sum over the keys received from it's mapper
        yield (key,sum(N_trans))
# this class will define the reducer method
    def reducer(self, key, N_trans):
        #after shuffle and sort, each reducer will receive all the pairs within certain keys. For each key they will sum the values.
        yield (key,sum(N_trans))

"""
the reducer output will be again a key value pair in the form of:
(year - month, total number of transadtions in that specific month)
As per default, we used 3 reducers for this job, hence we will have 3 output files that will need to be merged AND sorted before plotting the histogram.
"""

if __name__ == '__main__':
    Task01.JOBCONF= { 'mapreduce.job.reduces': '3' }
    Task01.run()
