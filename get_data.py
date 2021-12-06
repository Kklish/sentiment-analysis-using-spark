from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import * 
import json
import SGDlog_classifier

import SGDhinge_classifier

import test_runner_module

import naive_multinomial

import PAC_classifier as PassiveAggressive_Classifier

import KMeans_classifier

def stream_data_processing(rdd,id_inp):
    if not rdd.isEmpty():
            
            json_temp = rdd.collect()

            for records in json_temp:
                json_list = json.loads(records)

            rows = []
            for i in json_list.keys():
                instance = []
                instance.append(str(json_list[i]['feature1']))
                instance.append(str(json_list[i]['feature0']).strip(' '))
                rows.append(tuple(instance))
            print("\nStreaming batch size of {} received\n".format(len(rows)))       
            if id_inp == 1:
                SGDlog_classifier.SGD_Model(rows,spark_context)
            elif id_inp == 2:
                SGDhinge_classifier.SGDhinge_Model(rows,spark_context)
            elif id_inp == 3:
                naive_multinomial.multinomial_model(rows,spark_context)
            elif id_inp == 4:
                PassiveAggressive_Classifier.passiAggrClass_model(rows,spark_context)
            elif id_inp ==5 :
                KMeans_classifier.KMeans_model(rows,spark_context)
            elif id_inp==6:
                test_runner_module.test_model(rows,spark_context)

if __name__=="__main__" :
            id_inp = 5
            print("Available models:\n")
            print("SGDlog_classifier:'1'\n")
            print("SGDhinge_classifier:'2'\n")
            print("naive_Multinomial:'3'\n")
            print("PassiveAggressive_Classifier:'4'\n")
            print("KMeans_Classifier:'5'\n")
            print("Test_runner:'6'\n")
            print("Enter your Choice : ",id_inp)
            
            spark_context = SparkContext("local[2]","test")
            streaming_sparkcontext = StreamingContext(spark_context,5)
            spark = SparkSession(spark_context)
            
            lines = streaming_sparkcontext.socketTextStream('localhost',6100)
            line = lines.flatMap(lambda line: line.split("\n"))
            line.foreachRDD(lambda rdd : stream_data_processing(rdd,id_inp))
            streaming_sparkcontext.start()
            streaming_sparkcontext.awaitTermination()