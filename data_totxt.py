import re
import string
import csv
def text_cleaner(text):
    text = re.sub(r'\s+\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\.]', ' ', text)
    text = "".join(car for car in text if car not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    return text

# get Amazon review data
amazon_reviews=[]
with open('amazon_dataset.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)      
    f = open("amazon_reviews.txt", "a")  
    for row in reader:
        #print(row[2])
        #amazon_reviews.append(row[2])
        f.write(text_cleaner(row[2])+'\n')
    f.close()
