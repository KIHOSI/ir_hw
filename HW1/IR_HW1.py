import requests
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer

url = "https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt" # 目標網址
resp = requests.get(url)
soup = BeautifulSoup(resp.text,'html.parser') #抓html

# tokenize
# split non-alphanumeric characters(whitespace and punctuation)
# \W is the character class for all Non-Word characters, '.text' 轉成string型態
splitString = re.split('\W+',soup.text) 

# lowercasing
lowerString = [elements.lower() for elements in splitString]

# stemming
stemmer = PorterStemmer()
stemString = [stemmer.stem(lowerstring) for lowerstring in lowerString]

# stopword removal
url2 = "http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words" 
resp2 = requests.get(url2)
soup = BeautifulSoup(resp2.text,'html.parser') 
stopwordList = soup.text # 抓取stopword list
removeStopword = [word for word in stemString if word not in stopwordList]
print(removeStopword)

# output list to txt file
with open('filtered_file.txt','w') as f:
    for item in removeStopword:
        f.write("%s\n" % item)
