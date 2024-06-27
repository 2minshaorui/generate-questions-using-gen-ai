import requests 
from bs4 import BeautifulSoup
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# web scraping input URL for context 
def extract_text_from_webpage_content(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, "html.parser")
  paragraphs = soup.find_all("p")
  text = " ".join([p.get_text() for p in paragraphs])
  return text

# pre-process input text into sentences & words using NLTK library 
def preprocess_extracted_text(text):
  # tokenise text 
  stopWords = set(stopwords.words("english"))
  words = word_tokenize(text)

  # create a frequency table to keep score of each word 
  freqTable = dict()                 
  for word in words:               
    word = word.lower()                 
    if word in stopWords:                 
      continue                  
    if word in freqTable:                       
      freqTable[word] += 1            
    else:          
      freqTable[word] = 1

  # create a dictionary to keep score of each sentence 
  sentences = sent_tokenize(text)                 
  sentenceValue = dict()                     
  for sentence in sentences:               
    for word, freq in freqTable.items():              
      if word in sentence.lower():           
        if sentence in sentenceValue:                                 
          sentenceValue[sentence] += freq                       
        else:                       
          sentenceValue[sentence] = freq                    

  sumValues = 0                        
  for sentence in sentenceValue:              
    sumValues += sentenceValue[sentence] 

  # define the average value from original input text 
  average = int(sumValues / len(sentenceValue))
  return (sentences, sentenceValue, average)

# store sentences into summary 
def summarise(sentences, sentenceValue, average):
  summary = ""
  for sentence in sentences: 
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.3*average)):
      summary += " " + sentence
  return summary

def main():
  print("Starting up question generator...")

  parser = argparse.ArgumentParser(description="URL of webpage:")
  parser.add_argument("input_string", type=str, help="URL of webpage to generate questions")
  args = parser.parse_args()

  print(f"Web scraping from: {args.input_string}")
  pageContent = extract_text_from_webpage_content(args.input_string)

  print("Summarising...")
  (sentences, sentenceValue, average) = preprocess_extracted_text(pageContent)
  summary = summarise(sentences, sentenceValue, average)
  print(summary)

  print("Summary generated")

if __name__ == "__main__":
    main()
