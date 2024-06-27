import requests 
from bs4 import BeautifulSoup
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# web scraping input URL for context 
def extract_text_from_webpage_content(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, "html.parser")
  paragraphs = soup.find_all("p")
  text = " ".join([p.get_text() for p in paragraphs])
  return text

def summarise(text):
  # initialize tokenizer model 
  tokenizer = AutoTokenizer.from_pretrained('t5-base')                        
  model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', return_dict=True)
  
  # tokenize data 
  inputs = tokenizer.encode("summarize:" + text, return_tensors='pt', max_length=512, truncation=True)

  summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
  summary = tokenizer.decode(summary_ids[0])
  return summary

def main():
  print("Starting up T5-based summariser...")

  parser = argparse.ArgumentParser(description="URL of webpage:")
  parser.add_argument("input_string", type=str, help="URL of webpage to generate questions")
  args = parser.parse_args()

  print(f"Web scraping from: {args.input_string}")
  pageContent = extract_text_from_webpage_content(args.input_string)

  print("Summarising...")
  summary = summarise(pageContent)
  print(summary)

  print("Summary generated")

if __name__ == "__main__":
    main()
