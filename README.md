# Analyzing Stock SEC Filings

**Hypothesis:** 10Ks are long and boring to read, but contain valuable information about the kinds of risks a company is facing. Research (and practical experience) suggests that stock price is correlated with the year over year change in 10K language and with negative sentiment increase in the 10K. Fortunately, Natural Language Processing (NLP) is a machine-learning technique that analyzes bodies of texts and their positive and negative sentiments. The aim is to use NLP to analyze 10Ks and find possible arbitrage opportunities in the market. 

Along the way, we provide some functionality for private investors like my dad (who inspired this project to begin with).

##Contents 
- ```research```: file containing some preliminary notes and resources for initial research purposes.
- ```data_scrape_notebooks```: This file contains Jupyter Notebooks that gather and clean data from the SEC's website. Most of the code (save a few edits) come from [this](https://www.quantopian.com/posts/scraping-10-ks-and-10-qs-for-alpha) fantastic source. Also here is a python script where I used a lot of the code from the notebook to automate the data collection and to spit out a pdf of the similarity scores for each ticker of interest. 
- ```similarity_analysis.py```: using the data scraping notebooks, I put all the relevant functions in a giant script to automate the data collection/cleaning process for tickers of interest. The ```similarity_analysis_windows.py``` script is my attempt to make this script usable on my dad's vanilla PC. If you like reading about programmers in distress, read ```trials_tribulations.md``` to see my account of that process. 
- ```data```: Using the script above, I collected and cleaned data for three different companies: Google, Goldman Sachs, and Tesla. Also here are the outputs of the script (the pdf files showing ssimilarity values). 

  
