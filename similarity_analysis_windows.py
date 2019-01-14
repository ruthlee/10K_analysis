import re
import os
from time import gmtime, strftime
from datetime import datetime, timedelta
import unicodedata
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import bs4 as bs
from lxml import html
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# This is going to be a massive script that basically defines a bunch of huge data scraping functions and at the end of it makes some 10K similarity plots. Let's get started. Note that full documentation of these functions can be found within the Jupyter Notebooks. 

def TickertoCIK(tickers):
    url = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
    cik_re = re.compile(r'.*CIK=(\d{10}).*')

    cik_dict = {}
    for ticker in tqdm(tickers): # Use tqdm lib for progress bar
        results = cik_re.findall(requests.get(url.format(ticker)).text)
        if len(results):
            cik_dict[str(ticker).lower()] = str(results[0])
    
    return cik_dict

# GLOBAL VARIABLES DEFINED HERE:
cwd = os.getcwd() 
if not os.path.exists(cwd + r'\10K'):
    os.makedirs(cwd + r'\10K')
if not os.path.exists(cwd + r'\10Q'):
    os.makedirs(cwd + r'\10Q')
pathname_10k = cwd + r'\10K' 
pathname_10q = cwd + r'\10Q'  

# websites to scrape data (used later)
browse_url_base_10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-K'
filing_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
doc_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

ticker_list = input("Enter tickers here separated by commas.\n")
ticker_list = ticker_list.split(',')
cik_dict = TickertoCIK(list(ticker_list))

# Create dataframe for below functions
ticker_cik_df = pd.DataFrame.from_dict(data = cik_dict, orient='index')
ticker_cik_df.reset_index(inplace=True)
ticker_cik_df.columns = ['ticker', 'cik']
ticker_cik_df['cik'] = [str(cik) for cik in ticker_cik_df['cik']]

def WriteLogFile(log_file_name, text):
    
    '''
    Helper function.
    Writes a log file with all notes and
    error messages from a scraping "session".
    
    Parameters
    ----------
    log_file_name : str
        Name of the log file (should be a .txt file).
    text : str
        Text to write to the log file.
        
    Returns
    -------
    None.
    
    '''
    
    with open(log_file_name, "a") as log_file:
        log_file.write(text)

    return


def Scrape10K(browse_url_base, filing_url_base, doc_url_base, cik, log_file_name):
    
    '''
    Scrapes all 10-Ks and 10-K405s for a particular 
    CIK from EDGAR.
    
    Parameters
    ----------
    browse_url_base : str
        Base URL for browsing EDGAR.
    filing_url_base : str
        Base URL for filings listings on EDGAR.
    doc_url_base : str
        Base URL for one filing's document tables
        page on EDGAR.
    cik : str
        Central Index Key.
    log_file_name : str
        Name of the log file (should be a .txt file).
        
    Returns
    -------
    None.
    
    '''
    
    # Check if we've already scraped this CIK
    try:
        os.mkdir(cik)
    except OSError:
        print("Already scraped CIK", cik)
        return
    
    # If we haven't, go into the directory for that CIK
    os.chdir(cik)
    
    print('Scraping CIK', cik)
    
    # Request list of 10-K filings
    res = requests.get(browse_url_base % cik)
    
    # If the request failed, log the failure and exit
    if res.status_code != 200:
        os.chdir('..')
        os.rmdir(cik) # remove empty dir
        text = "Request failed with error code " + str(res.status_code) + \
               "\nFailed URL: " + (browse_url_base % cik) + '\n'
        WriteLogFile(log_file_name, text)
        return

    # If the request doesn't fail, continue...
    
    # Parse the response HTML using BeautifulSoup
    soup = bs.BeautifulSoup(res.text, "lxml")

    # Extract all tables from the response
    html_tables = soup.find_all('table')
    
    # Check that the table we're looking for exists
    # If it doesn't, exit
    if len(html_tables)<3:
        os.chdir('..')
        return
    
    # Parse the Filings table
    filings_table = pd.read_html(str(html_tables[2]), header=0)[0]
    filings_table['Filings'] = [str(x) for x in filings_table['Filings']]

    # Get only 10-K and 10-K405 document filings
    filings_table = filings_table[(filings_table['Filings'] == '10-K') | (filings_table['Filings'] == '10-K405')]

    # If filings table doesn't have any
    # 10-Ks or 10-K405s, exit
    if len(filings_table)==0:
        os.chdir('..')
        return
    
    # Get accession number for each 10-K and 10-K405 filing
    filings_table['Acc_No'] = [x.replace('\xa0',' ')
                               .split('Acc-no: ')[1]
                               .split(' ')[0] for x in filings_table['Description']]

    # Iterate through each filing and 
    # scrape the corresponding document...
    for index, row in filings_table.iterrows():
        
        # Get the accession number for the filing
        acc_no = str(row['Acc_No'])
        
        # Navigate to the page for the filing
        docs_page = requests.get(filing_url_base % (cik, acc_no))
        
        # If request fails, log the failure
        # and skip to the next filing
        if docs_page.status_code != 200:
            os.chdir('..')
            text = "Request failed with error code " + str(docs_page.status_code) + \
                   "\nFailed URL: " + (filing_url_base % (cik, acc_no)) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue

        # If request succeeds, keep going...
        
        # Parse the table of documents for the filing
        docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
        docs_html_tables = docs_page_soup.find_all('table')
        if len(docs_html_tables)==0:
            continue
        docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
        docs_table['Type'] = [str(x) for x in docs_table['Type']]
        
        # Get the 10-K and 10-K405 entries for the filing
        docs_table = docs_table[(docs_table['Type'] == '10-K') | (docs_table['Type'] == '10-K405')]
        
        # If there aren't any 10-K or 10-K405 entries,
        # skip to the next filing
        if len(docs_table)==0:
            continue
        # If there are 10-K or 10-K405 entries,
        # grab the first document
        elif len(docs_table)>0:
            docs_table = docs_table.iloc[0]
        
        docname = docs_table['Document']
        
        # If that first entry is unavailable,
        # log the failure and exit
        if str(docname) == 'nan':
            os.chdir('..')
            text = 'File with CIK: %s and Acc_No: %s is unavailable' % (cik, acc_no) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue       
        
        # If it is available, continue...
        
        # Request the file
        file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))
        
        # If the request fails, log the failure and exit
        if file.status_code != 200:
            os.chdir('..')
            text = "Request failed with error code " + str(file.status_code) + \
                   "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue
        
        # If it succeeds, keep going...
        
        # Save the file in appropriate format
        if '.txt' in docname:
            # Save text as TXT
            date = str(row['Filing Date'])
            filename = cik + '_' + date + '.txt'
            html_file = open(filename, 'a')
            html_file.write(file.text)
            html_file.close()
        else:
            # Save text as HTML
            date = str(row['Filing Date'])
            filename = cik + '_' + date + '.html'
            html_file = open(filename, 'a')
            html_file.write(file.text)
            html_file.close()
        
    # Move back to the main 10-K directory
    os.chdir('..')
        
    return

def Scrape10Q(browse_url_base, filing_url_base, doc_url_base, cik, log_file_name):
    
    '''
    Scrapes all 10-Qs for a particular CIK from EDGAR.
    
    Parameters
    ----------
    browse_url_base : str
        Base URL for browsing EDGAR.
    filing_url_base : str
        Base URL for filings listings on EDGAR.
    doc_url_base : str
        Base URL for one filing's document tables
        page on EDGAR.
    cik : str
        Central Index Key.
    log_file_name : str
        Name of the log file (should be a .txt file).
        
    Returns
    -------
    None.
    
    '''
    
    # Check if we've already scraped this CIK
    try:
        os.mkdir(cik)
    except OSError:
        print("Already scraped CIK", cik)
        return
    
    # If we haven't, go into the directory for that CIK
    os.chdir(cik)
    
    print('Scraping CIK', cik)
    
    # Request list of 10-Q filings
    res = requests.get(browse_url_base % cik)
    
    # If the request failed, log the failure and exit
    if res.status_code != 200:
        os.chdir('..')
        os.rmdir(cik) # remove empty dir
        text = "Request failed with error code " + str(res.status_code) + \
               "\nFailed URL: " + (browse_url_base % cik) + '\n'
        WriteLogFile(log_file_name, text)
        return
    
    # If the request doesn't fail, continue...

    # Parse the response HTML using BeautifulSoup
    soup = bs.BeautifulSoup(res.text, "lxml")

    # Extract all tables from the response
    html_tables = soup.find_all('table')
    
    # Check that the table we're looking for exists
    # If it doesn't, exit
    if len(html_tables)<3:
        print("table too short")
        os.chdir('..')
        return
    
    # Parse the Filings table
    filings_table = pd.read_html(str(html_tables[2]), header=0)[0]
    filings_table['Filings'] = [str(x) for x in filings_table['Filings']]

    # Get only 10-Q document filings
    filings_table = filings_table[filings_table['Filings'] == '10-Q']

    # If filings table doesn't have any
    # 10-Ks or 10-K405s, exit
    if len(filings_table)==0:
        os.chdir('..')
        return
    
    # Get accession number for each 10-K and 10-K405 filing
    filings_table['Acc_No'] = [x.replace('\xa0',' ')
                               .split('Acc-no: ')[1]
                               .split(' ')[0] for x in filings_table['Description']]

    # Iterate through each filing and 
    # scrape the corresponding document...
    for index, row in filings_table.iterrows():
        
        # Get the accession number for the filing
        acc_no = str(row['Acc_No'])
        
        # Navigate to the page for the filing
        docs_page = requests.get(filing_url_base % (cik, acc_no))
        
        # If request fails, log the failure
        # and skip to the next filing    
        if docs_page.status_code != 200:
            os.chdir('..')
            text = "Request failed with error code " + str(docs_page.status_code) + \
                   "\nFailed URL: " + (filing_url_base % (cik, acc_no)) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue
            
        # If request succeeds, keep going...
        
        # Parse the table of documents for the filing
        docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
        docs_html_tables = docs_page_soup.find_all('table')
        if len(docs_html_tables)==0:
            continue
        docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
        docs_table['Type'] = [str(x) for x in docs_table['Type']]
        
        # Get the 10-K and 10-K405 entries for the filing
        docs_table = docs_table[docs_table['Type'] == '10-Q']
        
        # If there aren't any 10-K or 10-K405 entries,
        # skip to the next filing
        if len(docs_table)==0:
            continue
        # If there are 10-K or 10-K405 entries,
        # grab the first document
        elif len(docs_table)>0:
            docs_table = docs_table.iloc[0]
        
        docname = docs_table['Document']
        
        # If that first entry is unavailable,
        # log the failure and exit
        if str(docname) == 'nan':
            os.chdir('..')
            text = 'File with CIK: %s and Acc_No: %s is unavailable' % (cik, acc_no) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue       
        
        # If it is available, continue...
        
        # Request the file
        file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))
        
        # If the request fails, log the failure and exit
        if file.status_code != 200:
            os.chdir('..')
            text = "Request failed with error code " + str(file.status_code) + \
                   "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue
            
        # If it succeeds, keep going...
        
        # Save the file in appropriate format
        if '.txt' in docname:
            # Save text as TXT
            date = str(row['Filing Date'])
            filename = cik + '_' + date + '.txt'
            html_file = open(filename, 'a')
            html_file.write(file.text)
            html_file.close()
        else:
            # Save text as HTML
            date = str(row['Filing Date'])
            filename = cik + '_' + date + '.html'
            html_file = open(filename, 'a')
            html_file.write(file.text)
            html_file.close()
        
    # Move back to the main 10-Q directory
    os.chdir('..')
        
    return

# Functions to clean up scraped data
def RemoveNumericalTables(soup):
    
    '''
    Removes tables with >15% numerical characters.
    
    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.
        
    Returns
    -------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup
        with numerical tables removed.
        
    '''
    
    # Determines percentage of numerical characters
    # in a table
    def GetDigitPercentage(tablestring):
        if len(tablestring)>0.0:
            numbers = sum([char.isdigit() for char in tablestring])
            length = len(tablestring)
            return numbers/length
        else:
            return 1
    
    # Evaluates numerical character % for each table
    # and removes the table if the percentage is > 15%
    [x.extract() for x in soup.find_all('table') if GetDigitPercentage(x.get_text())>0.15]
    
    return soup

def RemoveTags(soup):
    
    '''
    Drops HTML tags, newlines and unicode text from
    filing text.
    
    Parameters
    ----------a
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.
        
    Returns
    -------
    text : str
        Filing text.
        
    '''
    
    # Remove HTML tags with get_text
    text = soup.get_text()
    
    # Remove newline characters
    text = text.replace('\n', ' ')
    
    # Replace unicode characters with their
    # "normal" representations
    text = unicodedata.normalize('NFKD', text)
    
    return text

def ConvertHTML(cik):
    
    '''
    Removes numerical tables, HTML tags,
    newlines, unicode text, and XBRL tables.
    
    Parameters
    ----------
    cik : str
        Central Index Key used to scrape files.
    
    Returns
    -------
    None.
    
    '''
    
    # Look for files scraped for that CIK
    try: 
        os.chdir(cik)
    # ...if we didn't scrape any files for that CIK, exit
    except FileNotFoundError:
        print("Could not find directory for CIK", cik)
        return
        
    print("Parsing CIK %s..." % cik)
    parsed = False # flag to tell if we've parsed anything
    
    # Try to make a new directory within the CIK directory
    # to store the text representations of the filings
    try:
        os.mkdir('rawtext')
    # If it already exists, continue
    # We can't exit at this point because we might be
    # partially through parsing text files, so we need to continue
    except OSError:
        pass
    
    # Get list of scraped files
    # excluding hidden files and directories
    file_list = [fname for fname in os.listdir() if not (fname.startswith('.') | os.path.isdir(fname))]
    
    # Iterate over scraped files and clean
    for filename in file_list:
            
        # Check if file has already been cleaned
        new_filename = filename.replace('.html', '.txt')
        text_file_list = os.listdir('rawtext')
        if new_filename in text_file_list:
            continue
        
        # If it hasn't been cleaned already, keep going...
        
        # Clean file
        with codecs.open(filename, encoding='utf-8-sig', mode = 'r') as file:
            parsed = True
            soup = bs.BeautifulSoup(file.read(), "lxml")
            soup = RemoveNumericalTables(soup)
            text = RemoveTags(soup)
            with open('rawtext/'+new_filename, 'w') as newfile:
                newfile.write(text)
    
    # If all files in the CIK directory have been parsed
    # then log that
    if parsed==False:
        print("Already parsed CIK", cik)
    
    os.chdir('..')
    return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('NOW BEGINNING DATA SCRAPING')
print('Scraping 10Ks...')
# Run the function to scrape 10-Ks

# Define parameters
browse_url_base_10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-K'
filing_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
doc_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

# Set correct directory
os.chdir(pathname_10k)

# Initialize log file
# (log file name = the time we initiate scraping session)
time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
log_file_name = 'log '+time+'.txt'
with open(log_file_name, 'a') as log_file:
    log_file.close()

# Iterate over CIKs and scrape 10-Ks
for cik in tqdm(ticker_cik_df['cik']):
    Scrape10K(browse_url_base=browse_url_base_10k, 
          filing_url_base=filing_url_base_10k, 
          doc_url_base=doc_url_base_10k, 
          cik=cik,
          log_file_name=log_file_name)

print('Scraping 10Qs...')
# Run the function to scrape 10-Qs

# Define parameters
browse_url_base_10q = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-Q&count=1000'
filing_url_base_10q = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
doc_url_base_10q = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

# Set correct directory (fill this out yourself!)
os.chdir(pathname_10q)

# Initialize log file
# (log file name = the time we initiate scraping session)
time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
log_file_name = 'log '+time+'.txt'
log_file = open(log_file_name, 'a')
log_file.close()

# Iterate over CIKs and scrape 10-Qs
for cik in tqdm(ticker_cik_df['cik']):
    Scrape10Q(browse_url_base=browse_url_base_10q, 
          filing_url_base=filing_url_base_10q, 
          doc_url_base=doc_url_base_10q, 
          cik=cik,
          log_file_name=log_file_name)


print('Converting and cleaning html files...')

# For 10-Ks...

os.chdir(pathname_10k)

# Iterate over CIKs and clean HTML filings
for cik in tqdm(ticker_cik_df['cik']):
    ConvertHTML(cik)

# For 10-Qs...

os.chdir(pathname_10q)

# Iterate over CIKs and clean HTML filings
for cik in tqdm(ticker_cik_df['cik']):
    ConvertHTML(cik)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# SIMILARIY SCORE ANALYSIS
def ComputeCosineSimilarity(words_A, words_B):
    
    '''
    Compute cosine similarity between document A and
    document B.
    
    Parameters
    ----------
    words_A : set
        Words in document A.
    words_B : set
        Words in document B
        
    Returns
    -------
    cosine_score : float
        Cosine similarity between document
        A and document B.
        
    '''
    
    # Compile complete set of words in A or B
    words = list(words_A.union(words_B))
    
    # Determine which words are in A
    vector_A = [1 if x in words_A else 0 for x in words]
    
    # Determine which words are in B
    vector_B = [1 if x in words_B else 0 for x in words]
    
    # Compute cosine score using scikit-learn
    array_A = np.array(vector_A).reshape(1, -1)
    array_B = np.array(vector_B).reshape(1, -1)
    cosine_score = cosine_similarity(array_A, array_B)[0,0]
    
    return cosine_score


def ComputeJaccardSimilarity(words_A, words_B):
    
    '''
    Compute Jaccard similarity between document A and
    document B.
    
    Parameters
    ----------
    words_A : set
        Words in document A.
    words_B : set
        Words in document B
        
    Returns
    -------
    jaccard_score : float
        Jaccard similarity between document
        A and document B.
        
    '''
    
    # Count number of words in both A and B
    words_intersect = len(words_A.intersection(words_B))
    
    # Count number of words in A or B
    words_union = len(words_A.union(words_B))
    
    # Compute Jaccard similarity score
    jaccard_score = words_intersect / words_union
    
    return jaccard_score

def ComputeSimilarityScores10K(cik):
    
    '''
    Computes cosine and Jaccard similarity scores
    over 10-Ks for a particular CIK.
    
    Parameters
    ----------
    cik: str
        Central Index Key used to scrape and name
        files.
        
    Returns
    -------
    None.
    
    '''
    
    # Open the directory that holds plaintext
    # filings for the CIK
    os.chdir(cik+'/rawtext')
    print("Parsing CIK %s..." % cik)
    
    # Get list of files to over which to compute scores
    # excluding hidden files and directories
    file_list = [fname for fname in os.listdir() if not 
                 (fname.startswith('.') | os.path.isdir(fname))]
    file_list.sort()
    
    # Check if scores have already been calculated...
    try:
        os.mkdir('../metrics')
    # ... if they have been, exit
    except OSError:
        print("Already parsed CIK %s..." % cik)
        os.chdir('../..')
        return
    
    # Check if enough files exist to compute sim scores...
    # If not, exit
    if len(file_list) < 2:
        print("No files to compare for CIK", cik)
        os.chdir('../..')
        return
    
    # Initialize dataframe to store sim scores
    dates = [x[-14:-4] for x in file_list]
    cosine_score = [0]*len(dates)
    jaccard_score = [0]*len(dates)
    data = pd.DataFrame(columns={'cosine_score': cosine_score, 
                                 'jaccard_score': jaccard_score},
                       index=dates)
        
    # Open first file
    file_name_A = file_list[0]
    with open(file_name_A, 'r') as file:
        file_text_A = file.read()
        
    # Iterate over each 10-K file...
    for i in range(1, len(file_list)):

        file_name_B = file_list[i]

        # Get file text B
        with open(file_name_B, 'r') as file:
            file_text_B = file.read()

        # Get set of words in A, B
        words_A = set(re.findall(r"[\w']+", file_text_A))
        words_B = set(re.findall(r"[\w']+", file_text_B))

        # Calculate similarity scores
        cosine_score = ComputeCosineSimilarity(words_A, words_B)
        jaccard_score = ComputeJaccardSimilarity(words_A, words_B)

        # Store score values
        date_B = file_name_B[-14:-4]
        data.at[date_B, '10Kdates'] = date_B
        data.at[date_B, 'cosine_score'] = cosine_score
        data.at[date_B, 'jaccard_score'] = jaccard_score

        # Reset value for next loop
        # (We don't open the file again, for efficiency)
        file_text_A = file_text_B

    # Save scores
    os.chdir('../metrics')
    data.to_csv(cik+'_sim_scores.csv', index=False)
    os.chdir('../..')

def ComputeSimilarityScores10Q(cik):
    
    '''
    Computes cosine and Jaccard similarity scores
    over 10-Qs for a particular CIK.
    
    Compares each 10-Q to the 10-Q from the same
    quarter of the previous year.
    
    Parameters
    ----------
    cik: str
        Central Index Key used to scrape and name
        files.
        
    Returns
    -------
    None.
    
    '''
    
    # Define how stringent we want to be about 
    # "previous year"
    year_short = timedelta(345)
    year_long = timedelta(385)
    
    # Open directory that holds plain 10-Q textfiles
    # for the CIK
    os.chdir(cik+'/rawtext')
    print("Parsing CIK %s..." % cik)
    
    # Get list of files to compare
    file_list = [fname for fname in os.listdir() if not 
                 (fname.startswith('.') | os.path.isdir(fname))]
    file_list.sort()
    
    # Check if scores have already been calculated
    try:
        os.mkdir('../metrics')
    # ... if they have already been calculated, exit
    except OSError:
        print("Already parsed CIK %s..." % cik)
        os.chdir('../..')
        return
    
    # Check if enough files exist to compare
    # ... if there aren't enough files, exit
    if len(file_list) < 4:
        print("No files to compare for CIK", cik)
        os.chdir('../..')
        return
    
    # Initialize dataframe to hold similarity scores
    dates = [x[-14:-4] for x in file_list]
    cosine_score = [0]*len(dates)
    jaccard_score = [0]*len(dates)
    data = pd.DataFrame(columns={'cosine_score': cosine_score, 
                                 'jaccard_score': jaccard_score},
                       index=dates)
    
    # Iterate over each quarter...
    for j in range(3):
        
        # Get text and date of earliest filing from that quarter
        file_name_A = file_list[j]
        with open(file_name_A, 'r') as file:
            file_text_A = file.read()
        date_A = datetime.strptime(file_name_A[-14:-4], '%Y-%m-%d')
        
        # Iterate over the rest of the filings from that quarter...
        for i in range(j+3, len(file_list), 3):

            # Get name and date of the later file
            file_name_B = file_list[i]
            date_B = datetime.strptime(file_name_B[-14:-4], '%Y-%m-%d')
            
            # If B was not filed within ~1 year after A...
            if (date_B > (date_A + year_long)) or (date_B < (date_A + year_short)):
                
                print(date_B.strftime('%Y-%m-%d'), "is not within a year of", date_A.strftime('%Y-%m-%d'))
                
                # Record values as NaN
                data.at[date_B.strftime('%Y-%m-%d'), 'cosine_score'] = 'NaN'
                data.at[date_B.strftime('%Y-%m-%d'), 'jaccard_score'] = 'NaN'
                
                # Pretend as if we found new date_A in the next year
                date_A = date_A.replace(year=date_B.year)
                
                # Move to next filing
                continue
                
            # If B was filed within ~1 year of A...
            
            # Get file text
            with open(file_name_B, 'r') as file:
                file_text_B = file.read()

            # Get sets of words in A, B
            words_A = set(re.findall(r"[\w']+", file_text_A))
            words_B = set(re.findall(r"[\w']+", file_text_B))

            # Calculate similarity score
            cosine_score = ComputeCosineSimilarity(words_A, words_B)
            jaccard_score = ComputeJaccardSimilarity(words_A, words_B)

            # Store value (indexing by the date of document B)
            data.at[date_B.strftime('%Y-%m-%d'), 'cosine_score'] = cosine_score
            data.at[date_B.strftime('%Y-%m-%d'), 'jaccard_score'] = jaccard_score

            # Reset value for next loop
            # Don't re-read files, for efficiency
            file_text_A = file_text_B
            date_A = date_B

    # Save scores
    os.chdir('../metrics')
    data.to_csv(cik+'_sim_scores.csv', index=True)
    os.chdir('../..')

# Computing scores for 10-Ks...

os.chdir(pathname_10k)

for cik in tqdm(ticker_cik_df['cik']):
    ComputeSimilarityScores10K(cik)

# Computing scores for 10-Qs...

os.chdir(pathname_10q)

for cik in tqdm(ticker_cik_df['cik']):
    ComputeSimilarityScores10Q(cik)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# CONSOLIDATING DATA

def GetData(cik, pathname_10k, pathname_10q, pathname_data):
    
    '''
    Consolidate 10-K and 10-Q data into a single dataframe
    for a CIK.
    
    Parameters
    ----------
    cik : str
        Central Index Key used to scrape and
        store data.
    pathname_10k : str
        Path to directory holding 10-K files.
    pathname_10q : str
        Path to directory holding 10-Q files.
    pathname_data : str
        Path to directory holding newly
        generated data files.
        
    Returns
    -------
    None.
    
    '''
    
    # Flags to determine what data we have
    data_10k = True
    data_10q = True
    
    print("Gathering data for CIK %s..." % cik)
    file_name = ('%s_sim_scores_full.csv' % cik)
    
    # Check if data has already been gathered...
    os.chdir(pathname_data)
    file_list = [fname for fname in os.listdir() if not fname.startswith('.')]
    
    # ... if it has been, exit
    if file_name in file_list:
        print("Already gathered data for CIK", cik)
        return
    
    # Try to get 10-K data...
    os.chdir(pathname_10k+'/%s/metrics' % cik)
    try:
        sim_scores_10k = pd.read_csv(cik+'_sim_scores.csv')
    # ... if it doesn't exist, set 10-K flag to False
    except FileNotFoundError:
        print("No data to gather.")
        data_10k = False
    
    # Try to get 10-Q data...
    os.chdir(pathname_10q+'/%s/metrics' % cik)
    try:
        sim_scores_10q = pd.read_csv(cik+'_sim_scores.csv')
    # ... if it doesn't exist, set 10-Q flag to False
    except FileNotFoundError:
        print("No data to gather.")
        data_10q = False
    
    # Merge depending on available data...
    # ... if there's no 10-K or 10-Q data, exit
    if not (data_10k and data_10q):
        return
    
    # ... if there's no 10-Q data (but there is 10-K data),
    # only use the 10-K data
    if not data_10q:
        sim_scores = sim_scores_10k
    # ... if the opposite is true, only use 10-Q data
    elif not data_10k:
        sim_scores = sim_scores_10q
    # ... if there's both 10-K and 10-Q data, merge
    elif (data_10q and data_10k):
        sim_scores = pd.concat([sim_scores_10k, sim_scores_10q], 
                           axis='index')
    
    # Rename date column
    sim_scores.rename(columns={'Unnamed: 0': '10Qdates'}, inplace=True)

    # Set CIK column
    sim_scores['cik'] = cik
    
    # Save file in the data dir
    os.chdir(pathname_data)
    sim_scores.to_csv('%s_sim_scores_full.csv' % cik, index=False)
    
    return

pathname_data = cwd
for cik in tqdm(ticker_cik_df['cik']):
    GetData(cik, pathname_10k, pathname_10q, pathname_data) 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PLOTTING SIMILARITY SCORES

# we need to turn this dataframe into a numpy array so that we can plot stuff. Let's clean up the data while 
# we're at it and create two numpy arrays for 10Q and 10K similarity data.

def get_10K_data(data):
    '''
    Helper function: 
    Takes in the pandas dataframe containing all the sim scores for 10Ks and 10Qs and generates a 
    cleaned up dataframe with just 10Ks of the form: 
    
    10Q Dates | cosine_score | jaccard_score
    
    '''
    df = data.copy()
    df = df[pd.notnull(df['10Kdates'])]
    df.drop(['10Qdates'], 1, inplace=True)
    df.drop(['cik'], 1, inplace=True)
    
    return df

def get_10Q_data(data):
    '''
    Helper function: 
    Takes in the pandas dataframe containing all the sim scores for 10Ks and 10Qs and generates a 
    cleaned up dataframe with just 10Qs of the form: 
    
    10Q Dates | cosine_score | jaccard_score
    
    '''
    df = data.copy()
    df = df[pd.notnull(df['10Qdates'])]
    df.drop(['10Kdates'], 1, inplace=True)
    df.drop(['cik'], 1, inplace=True)
    
    return df

# time to automate these and test before we move to making the python script. 

def plot(ticker, data):
    '''
    Inputs: 
        1. ticker: the name of the ticker which the data (#2) refers to. 
        2. data: The pandas dataframe for the ticker in #1 containing all the sim scores for 10Ks and 10Qs
    
    This function cleans the similarity score data up into two dataframes (Q_data, K_data) of dates and sim scores 
    for 10Ks, 10Qs, plots a single figure containing two subplots plotting similarity scores for the 10Ks and 10Qs, 
    and saves the file as a pdf in the current directory. 
    
    '''
    K_data = get_10K_data(data)
    Q_data = get_10Q_data(data)
    
    K_array = K_data.values
    Q_array = Q_data.values

    with PdfPages(ticker + '.pdf') as pdf:
        fig = plt.figure(figsize = (15, 12))

        axes = fig.add_subplot(2, 1, 1)
        axes.plot(K_array[:, 0], K_array[:, 1], '-r', label = "Cosine Score")
        axes.plot(K_array[:, 0], K_array[:, 2], '-b', label = "Jaccard Score")
        axes.legend()
        axes.set_title(ticker.upper() + ' 10K Similarity Scores')
        axes.set_xlabel('Date of Filing')
        axes.set_ylabel("Similarity Score Values")
        plt.xticks(rotation=70)

        axes = fig.add_subplot(2, 1, 2)
        axes.plot(Q_array[:, 0], Q_array[:, 1], '-r', label = "Cosine Score")
        axes.plot(Q_array[:, 0], Q_array[:, 2], '-b', label = "Jaccard Score")
        axes.legend()
        axes.set_title(ticker.upper() + ' 10Q Similarity Scores')
        axes.set_xlabel('Date of Filing')
        axes.set_ylabel("Similarity Score Values")
        plt.xticks(rotation=70)

        plt.subplots_adjust(hspace = 0.6)
        pdf.savefig()

for i in range(len(ticker_list)):
    cik_dict = TickertoCIK([ticker_list[i]])
    cik = cik_dict[ticker_list[i]]
    
    data = pd.read_csv(cik + '_sim_scores_full.csv')
    
    plot(ticker_list[i], data)


