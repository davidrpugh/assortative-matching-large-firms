import csv
import urllib2

import pandas as pd
from BeautifulSoup import BeautifulSoup

# base url to which we will append a specific job code
base_url = 'http://www.onetonline.org/link/summary/'

# read in the text file containing the complete listing of occupation codes
data = pd.read_csv('ftp://ftp.bls.gov/pub/time.series/oe/oe.occupation', 
                   delimiter='\t', index_col=False)

# only level 3 codes have suggested related occupations
level_3_codes = data[data.display_level == 3].occupation_code

# write the data to a tab delimited csv file
final_csv_file    = open('related_occupation_codes_20131009.csv','w')
final_data_writer = csv.writer(final_csv_file, dialect='excel', delimiter='\t')

# O*net has arbitrarily split some BLS job codes
tmp_csv_file    = open('onet_occupation_codes_20131009.csv','w')
tmp_data_writer = csv.writer(tmp_csv_file, dialect='excel', delimiter='\t')

for i, code in enumerate(level_3_codes):
    
    # one codes in the master list didn't have web-pages:
    if code in [253098]:
        continue 
        
    # convert int to string
    tmp_code = str(code)[:2] + '-' + str(code)[2:] + '.00'
    
    # extract job title
    tmp_title = data[data.occupation_code == code]['occupation_name'].values[0]
    
    # construct the url for the occupation code
    tmp_url = base_url + tmp_code
    
    # connect to url
    tmp_connection = urllib2.urlopen(tmp_url)
    
    # extract the html code as a soup
    tmp_soup = BeautifulSoup(tmp_connection.read())
    
    # extract the tables from the webpage
    tmp_tabs = tmp_soup('table')
    
    # not all level 3 job codes seem to have a related occupations table
    try:
        tmp_tab = next(t for t in tmp_tabs if t['summary'] == 'Occupation listing')
    
    # o*net has probably split job codes
    except StopIteration:
        
        # collect these job codes and title for future processing
        tmp_onet_occs = tmp_soup.findAll('div', {"class" : "excitem"})
        
        for j in range(len(tmp_onet_occs)):
            tmp_onet_code  = tmp_onet_occs[j].next.split(' ')[0]
            tmp_onet_title = tmp_onet_occs[j].findChild().next
            tmp_data_writer.writerow([tmp_onet_code, tmp_onet_title])
            
    # extract the table rows which contain the codes and titles for related jobs  
    tmp_trs = tmp_tab('tr')  
    
    # loop over the rows to extract the information we want!
    for tr in tmp_trs:
        
        # each row contains a code and a title element
        tmp_tds = tr.findChildren('td')
        
        # convert the job code to a string
        tmp_rel_code = str(tmp_tds[0].next)
        
        # reformat job code by dropping '.00' from the end
        #tmp_rel_code = tmp_rel_code[:-3]
        
        # reformat job code by removing the '-'
        #tmp_rel_code = tmp_rel_code.replace('-', '')
        
        # extract the job title
        tmp_rel_title = tmp_tds[1].findChild('a').next
        
        final_data_writer.writerow([tmp_code, tmp_title, tmp_rel_code, tmp_rel_title])
   
    if i % 100 == 0:
        print 'Done with %i out of %i job codes.' % (i, level_3_codes.size)

tmp_csv_file.close()

# read in the text file containing o*net occupation codes
data = pd.read_csv('onet_occupation_codes_20131009.csv', delimiter='\t', 
                   header=None, index_col=False, )

for i, code in enumerate(data[0]):
            
    # extract job title
    tmp_title = data[1][i]
    
    # construct the url for the occupation code
    tmp_url = base_url + code
    
    # connect to url
    tmp_connection = urllib2.urlopen(tmp_url)
    
    # extract the html code as a soup
    tmp_soup = BeautifulSoup(tmp_connection.read())
    
    # extract the tables from the webpage
    tmp_tabs = tmp_soup('table')
    
    # all o*net codes should have a related occupations table
    try:
        tmp_tab = next(t for t in tmp_tabs if t['summary'] == 'Occupation listing')
    
    except StopIteration:
        print code
        final_data_writer.writerow([code, tmp_title, code, tmp_title])
        continue
        
    # extract the table rows which contain the codes and titles for related jobs  
    tmp_trs = tmp_tab('tr')  
    
    # loop over the rows to extract the information we want!
    for tr in tmp_trs:
        
        # each row contains a code and a title element
        tmp_tds = tr.findChildren('td')
        
        # convert the job code to a string
        tmp_rel_code = str(tmp_tds[0].next)
        
        # reformat job code by dropping '.00' from the end
        #tmp_rel_code = tmp_rel_code[:-3]
        
        # reformat job code by removing the '-'
        #tmp_rel_code = tmp_rel_code.replace('-', '')
        
        # extract the job title
        tmp_rel_title = tmp_tds[1].findChild('a').next
        
        final_data_writer.writerow([code, tmp_title, tmp_rel_code, tmp_rel_title])
   
    if i % 100 == 0:
        print 'Done with %i out of %i o*net job codes.' % (i, data[0].size)

# close the file
final_csv_file.close() 
