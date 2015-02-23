import csv
import numpy as np

'''
This code imports the data from 'PD1.csv'

Output
------
Following np.ndarrays: 

firmID (str), size (int), wage (float), profit (float), skill_w (float), 
firm_age (float), industry_code (int), region (int)

'''

# Opening data
with open('PD1.csv', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)

# Passing data to lists, then to arrays (should change this to make it all in one) 
firmID = []
size = []
wage = []
profit = []
skill_w = []
firm_age = []
industry_code = []
region = []
for row in data[1:]:
    firmID.append(row[0])
    size.append(int(row[1]))
    wage.append(float(row[2]))
    profit.append(float(row[3]))
    skill_w.append(float(row[4]))
    if row[5]== '':
        firm_age.append(0)
    else:
        firm_age.append(float(row[5]))
    if row[6]== '':
        industry_code.append(0)
    else:
        industry_code.append(int(row[6]))
    region.append(int(row[7]))
# Firm unique code (string)
firmID = np.asarray(firmID)
# Firm size in workers (int)
size = np.asarray(size)
# Daily average wage for each firm, in euros (float)
wage = np.asarray(wage)
# Declared average profits for each firm per year, in euros (float)
profit = np.asarray(profit)
# Average education level of workers per firm, from 0 to 6 (float)
skill_w = np.asarray(skill_w)
# Firm age in years (float) Missing observations have a 0.
firm_age = np.asarray(firm_age)
# Industry codes (int) Missing observations have a 0.
industry_code = np.asarray(industry_code)
# Regional code (int)
region = np.asarray(region)