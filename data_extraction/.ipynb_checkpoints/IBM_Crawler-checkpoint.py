import sys
sys.path.append('Users/u6676643/opt/anaconda3/lib/python3.8/site-packages/selenium')

from selenium import webdriver
import time
import tqdm

browser = webdriver.Chrome()
browser.get("https://pubchem.ncbi.nlm.nih.gov")

time.sleep(1)

mol_list = [HCl,HBr,HI,HClO4,TfOH]
SMILES_list = []

for i,mol in enumerate(tqdm.tqdm(mol_list)):
    print(i)
    
    #search query
    browser.get("https://pubchem.ncbi.nlm.nih.gov/#query="+str(mol))
    
    #press first entry
    first_result_button = browser.find_element_by_xpath("/html/body/div[1]/div/div/main/div[2]/div[2]/div[2]/div/div/div/div[2]/ul/li[1]/div/div/div[1]/div[2]/div[1]/a")
    first_result_button.click()
    
    #find SMILES string
    SMILES_string = browser.find_element_by_xpath("/html/body/div[1]/div/main/div[2]/div/div/div[5]/section[2]/section[1]/section[4]/div[2]/div[1]/p").text
    SMILES_list.append(SMILES_string)
    
    print(SMILES_list)