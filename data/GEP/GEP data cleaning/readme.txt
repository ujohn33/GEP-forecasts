- All the values are in kWh
- The csv files inside each building folder are separated yearly with their corresponding pickled version (cleaned, i.e. contains only datestamp and value) with a single pickle file containing the full profile
- Building 1, 2, 3 and 5 data start from 01/06/2014 02:00, and they are grouped in the gep_consumption.pkl file
- Building 4 measurements only starts from 03/05/2018 00:15
- All measurements ends the 01/07/2020 00:00
- Building 5 consumption not reliable
- Building 5 PV production has a hole in the data between 2017-2018
- the script read_and_clean_csv.py works if it this run in the same directories of the csv files
- yearly_profile_statistics.xlsx contains some statistics to asses the reliability of the data divided by year for each building. The column name are named by the EAN code of each building and the date (you can retrieve the info looking inside each folder) but in any case they are order for building number and growing year: building 1 2014, building 1, 2015, etc.

