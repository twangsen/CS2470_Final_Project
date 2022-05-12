import pandas as pd
import glob

# read the csv files in different folders
def csv_read(path):
    csv_list = []
    csv_name = []
    for fname in glob.glob(path):
        df = pd.read_csv(fname)
        csv_list.append(df)
        csv_name.append(fname)
        # print(fname)
    
    return csv_list, csv_name

path1 = r"/Users/mac/Desktop/CS2470_Final_Project/data/csv_file/*.csv"
path2 = r"/Users/mac/Desktop/CS2470_Final_Project/data/csv_file/2011Questionnaire/*.csv"
path3 = r"/Users/mac/Desktop/CS2470_Final_Project/data/csv_file/2013Questionnaire/*.csv"
path4 = r"/Users/mac/Desktop/CS2470_Final_Project/data/csv_file/2015Questionnaire/*.csv"
path5 = r"/Users/mac/Desktop/CS2470_Final_Project/data/csv_file/2017Questionnaire/*.csv"

csv_comb, csv_name = csv_read(path1)
csv_2011, _ = csv_read(path2)
csv_2013, _ = csv_read(path3)
csv_2015, _ = csv_read(path4)
csv_2017, _ = csv_read(path5)

# combine the data by year
for i in range(len(csv_name)):
    if "2011" in csv_name[i]:
        csv_2011.append(csv_comb[i])
    elif "2013" in csv_name[i]:
        csv_2013.append(csv_comb[i])
    elif "2015" in csv_name[i]:
        csv_2015.append(csv_comb[i])
    else:
        csv_2017.append(csv_comb[i])


def comb_data(csv_list):
    df = csv_list[0]
    for i in range(1, len(csv_list)):
        df = df.merge(csv_list[i], how='left', on='SEQN')
    
    return df

df_2011 = comb_data(csv_2011)
df_2013 = comb_data(csv_2013)
df_2015 = comb_data(csv_2015)
df_2017 = comb_data(csv_2017)

# print(df_2011.columns, df_2013.columns, df_2015.columns, df_2017.columns)

# Combine horizontally
df = pd.concat([df_2011, df_2013, df_2015, df_2017], ignore_index = True, sort = False)
df.to_csv('/Users/mac/Desktop/CS2470_Final_Project/data/csv_file/combined_data.csv')
