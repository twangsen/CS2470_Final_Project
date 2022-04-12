import xport, csv

with open('DEMO_2013.XPT', 'rb') as f:
    i = 0
    for row in xport.Reader(f):
        if i< 10:
            print(row)
        i+=1
    print(i)

