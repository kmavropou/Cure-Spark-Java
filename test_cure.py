from sklearn.cluster import AgglomerativeClustering
import pandas as pd

def distclust(file, c):
    filefinal=open(file,"r")
    listofpoints= []
    for line in filefinal.readlines():
        line=line[1:-2]
        spliting=line.split(",")
        splitfloat=[]
        for s in spliting:
            splitfloat.append(float(s))

        listofpoints.append(splitfloat)

    x=[]
    y=[]
    for point in listofpoints:
        x.append(point[0])
        y.append(point[1])

    model = AgglomerativeClustering(n_clusters=c,affinity="euclidean",linkage ="single")
    predictions = model.fit_predict(listofpoints)
    data={'x':x,'y':y,'Predictions':predictions}

    results=pd.DataFrame(data)
    print(results.head(10))

    results.to_csv('predictions.csv')

distclust("resultsjava",5)
