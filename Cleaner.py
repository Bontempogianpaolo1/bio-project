import pandas as pd


def reducefileintofile(file, file2):
    df = pd.read_csv(file)
    df = df.tail(100)
    df.drop(['Unnamed: 0'], inplace=True, axis=1)
    print("old dimension: "+ str(df.shape[0])+"x"+str(df.shape[1]))
    # remove not significant columns
    hl = []
    l = list(df.columns.values)
    for l1 in l:
        if l1 == "l":
            hl.append(l1)
        elif sum(df[l1]) != 0:
            hl.append(l1)
    df2 = df[hl]
    # remove not significant rows (like normal-like)
    df3 = df2.fillna("")
    df4 = df3[df3.l != "normal-like"]
    df5= df4[df4.l != ""]
    print("new dimension: " + str(df5.shape[0]) + "x" + str(df5.shape[1]))
    df5.to_csv(file2)


def clean(df):

    print("old dimension: " + str(df.shape[0])+"x"+str(df.shape[1]))
    # remove not significant columns
    df.columns = df.columns.str.replace(' ', '')
    # remove not significant rows (like normal-like)
    df3 = df.fillna("")
    df5 = df3[df3.l != ""]
    catenc = pd.factorize(df5['l'])
    df5['labels'] = catenc[0]
    print("new dimension: " + str(df5.shape[0]) + "x" + str(df5.shape[1]))
    return df5
