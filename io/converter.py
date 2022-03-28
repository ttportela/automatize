from automatize.main import importer #, display
importer(['S'], globals())


def convert_zip2csv(folder, file, cols=None, class_col = 'label', tid_col='tid', missing='?'):
#     from ..main import importer
    importer(['S', 'zip'], globals())
    
#     data = pd.DataFrame()
    print("Converting "+file+" data from... " + folder)
    if '.zip' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.zip')
        
#     with ZipFile(url) as z:
#         files = z.namelist()
#         files.sort()
#         for filename in files:
# #             data = filename.readlines()
# #             print(filename)
#             if cols is not None:
#                 df = pd.read_csv(z.open(filename), names=cols, na_values='?')
#             else:
#                 df = pd.read_csv(z.open(filename), header=None, na_values='?')
#             df['tid']   = filename.split(" ")[1][1:]
#             df[class_col] = filename.split(" ")[2][1:-3]
#             data = pd.concat([data,df])
    print("Done.")
    data = read_zip(ZipFile(url), cols, class_col, tid_col, missing)
    return data
    
def zip2csv(folder, file, cols, class_col = 'label', tid_col='tid', missing='?'):
#     from ..main import importer
#     importer(['S'], locals())
    
#     data = pd.DataFrame()
#     print("Converting "+file+" data from... " + folder)
#     if '.zip' in file:
#         url = os.path.join(folder, file)
#     else:
#         url = os.path.join(folder, file+'.zip')
#     with ZipFile(url) as z:
#         for filename in z.namelist():
# #             data = filename.readlines()
#             df = pd.read_csv(z.open(filename), names=cols)
# #             print(filename)
#             df['tid']   = filename.split(" ")[1][1:]
#             df[class_col] = filename.split(" ")[2][1:-3]
#             data = pd.concat([data,df])
#     print("Done.")
    data = convert_zip2csv(folder, file, cols, class_col, tid_col, missing)
    print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
    data.to_csv(os.path.join(folder, file+'.csv'), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

# def convertToCSV(path): 
# #     from ..main import importer
# #     importer(['S'], locals())
    
#     dir_path = os.path.dirname(os.path.realpath(path))
#     files = [x for x in os.listdir(dir_path) if x.endswith('.csv')]

#     for file in files:
#         try:
#             df = pd.read_csv(file, sep=';', header=None)
#             print(df)
#             df.drop(0, inplace=True)
#             print(df)
#             df.to_csv(os.path.join(folder, file), index=False, header=None)
#         except:
#             pass

def zip2arf(folder, file, cols, tid_col='tid', class_col = 'label', missing='?'):
    data = pd.DataFrame()
    print("Converting "+file+" data from... " + folder)
    if '.zip' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.zip')
    with ZipFile(url) as z:
        for filename in z.namelist():
#             data = filename.readlines()
            df = pd.read_csv(z.open(filename), names=cols, na_values=missing)
#             print(filename)
            df[tid_col]   = filename.split(" ")[1][1:]
            df[class_col] = filename.split(" ")[2][1:-3]
            data = pd.concat([data,df])
    print("Done.")
    
    print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
    data.to_csv(os.path.join(folder, file+'.csv'), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def convert2ts(data_path, folder, file, cols=None, tid_col='tid', class_col = 'label'):
    print("Converting "+file+" data from... " + data_path + " - " + folder)
    data = readDataset(data_path, folder, file, class_col)
    
    file = file.replace('specific_',  '')
    
    tsName = os.path.join(data_path, folder, folder+'_'+file.upper()+'.ts')
    tsDesc = os.path.join(data_path, folder, folder+'.md')
    print("Saving dataset as: " + tsName)
    if cols == None:
        cols = [x for x in data.columns if x not in [tid_col, class_col]]
    
    f = open(tsName, "w")
    
    if os.path.exists(tsDesc):
        fd = open(tsDesc, "r")
        for line in fd:
            f.write("# " + line)
#         fd.close()

    f.write("#\n")
    f.write("@problemName " + folder + '\n')
    f.write("@timeStamps false")
    f.write("@missing "+ str('?' in data)+'\n')
    f.write("@univariate "+ ('false' if len(cols) > 1 else 'true') +'\n')
    f.write("@dimensions " + str(len(cols)) + '\n')
    f.write("@equalLength false" + '\n')
    f.write("@seriesLength " + str(len(data[data[tid_col] == data[tid_col][0]])) + '\n')
    f.write("@classLabel true " + ' '.join([str(x).replace(' ', '_') for x in list(data[class_col].unique())]) + '\n')
    f.write("@data\n")
    
    for tid in data[tid_col].unique():
        df = data[data[tid_col] == tid]
        line = ''
        for col in cols:
            line += ','.join(map(str, list(df[col]))) + ':'
        f.write(line + str(df[class_col].unique()[0]) + '\n')
        
    f.write('\n')
    f.close()
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def xes2csv(folder, file, cols=None, tid_col='tid', class_col = 'label', show_progress=True, save=False):
    def getTrace(log, tid):
        t = dict(log[tid].attributes)
    #     t.update(log[tid].attributes)
        return t
    
    def getEvent(log, tid , j, attrs):
        ev = dict(log[tid][j])
        ev.update(attrs)
        ev['tid'] = tid+1
        return ev
    
    
    import pm4py
    if '.xes' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.xes')
    
    print("Reading "+file+" data from: " + folder)
    log = pm4py.read_xes(url)
    
    if show_progress:
        import tqdm
        data = list(map(lambda tid: 
                    pd.DataFrame(list(map(lambda j: getEvent(log, tid , j, getTrace(log, tid)), range(len(log[tid]))))),
                    tqdm.notebook.trange(len(log), desc='Converting')))
    else:
        data = list(map(lambda tid: 
                    pd.DataFrame(list(map(lambda j: getEvent(log, tid , j, getTrace(log, tid)), range(len(log[tid]))))),
                    range(len(log))))

    df = pd.concat(data, ignore_index=True)
    
    if save:
        print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
        df.to_csv(os.path.join(folder, file+'.csv'), index = False)

    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return df