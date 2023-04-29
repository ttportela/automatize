def update_report(df, names, *params):
    for index, (att, val) in enumerate(zip(names.split(', '), params)):
        df[att] = [val]
    return df

def print_params(names, *params):
    return '_'.join([str(x)+'_'+str(y) for i, (x,y) in enumerate(zip(names.split(', '), params))])

def concat_params(*params):
    return '-'.join([str(y) for y in params])