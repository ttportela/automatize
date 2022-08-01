# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from pandas import json_normalize
from .main import importer #, display
importer(['S', 'glob', 'np'], globals())

# ------------------------------------------------------------------------------------------------------------
# TRAJECTORY 
# ------------------------------------------------------------------------------------------------------------
class Trajectory:
    def __init__(self, tid, label, attributes, new_points, size):
        self.tid          = tid
        self.label        = label
        self.attributes   = []
        self.size         = size
        
        for attr in attributes:
            if (attr == 'lat_lon' or attr == 'space') and 'lat' not in self.attributes:
                self.attributes.append('lat')
                self.attributes.append('lon')
            else:
                self.attributes.append(attr)
        
        self.points       = []
        if new_points is not None:
            for point in new_points:
                self.add_point(point)
                
    def __repr__(self):
        return '=>'.join([str(x) for x in self.points])
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        if isinstance(other, Movelet):
            return self.__hash__() == other.__hash__()
        else:
            return False
                
    def add_point(self, point):
        assert isinstance(point, dict)
        px = {}
        for k, v in point.items():
            if isinstance(v, dict):
                if k == 'lat_lon' or k == 'space':
                    px['lat'] = v['x']
                    px['lon'] = v['y']
                else:
                    px[k] = v['value']
            else:
                if k == 'lat_lon' or k == 'space':
                    v = v.split(' ')
                    px['lat'] = v[0]
                    px['lon'] = v[1]
                elif k == 'space3d':
                    v = v.split(' ')
                    px['x'] = v[0]
                    px['y'] = v[1]
                    px['z'] = v[2]
                else:
                    px[k] = v
                
        self.points.append(px)
        
#         self.points.append(point)
        
    def toString(self):
        return str(self)
        
    def toText(self):
        return ' >> '.join([("\n".join("{}: {}".format(k, v) for k, v in x.items())) for x in self.points])
    
    def points_trans(self):
        pts_trans = []
        for attr in self.attributes:
            col = {}
            col['attr'] = attr
            for i in range(self.size):
                col['p'+str(i)] = self.points[i][attr]
            pts_trans.append(col)
        return pts_trans
    
def parse_trajectories(df, tid_col='tid', label_col='label'):
    ls_trajs = []
    for tid in df[tid_col].unique():
        df_aux = df[df[tid_col] == tid]
        label = df_aux[label_col].unique()[0]
        features = [x for x in df.columns if x not in [tid_col, label_col]]
        points = df_aux[features].to_dict(orient='records')
        ls_trajs.append(
            Trajectory(tid, label, features, points, len(points))
        )
        
    return ls_trajs


# ------------------------------------------------------------------------------------------------------------
# MOVELETS 
# ------------------------------------------------------------------------------------------------------------
class Movelet:
    def __init__(self, mid, tid, points, quality, label, start, size, children=None):
        self.mid     = mid
        self.quality = quality
        self.tid     = tid
        self.label   = label
        self.start   = start
        self.size    = size
        
        self.data     = []
        if points is not None:
            for point in points:
                self.add_point(point)
                
    def __repr__(self):
        return '=>'.join([str(x) for x in self.data])
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        if isinstance(other, Movelet):
            return self.__hash__() == other.__hash__()
        else:
            return False
                
    def attributes(self):
        return self.data[0].keys()
    
    def add_point(self, point):
        assert isinstance(point, dict)
        px = {}
        for k, v in point.items():
            if isinstance(v, dict):
                if k == 'lat_lon' or k == 'space':
                    px['lat'] = v['x']
                    px['lon'] = v['y']
                else:
                    px[k] = v['value']
            else:
                if k == 'lat_lon' or k == 'space':
                    v = v.split(' ')
                    px['lat'] = v[0]
                    px['lon'] = v[1]
                elif k == 'space3d':
                    v = v.split(' ')
                    px['x'] = v[0]
                    px['y'] = v[1]
                    px['z'] = v[2]
                else:
                    px[k] = v
                
        self.data.append(px)
        
#         if isinstance(list(point.values())[0], dict):
#             px = {}
#             for k, v in point.items():
#                 px[k] = list(v.values())[0]
#             self.data.append(px)
#         else:
#             self.data.append(point)
        
    def toString(self):
        return str(self) + ' ('+'{:3.2f}'.format(self.quality)+'%)'    
    
    def diffToString(self, mov2):
        dd = self.diffPairs(mov2)
        return ' >> '.join([str(x) for x in dd]) + ' ('+'{:3.2f}'.format(self.quality)+'%)' 
        
    def toText(self):
        return ' >> '.join([("\n".join("{}: {}".format(k, v) for k, v in x.items())) for x in self.data]) \
                    + '\n('+'{:3.2f}'.format(self.quality)+'%)'
    
    def commonPairs(self, mov2):
        common_pairs = set()
        
        for dictionary1 in self.data:
            for dictionary2 in mov2.data:
                for key in dictionary1:
                    if (key in dictionary2 and dictionary1[key] == dictionary2[key]):
                        common_pairs.add( (key, dictionary1[key]) )
                        
        return common_pairs
      
    def diffPairs(self, mov2):
        diff_pairs = [dict() for x in range(self.size)]
        
        for x in range(self.size):
            dictionary1 = self.data[x]
            for dictionary2 in mov2.data:
                for key in dictionary1:
                    if (key not in dictionary2):
                        diff_pairs[x][key] = dictionary1[key]
                    elif (key in dictionary2 and dictionary1[key] != dictionary2[key]):
                        diff_pairs[x][key] = dictionary1[key]
                    elif (key in dictionary2 and key in diff_pairs[x] and dictionary1[key] == dictionary2[key]):
                        del diff_pairs[x][key]
                        
        return diff_pairs

def read_movelets(path_name, name='movelets'):
    count = 0
    path_to_file = glob.glob(os.path.join(path_name, '**', 'moveletsOnTrain.json'), recursive=True)

    movelets = []
    for file_name in path_to_file:
        aux_mov = read_movelets_json(file_name, name, count)
        movelets = movelets + aux_mov
        count = len(movelets)

    return movelets
    
def read_movelets_json(file_name, name='movelets', count=0):
    with open(file_name) as f:
        return parse_movelets(f, name, count)
    return []

def parse_movelets(file, name='movelets', count=0):
    importer(['json'], globals())
    
    ls_movelets = []
    data = json.load(file)
    if name not in data.keys():
        name='shapelets'
    l = len(data[name])
    for x in range(0, l):
        points = data[name][x]['points_with_only_the_used_features']
        ls_movelets.append(
            Movelet(\
                count, data[name][x]['trajectory'],\
                points,\
                float(data[name][x]['quality']['quality'] * 100.0),\
                data[name][x]['label'],\
                data[name][x]['start'],\
                int(data[name][x]['quality']['size']))\
        )

        count += 1
    ls_movelets.sort(key=lambda x: x.quality, reverse=True)
    return ls_movelets


# ------------------------------------------------------------------------------------------------------------
# MOVELETS TREE
# ------------------------------------------------------------------------------------------------------------
class MoveletTree:
    def __init__(self, data, children=None):
        self.data      = data
        self.parentSim = 0
        
        self.children = []
        if children is not None:
            for child in self.children:
                self.add_child(child)
                
    def __repr__(self):
        return self.data.toString() + ' - ' + '{:3.2f}'.format(self.parentSim)
                
    def add(self, mov):
        assert isinstance(mov, Movelet)
        node = MoveletTree(mov)
        self.add_child(node)
        return node
                
    def add_child(self, node):
        assert isinstance(node, MoveletTree)
        self.children.append(node)
        
    def findSimilar(self, movelet, similarityFunction):
        sim  = similarityFunction(self.data, movelet)
        node = self
        for child in self.children:
            childSim, childNode = child.findSimilar(movelet, similarityFunction)
            if (childSim > sim):
                node = childNode
                sim  = childSim
        return sim, node
              
    def printNode(self, spacing='', parent=None):
        if parent is None:
            return (spacing + ' ' + self.data.toString()) + '\n'
        else:
            return (spacing + ' ' + self.data.diffPairs(parent.data)) + '\n'
        
    def traversePrint(self, spacing=''):
        s = self.printNode(spacing)
        for child in self.children:
            s += child.traversePrint(spacing + '-')
        return s

# ---------------------------
def similarity(mov1, mov2):
    total_size = mov1.size * len(mov1.data[0]) + mov2.size * len(mov2.data[0])
#     common_pairs = mov1.commonPairs(mov2)
#     return (len(common_pairs)*2 / total_size)
    common_pairs = mov1.commonPairs(mov2)
    common_size  = 0.0
        
    for m in [mov1, mov2]:
        for dictionary in m.data:
            for key in dictionary:
                if (key in dictionary and (key, dictionary[key]) in common_pairs):
                    common_size += 1.0
                    
#     print(common_pairs)
    return common_size / float(total_size)

# ---------------------------
def createTree(movelets):
    movelets.sort(key=lambda x: x.quality, reverse=True)
    
    tree = MoveletTree(movelets.pop(0))
    while len(movelets) > 0:
        mov = movelets.pop(0)
#         print(mov.toString())
        sim, node = tree.findSimilar(mov, similarity)
        node.add(mov).parentSim = sim
    return tree

def movelets_tree(path_name, label=None):
    movelets = []
    count = 0
    path_to_file = glob.glob(os.path.join(path_name, '**', 'moveletsOnTrain.json'), recursive=True)
    df = pd.DataFrame()
    for path in path_to_file:
        with open(path) as f:
            data = json.load(f)  
            
        if label is not None and label != data[name][0]['label']:
            continue
            
        l = len(data[name])
        for x in range(0, l):
            aux_df = []
            
            points = data[name][x]['points_with_only_the_used_features']
            movelets.append(\
                    Movelet(\
                        count, data[name][x]['trajectory'],\
                        points,\
                        float(data[name][x]['quality']['quality'] * 100.0),\
                        data[name][x]['label'],\
                        int(data[name][x]['quality']['size']))\
                    )
            
            count += 1
        
    return movelets

# -----------------------------------------------------------------------
def convert2anytree(tree, parent=None, parentNode=None):
    importer(['G', 'anytree'], globals())
    
#     from anytree import Node, RenderTree
#     from anytree.exporter import DotExporter
    if parent is None:
        root = Node(tree)
    else:
        root = Node(tree, parent=parentNode)
#         root = Node(tree.data.diffToString(parent.data), parent=parentNode)
                
    for child in tree.children:
        convert2anytree(child, tree, root)
        
    return root

def resder_anytree(tree):
    from anytree import RenderTree
    root_node = convert2anytree(tree)
    root_node = RenderTree(root_node)
    return root_node

def convert2digraph(tree, dot=None):
#     from graphviz import Digraph
    importer(['Digraph'], globals())
    
    if dot is None:
        dot = Digraph(comment='Tree')
        dot.node(str(tree.data.mid), tree.data.toText())
    
    for node in tree.children:
        dot.node(str(node.data.mid), node.data.toText() + ' - ' + '{:3.2f}'.format(node.parentSim))
        dot.edge(str(tree.data.mid), str(node.data.mid))
        convert2digraph(node, dot)
        
    return dot
# -----------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------
# VISUALIZATION METHODS
# ------------------------------------------------------------------------------------------------------------

def get_pointfeature(p, attribute=None):
    if attribute is None:
        return '\\n'.join([format_attr(k, v) for k,v in p.items()])
    else:
        return str(p[attribute])
    
def format_attr(key, val, masks={}):
    return str(str(key)+' '+str(val))
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
def movelets_sankey(movelets, attribute=None, title='Movelets Sankey Diagram'):
    importer(['G', 'sankey'], globals())
    # Source: https://gist.github.com/praful-dodda/c98d9fd5dab6e6a9e68bf96ee73630e9
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896',
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
                    '#bcbd22', '#dbdb8d','#17becf', '#9edae5']
    colorNumList = []
    specials = [". ","> ",">> ",".. ","- "]
    
    labels = set()
    labelDict = {}
    lvlDict = {}
    cat_cols = []
    
    def lbladd(x, i):
        x_str = x + ' |'+str(i)#+']'
        if x_str not in labels:
            labels.add(x_str)
            labelDict[x_str] = len(labelDict)
        lvladd(i, labelDict[x_str], x_str)
        return (labelDict[x_str], x_str)

    def lvladd(lvl, idx, x):
        l = 'lv'+str(lvl)
        if l not in cat_cols:
            cat_cols.append(l)
        if l not in lvlDict.keys():
            lvlDict[l] = dict()
        lvlDict[l][idx] = x
        
    sourceTargetDf = pd.DataFrame()
    
    aux_mov = []
    
    for m in movelets:
        if attribute and attribute not in m.attributes():
            continue
        idx, x = lbladd(m.label, 1)
#         lvladd(1, idx, x)
        aux_mov.append(m)
    
#     aux_mov = movelets.copy()
    
    has_lvl = True
    i = 0
    while has_lvl:
        has_lvl = False
        for m in aux_mov:
#             last_idx, last_x = lbladd(m.label, 1)
#             lvladd(1, last_idx, last_x)
#             for i in range(0, len(m.data)):

            if i < len(m.data):
                has_lvl = True
            
                if i == 0:
                    last_idx, last_x = lbladd(m.label, 1)
                else:
                    last_idx, last_x = lbladd(get_pointfeature(m.data[i-1], attribute), i+1)

                idx, x = lbladd(get_pointfeature(m.data[i], attribute), i+2)
#                 lvladd(i+1, idx, x)

                aux_df = {}
                aux_df['source'] = last_x
                aux_df['target'] = x
                aux_df['s'] = 'lv'+str(i+1)
                aux_df['t'] = 'lv'+str(i+2)
                aux_df['label'] = m.label
                aux_df['value'] = 1
    #             aux_df['value'] = m.quality
                sourceTargetDf = sourceTargetDf.append(aux_df, ignore_index=True)
#                 last_idx = idx
#                 last_x = x
            else:
                aux_mov.remove(m)
        i += 1
        
#     return sourceTargetDf
#     sourceTargetDf = sourceTargetDf.sort_values(by=['s', 't'])
    sourceTargetDf = sourceTargetDf.groupby(['source','target','s','t','label']).agg({'value':'sum'}).reset_index()
    sourceTargetDf = sourceTargetDf.sort_values(by=['s', 't', 'label', 'source','target'])
#     return sourceTargetDf
    # List of labels:
    labelList = list((v, k) for k, v in labelDict.items())
    
    # revese the dict 
    rDict = {}
    for k,v in lvlDict.items():
        rDict[k] = {str(v) : k for k,v in v.items()}
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    reset_level = 0
    
    ### Combining codes for placing elements at their corresponding vertical axis.
    unique_list = []
    for k,v in rDict.items():
        v_keys = [x+'_'+str(reset_level) for x in list(v.keys())]
        reset_level += 1
        if v_keys[0][:3] == 'nan':
            v_keys.pop(0)
        [unique_list.append(x) for x in v_keys]
    nodified = nodify_sankey(unique_list)
    
    sourceTargetDf = sourceTargetDf[(sourceTargetDf['source']!='nan') & (sourceTargetDf['target']!='nan')]
    sourceTargetDf['sourceID'] = sourceTargetDf.apply(lambda x: rDict[x['s']][x['source']],axis=1)
    sourceTargetDf['targetID'] = sourceTargetDf.apply(lambda x: rDict[x['t']][x['target']],axis=1)
    
#     return sourceTargetDf
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        arrangement = "snap",
        orientation = 'h',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = [x[1] for x in labelList],
#           color = colorList,
          x=nodified[0],
          y=nodified[1]
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['value'],
          label = sourceTargetDf['label'],
#           color = colorList
        )
      )
    
    layout =  dict(
        title = title,
        height = 772,
        width = 950,
        font = dict(
          size = 12
        )
    )
       
    fig = go.Figure(data = [go.Sankey(data)], layout=layout)
    return fig

def nodify_sankey(node_names):
    ends = sorted(list(set([e[-1] for e in node_names])))
    
    # intervals
    steps = 1.3/len(ends)

    # x-values for each unique name ending
    # for input as node position
    nodes_x = {}
    xVal = 0
    for e in ends:
        nodes_x[str(e)] = xVal
        xVal += steps

    # x and y values in list form
    x_values = [nodes_x[n[-1]] for n in node_names]
    y_values = [.1]*len(x_values)
    
    return x_values, y_values

# -----------------------------------------------------------------------
def movelets_markov(movelets, attribute=None, concat_edges=True):
    importer(['markov'], globals())
    
    from graphviz import Digraph
    G = Digraph(comment='Movelets Markov Tree')
#     G.attr(fontsize='10')
    nodes = set()
    edges = dict()
    for mov in movelets:
        p1 = mov.data[0]
        if attribute == None or attribute in p1.keys():
            with G.subgraph(name='cluster_'+str(mov.mid)) as dot:
                dot.attr(color='blue')
                dot.attr(label='#'+str(mov.mid)+' ({:3.2f}%)'.format(mov.quality))
                if get_pointfeature(p1,attribute) not in nodes:
                    nodes.add(get_pointfeature(p1,attribute))
                    dot.node(get_pointfeature(p1,attribute), get_pointfeature(p1,attribute))#, fontsize='10')
#                     dot.attr(fontsize='10')

                if len(mov.data) > 1:
                    for i in range(1, len(mov.data)):
                        p = mov.data[i]
                        if get_pointfeature(p,attribute) not in nodes:
                            nodes.add(get_pointfeature(p,attribute))
                            dot.node(get_pointfeature(p,attribute), get_pointfeature(p,attribute))#, fontsize='10')
                        # EDGE:
                        ed = get_pointfeature(p1,attribute)+','+get_pointfeature(p,attribute)
                        edlbl = ' #'+str(mov.mid)+'.'+str(i)
                        if concat_edges and ed in edges.keys():
                            edges.update({ed: edges[ed] +', '+ edlbl})
                        else:
                            edges.update({ed: edlbl})
                        
                        p1 = p
        else: 
            continue
        
    for key, value in edges.items():
        key = key.split(',')
        G.edge(key[0], key[1], value, fontsize='10')
    return G

def movelets_markov2(movelets, attribute=None, concat_edges=True):
#     import networkx as nx
#     import pandas as pd
#     import matplotlib as mat

    SEP = '|'
    
    nodes = []
    groups = []
    edges = dict()
    no_colors = dict()
    ed_colors = dict()
    for mov in movelets:
        p1 = mov.data[0]
        if attribute == None or attribute in p1.keys():
            gp = '#'+str(mov.mid)+' ({:3.2f}%)'.format(mov.quality)
            if gp not in groups:
                groups.append(gp)
                
            no1 = get_pointfeature(p1,attribute)
            if no1 not in nodes:
                nodes.append(no1)
                no_colors[no1] = gp
                
            if len(mov.data) > 1:
                for i in range(1, len(mov.data)):
                    p2 = mov.data[i]
                    no2 = get_pointfeature(p2,attribute)
                    if no2 not in nodes:
                        nodes.append(no2)
                        no_colors[no2] = gp
                    # EDGE:
                    ed  = no1+SEP+no2
                    edl = ' #'+str(mov.mid)+'.'+str(i)
                
                    if ed not in ed_colors.keys():
                        ed_colors[ed] = gp
                        
                    if concat_edges and ed in edges.keys():
                        edges.update({ed: edges[ed] + 1})
                    elif concat_edges:
                        edges.update({ed: 1})
                    else:
                        edges.update({ed: edl})
                        
                    p1 = p2

    # Create Graph
    name="Movelets Markov Graph"
    return name, nodes, edges, groups, no_colors, ed_colors

# -----------------------------------------------------------------------        
def graph_nx(name, nodes, edges, groups, no_colors, ed_colors, draw=True):
    importer(['nx', 'mat'], globals())
    SEP = '|'
    G = nx.DiGraph(name="Movelets Markov Graph")
    G.add_nodes_from(nodes)
#     edges_aux = [x.split(SEP) for x in edges.keys()]
    edges_aux = [tuple(k.split(SEP)+[{'weight': v}]) for k,v in edges.items()]
    G.add_edges_from(edges_aux)
    
#     edge_labels = {tuple(list(k.split(SEP))): v for k,v in edges.items()}

    cmap = mat.colors.ListedColormap(['C0', 'darkorange'])
    
    # Draw graph
    if draw:
        # Specify layout and colors
        ccod = pd.Categorical(groups).codes
        ecod = pd.Categorical([ccod[groups.index(ed_colors[x])] for x in edges.keys()]).codes
        ncod = pd.Categorical([ccod[groups.index(no_colors[x])] for x in nodes]).codes

        paux = max(edges.values())
        edge_sizes = [(x/paux)+1 for x in edges.values()]
        
        return nx.draw(G, with_labels=True, node_size=10000, node_color=ncod, cmap=cmap, width=edge_sizes, edge_color=ecod)
    else:
        
        # 2 ) get node pos
        pos = nx.circular_layout(G)
        # 3.) get cytoscape layout
        cy = nx.readwrite.json_graph.cytoscape_data(G)
#         return cy
        # 4.) Add the dictionary key label to the nodes list of cy
        for n in cy['elements']['nodes']:
            for k,v in n.items():
                v['label'] = v.pop('value')
        # 5.) Add the pos you got from (2) as a value for data in the nodes portion of cy
        scale = 150
        
        ecod = [mat.colors.rgb2hex(cmap(groups.index(ed_colors[x]))) for x in edges.keys()]
        ncod = [mat.colors.rgb2hex(cmap(groups.index(no_colors[x]))) for x in nodes]
        
        for n,p in zip(cy['elements']['nodes'],pos.values()):
#             n['pos'] = {'x':p[0]*scale,'y':p[1]*scale}
#             n['position'] = {'x':p[0]*scale,'y':p[1]*(scale)}
#             n['classes']  = 'red'
            n['data']['color'] = str(ncod[nodes.index(n['data']['id'])])
#             n['style']  = {
#                 'background-color': ncod[nodes.index(n['data']['id'])],
#                 'line-color': ncod[nodes.index(n['data']['id'])]
#             }
        
        for n in cy['elements']['edges']:
            n['data']['color'] = str(ecod[list(edges.keys()).index(n['data']['source']+SEP+n['data']['target'])])
#             n['style']  = {
#                 'background-color': ecod[list(edges.keys()).index(n['data']['source']+SEP+n['data']['target'])],
#                 'line-color': ecod[list(edges.keys()).index(n['data']['source']+SEP+n['data']['target'])],
#                 'target-arrow-color': ecod[list(edges.keys()).index(n['data']['source']+SEP+n['data']['target'])],
#                 'label': 'data(weight)',
#             }

#         return cy
        # 6.) Take the results of (3)-(5) and write them to a list, like elements_ls
        elements_ls = cy['elements']['nodes'] + cy['elements']['edges']

        return elements_ls

# -----------------------------------------------------------------------
