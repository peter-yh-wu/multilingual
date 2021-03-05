"""
Script to evaluate embeddings
"""

import matplotlib
matplotlib.use('Agg')
import anytree
import argparse
import cartopy.crs as ccrs
import json
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from anytree import RenderTree
from anytree.importer import DictImporter
from functools import partial
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='one-time computations')
    parser.add_argument('-p', '--path', default='embs.npy', type=str, help='path to language embeddings')
    parser.add_argument('-r', '--radius', default=500, type=int, help='radius')
    args = parser.parse_args()

    embs = np.load(args.path)
    if not os.path.exists('../outputs'):
        os.makedirs('../outputs')

    # for each k, we return fraction of languages that have a top k embedding distance
    # neighbor that's in the family/tree
    ks = {2,4,8,16,32}

    metrics = {}  # where all the metrics are added
    
    lang_list_pth = '../metadata/LangList.txt'
    with open(lang_list_pth, 'r') as inf:
        lines = inf.readlines()
    langlist_lines = [l.strip() for l in lines]

    lang_to_link = {l.split()[0]:l.split()[2] for l in langlist_lines}
        # lang_to_link['ABIWBT'] = 'https://en.wikipedia.org/wiki/Abidji_language'
    lang_to_3 = {l.split()[0]:l.split()[1] for l in langlist_lines}
        # lang_to_3['ABIWBT'] = 'abi'
    lang_to_coord = {l.split()[0]:(l.split()[4], l.split()[5]) for l in langlist_lines} # latitude, longitude
        # lang_to_coord['ABIWBT'] = ('5.65656', '-4.58421')

    with open('../metadata/lang195.txt', 'r') as inf:
        lines = inf.readlines()
    lang195_langs = [l.strip() for l in lines]  # ['ABIWBT', ...]
    lang195_langs_set = set(lang195_langs)

    lons = [float(lang_to_coord[l][1]) for l in lang195_langs]
    lats = [float(lang_to_coord[l][0]) for l in lang195_langs]

    if not os.path.exists('../metadata/dists_km.npy'):
        coords = [(lat, lon) for lat, lon in zip(lats, lons)]
        dists = [[geopy.distance.distance(c1, c2) for c1 in coords] for c2 in coords]
        dists_km = [[d.km for d in ds] for ds in dists]
            # dists_km[i][j] = dist btw lang_i and lang_j
        np.save('../metadata/dists_km.npy', dists_km)
    dists_km = np.load('../metadata/dists_km.npy')

    emb_dists = [[np.linalg.norm(a-b) for a in embs] for b in embs]

    correlation_mean = np.mean([pearsonr(dists_km[i], emb_dists[i])[0] for i in range(len(dists_km))])
    correlation_std = np.std([pearsonr(dists_km[i], emb_dists[i])[0] for i in range(len(dists_km))])
    metrics['correlation_mean'] = correlation_mean
    metrics['correlation_std'] = correlation_std

    idxs_under_threshold = [[i for i, d in enumerate(ds) if d < args.radius] for ds in dists_km]
        # row i has the languages < 500 km of lang i
    dists_km_subsets = [[ds[i] for i in idxs] for idxs, ds in zip(idxs_under_threshold, dists_km)]
    emb_dists_subsets = [[ds[i] for i in idxs] for idxs, ds in zip(idxs_under_threshold, emb_dists)]
    rad_correlation_mean = np.mean([pearsonr(dists_km_subsets[i], emb_dists_subsets[i])[0] for i in range(len(dists_km)) if len(dists_km_subsets[i]) > 1])
    rad_correlation_std = np.std([pearsonr(dists_km_subsets[i], emb_dists_subsets[i])[0] for i in range(len(dists_km)) if len(dists_km_subsets[i]) > 1])
    metrics['rad_correlation_mean'] = rad_correlation_mean
    metrics['rad_correlation_std'] = rad_correlation_std

    ethnologue_tree_path = '../outputs/ethnologue_tree.txt'
    ethnologue_ancestor_mat_path = '../metadata/ethnologue_lang195_ancestor_mat.npy'
    ethnologue_avg_ancestor_mat_path = '../metadata/ethnologue_lang195_avg_ancestor_mat.npy'
    ethnologue_tree_idxs_path = '../metadata/ethnologue_lang195_tree_idxs.npy'

    if not (os.path.exists(ethnologue_tree_path) and 
            os.path.exists(ethnologue_ancestor_mat_path) and
            os.path.exists(ethnologue_avg_ancestor_mat_path) and
            os.path.exists(ethnologue_tree_idxs_path)):
        def contains_pattern(p, node):
            return p in node.name

        filename = "../metadata/ethnologue_forest.json"
        with open(filename, 'r') as f:
            ethnologue_forest = json.load(f)

        importer = DictImporter()
        ethnologue_trees = []
        for cur in ethnologue_forest:
            root = importer.import_(cur)
            ethnologue_trees.append(root)

        output_str = ''
        for root in ethnologue_trees:
            for pre, _, node in RenderTree(root):
                output_str += "%s%s\n" % (pre, node.name)
        with open(ethnologue_tree_path, 'w+') as ouf:
            ouf.write(output_str)

        def get_node_ethnologue(lang):
                l1 = lang_to_3[lang]
                f = partial(contains_pattern, l1)
                for ti, t in enumerate(ethnologue_trees):
                    nodes = anytree.search.findall(t, filter_=f)
                    if len(nodes) > 0:
                        return nodes[0], ti
                return None, -1

        def tree_dist(node0, node1, mode='min'):
            '''assumes nodes are in the same tree

            e.g. if node0 is depth 7 and node1 is depth 5 and their common ancestor is depth 3,
                then we return 5-3 = 2
            '''
            ancestor_depth = len(anytree.util.commonancestors(node0, node1))
            if mode == 'avg':
                return (node0.depth+node1.depth)/2-ancestor_depth
            else:
                return min(node0.depth, node1.depth)-ancestor_depth

        def mk_ancestor_mat(languages, mode='min'):
            '''
            Args:
                languages: list of strings (each string is 6-letter language id)

            Returns:
                ancestor_mat: entry [i][j] is the tree_dist between lang_i and lang_j
            '''
            nodes = []
            tree_idxs = []
            for lang in languages:
                n, ti = get_node_ethnologue(lang)
                nodes.append(n)
                tree_idxs.append(ti)
            ancestor_mat = np.zeros((len(languages), len(languages)), dtype=int)
            for i, lang1 in enumerate(languages):
                for j, lang2 in enumerate(languages):
                    if lang1 == lang2:
                        ancestor_mat[i][j] = 0
                    elif i < j:
                        node0 = nodes[i]
                        node1 = nodes[j]
                        if tree_idxs[i] != tree_idxs[j]:
                            ancestor_mat[i][j] = -1
                        else:
                            ancestor_mat[i][j] = tree_dist(node0, node1, mode=mode)
                    else:
                        ancestor_mat[i][j] = ancestor_mat[j][i]
            return ancestor_mat

        if not os.path.exists(ethnologue_ancestor_mat_path):
            lang195_ancestor_mat = mk_ancestor_mat(lang195_langs)
            np.save(ethnologue_ancestor_mat_path, lang195_ancestor_mat)
        if not os.path.exists(ethnologue_avg_ancestor_mat_path):
            lang195_ancestor_mat = mk_ancestor_mat(lang195_langs, mode='avg')
            np.save(ethnologue_avg_ancestor_mat_path, lang195_ancestor_mat)
        if not os.path.exists(ethnologue_tree_idxs_path):
            lang195_tree_idxs = []
            for lang in lang195_langs:
                n, ti = get_node_ethnologue(lang)
                lang195_tree_idxs.append(ti)
            np.save(ethnologue_tree_idxs_path, lang195_tree_idxs)

    ethnologue_lang195_ancestor_mat = np.load(ethnologue_ancestor_mat_path)
    ethnologue_lang195_avg_ancestor_mat = np.load(ethnologue_avg_ancestor_mat_path)

    lang195_tree_idxs = np.load(ethnologue_tree_idxs_path)

    emb_dists_np = np.array(emb_dists)

    # for each k, we return fraction of languages that have a top k embedding distance
    # neighbor that's in the family/tree

    langs_with_fam = []  # list of (index in lang195, lang)
    for i, lang in enumerate(lang195_langs):
        if np.sum(ethnologue_lang195_ancestor_mat[i] >= 0) > 1:
            langs_with_fam.append((i, lang))
    tot = len(langs_with_fam)

    all_closest_idxs = {i:np.argsort(emb_dists_np[i])[1:] for i, lang in langs_with_fam}

    max_k = max(ks)
    num_outliers = 10
    num_non_outliers = tot-num_outliers

    for ancestor_mat in [ethnologue_lang195_ancestor_mat]:
        k_outlier = -1
        outliers = []  # list of 6-letter language ids
        for k in range(1, tot):
            count = 0
            curr_outliers = []
            for i, lang in langs_with_fam:
                closest_idxs = all_closest_idxs[i][:k]
                ti = lang195_tree_idxs[i]
                ctis = lang195_tree_idxs[closest_idxs]
                has_fam = (ctis == ti).sum() > 0
                if not has_fam:
                    curr_outliers.append(lang)
                count += has_fam
            if count >= num_non_outliers and k > max_k:
                k_outlier = k
                outliers = curr_outliers
                break
            if k in ks:
                metrics['k-%d' % k] = count/tot
    metrics['k_outlier'] = k_outlier
    metrics['outliers'] = outliers

    ti_to_195_idxs = {}
    ti_to_195_idxs_set = {}
    for i, ti in enumerate(lang195_tree_idxs):
        if ti in ti_to_195_idxs:
            ti_to_195_idxs[ti].append(i)
        else:
            ti_to_195_idxs[ti] = [i]
    for ti in ti_to_195_idxs:
        ti_to_195_idxs_set[ti] = set(ti_to_195_idxs[ti])

    # languages on map colored by language family
    if args.verbose:
        import matplotlib
        k = 10
        tis = lang195_tree_idxs
        ti_list = []
        new_tis = []
        for ti in tis:
            if ti in ti_list:
                new_tis.append(ti_list.index(ti))
            else:
                ti_list.append(ti)
                new_tis.append(len(ti_list)-1)
        tis = new_tis
        plt.figure(figsize=(20,20))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        plt.scatter(lons, lats, marker='o', c=tis, cmap=plt.cm.jet)
        plt.title('Language Families')
        plt.colorbar(fraction=0.02, pad=0.01)
        plt.savefig('../outputs/lang_fams.png')

    kmeans = KMeans(n_clusters=5, random_state=0).fit(embs)
    cd = ['b', 'c', 'y', 'm', 'r']  # color scheme, but can also use colormap
    plt.figure(figsize=(20,20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    plt.scatter(lons, lats,
            c=[cd[i] for i in kmeans.labels_], marker='o'
            )
    plt.savefig('../outputs/cluster_map.png')

    # tsne plot: same language family has same color
    tis = lang195_tree_idxs
    min_fam_size = 5
    min_fam_ti = -1
    for ti in tis:
        if len(ti_to_195_idxs_set[ti]) < min_fam_size:
            min_fam_ti = ti
            break
    idxs_kept = []
    for i, ti in enumerate(tis):
        if len(ti_to_195_idxs_set[ti]) < min_fam_size:
            tis[i] = min_fam_ti
        else:
            idxs_kept.append(i)
    ti_to_color_id = {min_fam_ti: 0}
    count = 1
    for ti in tis:
        if ti not in ti_to_color_id:
            ti_to_color_id[ti] = count
            count += 1
    tis = np.array([ti_to_color_id[ti] for ti in tis])[idxs_kept]
    perplexity = 15.0
    learning_rate = 3e2
    n_iter = 600
    n_iter_without_progress = 300
    random_state = 0
    X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress, random_state=random_state).fit_transform(embs)
    plt.figure(figsize=(10,10))
    plt.scatter(X_embedded[idxs_kept,0], X_embedded[idxs_kept,1], c=tis, cmap=plt.cm.jet)
    plt.savefig('../outputs/tnse.png')

    with open('../outputs/metrics.json', 'w+') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
