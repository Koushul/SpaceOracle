from typing import Optional, Dict
import scanpy as sc
import networkx as nx
import numpy as np
import pandas as pd
import graphical_models as gpm
import networkx as nx 
from collections import defaultdict


class LigandReceptorNetwork:
    def __init__(self, adata=None, gem_tfs_only=False):
        self.adata = adata
        self.flow_network = self.construct_intercellular_flow_network(adata)

        self.gem_contributors = self.get_gem_genes(tfs_only=gem_tfs_only)
        self.lr_dict = self.get_LRs_dict()
        self.gl_dict = self.get_affected_genes(self.gem_contributors)

    def get_gem_genes(self, tfs_only=False, grn_links=None, ntop=10):
        gene_names = np.array(self.adata.var_names)
        gems = [node for node in self.flow_network.nodes if 'GEM' in node]

        if tfs_only:
            tfs = set()
            for k, df in grn_links.items():
                tfs.update(df.source)

        gem_contributors = {}
        for i, gem_label in enumerate(gems):
            contributing = self.adata.uns['nsf_info']['loadings'][:,i]

            if tfs_only:
                gene_mask = np.isin(gene_names, list(tfs))
                contr_tfs = contributing[gene_mask]
                tf_names = gene_names[gene_mask]
                ind = np.argpartition(contr_tfs, -ntop)[-ntop:]
                top_contributors = tf_names[ind]
            else:
                ind = np.argpartition(contributing, -ntop)[-ntop:]
                top_contributors = gene_names[ind]

            gem_contributors[gem_label] = top_contributors

        return gem_contributors

    def get_LRs_dict(self):
        # return {ligand: receptors}

        gems = [node for node in self.flow_network.nodes if 'GEM' in node]

        gem_ligands = {} # these are the ligands of cell_b that affect the GEMs in cell_a
        for gem in gems:
            preds = self.flow_network.predecessors(gem) 
            gem_ligands[gem] = set(preds) - set(gems)

        ligands = set()
        for gem, genes in gem_ligands.items():
            ligands.update([node.replace('inflow-','') for node in genes])

        df = self.adata.uns['flowsig_network']['flow_var_info'].loc[list(ligands)]
        lr_dict = {}
        for ligand, row in df.iterrows():
            receptors = str(row['Interaction']).split('/')
            lr_dict[ligand] = [r.replace(f'{ligand}-','') for r in receptors]
        return lr_dict

    def get_affected_genes(self, gem_genes):
        # return { gene: ligands that affect gene } 
        gems = [node for node in self.flow_network.nodes if 'GEM' in node]
        
        gem_ligands = {} # these are the ligands of cell_b that affect the GEMs in cell_a
        for gem in gems:
            preds = self.flow_network.predecessors(gem) 
            gem_ligands[gem] = set(preds) - set(gems)
        
        gl_dict = defaultdict(list)
        for gem in gems:
            ligands = gem_ligands[gem]
            ligands = [l.replace('inflow-','') for l in ligands]
            if len(ligands) <= 0: continue
            for gene in gem_genes[gem]:
                gl_dict[gene].extend(ligands)
        return gl_dict


    def construct_intercellular_flow_network(
            self, adata: sc.AnnData,
            flowsig_network_key: str = 'flowsig_network',
            adjacency_key: str = 'adjacency_validated_filtered'
        ):

        # from flowsig 
        flow_vars = adata.uns[flowsig_network_key]['flow_var_info'].index.tolist()
        flow_var_info = adata.uns[flowsig_network_key]['flow_var_info']

        flow_adjacency = adata.uns[flowsig_network_key]['network'][adjacency_key]

        nonzero_rows, nonzero_cols = flow_adjacency.nonzero()

        total_edge_weights = {}

        for i in range(len(nonzero_rows)):
            row_ind = nonzero_rows[i]
            col_ind = nonzero_cols[i]

            node_1 = flow_vars[row_ind]
            node_2 = flow_vars[col_ind]

            edge = (node_1, node_2)            

            if ( (edge not in total_edge_weights)&((edge[1], edge[0]) not in total_edge_weights) ):
                total_edge_weights[edge] = flow_adjacency[row_ind, col_ind]
            else:
                if (edge[1], edge[0]) in total_edge_weights:
                    total_edge_weights[(edge[1], edge[0])] += flow_adjacency[row_ind, col_ind]
        
        flow_network = nx.DiGraph()

        # Now let's consturct the graph from the CPDAG
        cpdag =  gpm.PDAG.from_amat(flow_adjacency)

        # Add the directed edges (arcs) first
        for arc in cpdag.arcs:

            node_1 = flow_vars[tuple(arc)[0]]
            node_2 = flow_vars[tuple(arc)[1]]

            # Classify node types
            node_1_type = flow_var_info.loc[node_1]['Type']
            node_2_type = flow_var_info.loc[node_2]['Type']

            # Now we decide whether or not to add the damn edges
            add_edge = False

            # Define the edge because we may need to reverse it
            edge = (node_1, node_2)

            # If there's a link from the expressed morphogen to the received morphogen FOR the same morphogen
            if ( (node_1_type == 'inflow')&(node_2_type == 'module') ):
                add_edge = True

            # If there's a link from received morphogen to a TF
            if ( (node_1_type == 'module')&(node_2_type == 'outflow') ):

                add_edge = True

            if ( (node_1_type == 'module')&(node_2_type == 'module') ):

                add_edge = True

            if add_edge:

                # Get the total edge weight
                total_edge_weight = 0.0

                if edge in total_edge_weights:

                    total_edge_weight = total_edge_weights[edge]

                else:

                    total_edge_weight = total_edge_weights[(edge[1], edge[0])]

                edge_weight = flow_adjacency[tuple(arc)[0], tuple(arc)[1]]

                flow_network.add_edge(*edge)
                flow_network.edges[edge[0], edge[1]]['weight'] = edge_weight / total_edge_weight
                flow_network.nodes[edge[0]]['type'] = node_1_type
                flow_network.nodes[edge[1]]['type'] = node_2_type

        for edge in cpdag.edges:

            node_1 = flow_vars[tuple(edge)[0]]
            node_2 = flow_vars[tuple(edge)[1]]

            # Classify node types
            node_1_type = flow_var_info.loc[node_1]['Type']
            node_2_type = flow_var_info.loc[node_2]['Type']

            # Define the edge because we may need to reverse it
            undirected_edge = (node_1, node_2)

            add_edge = False
            # If there's a link from the expressed morphogen to the received morphogen FOR the same morphogen
            if ( (node_1_type == 'inflow')&(node_2_type == 'module') ):

                add_edge = True

            if ( (node_1_type == 'module')&(node_2_type == 'inflow') ):

                add_edge = True
                undirected_edge = (node_2, node_1)

            if ( (node_1_type == 'module')&(node_2_type == 'outflow') ):

                add_edge = True

            if ( (node_1_type == 'outflow')&(node_2_type == 'module') ):

                add_edge = True
                undirected_edge = (node_2, node_1)

            if ((node_1_type == 'module')&(node_2_type == 'module')):

                add_edge = True

            if add_edge:

                # Get the total edge weight
                total_edge_weight = 0.0

                if undirected_edge in total_edge_weights:

                    total_edge_weight = total_edge_weights[undirected_edge]

                else:

                    total_edge_weight = total_edge_weights[(undirected_edge[1], undirected_edge[0])]

                flow_network.add_edge(*undirected_edge)
                flow_network.edges[undirected_edge[0], undirected_edge[1]]['weight'] = min(total_edge_weight, 1.0)
                flow_network.nodes[undirected_edge[0]]['type'] = node_1_type
                flow_network.nodes[undirected_edge[1]]['type'] = node_2_type

                # Add the other way if we have modules
                if ((node_1_type == 'module')&(node_2_type == 'module')):

                    flow_network.add_edge(undirected_edge[1], undirected_edge[0])
                    flow_network.edges[undirected_edge[1], undirected_edge[0]]['weight'] = min(total_edge_weight, 1.0)
                    flow_network.nodes[undirected_edge[0]]['type'] = node_2_type
                    flow_network.nodes[undirected_edge[1]]['type'] = node_1_type

        return flow_network