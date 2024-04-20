import numpy as np
import torch 
import graphviz
import os
import torch.nn.functional as F
from collections import defaultdict
import pdb

def split_value(M, n):
    quotient, remainder = divmod(M, n)
    parts = [quotient] * n
    for i in range(remainder):
        parts[i] += 1
    return parts

class Node:

    def __init__(self, name, parent = None, label = None):

        self.parent = parent
        self.children = []
        self.children_to_labels = {}
        self.name = name
        self.label = label
        self.weights = None
        self.num_protos_per_child = None #{}
        self.discretized_tree_label = None

    def set_loss_weightage(self, class_size_count):
        self.num_images_of_each_child = []
        for child in self.children:
            num_images_of_child = 0
            for leaf_descendent_name in self.leaf_descendents_of_child[child.name]:
                num_images_of_child += class_size_count[leaf_descendent_name]
            self.num_images_of_each_child.append(num_images_of_child)
        self.weights = min(self.num_images_of_each_child) / torch.tensor(self.num_images_of_each_child, requires_grad=False)

    def set_loss_weightage_using_descendants_count(self):
        self.num_descendants_of_each_child = []
        for child in self.children:
            self.num_descendants_of_each_child.append(len(self.leaf_descendents_of_child[child.name]))
        self.weights = min(self.num_descendants_of_each_child) / torch.tensor(self.num_descendants_of_each_child, requires_grad=False)

    def set_num_protos(self, num_protos_per_descendant, num_protos_per_child, min_protos=0, split_protos=False):

        if num_protos_per_child > 0:
            # self.num_protos_per_child = {}
            # for i, child in enumerate(self.children):
            #     self.num_protos_per_child[child.name] = num_protos_per_child
            # self.num_protos = len(self.children) * num_protos_per_child
            self.num_protos_per_child = {}
            self.num_protos = 0
            for i, child in enumerate(self.children):
                self.num_protos_per_child[child.name] = max(num_protos_per_child, num_protos_per_descendant * child.num_leaf_descendents())
                self.num_protos += max(num_protos_per_child, num_protos_per_descendant * child.num_leaf_descendents())
            return

        self.num_protos = max(min_protos, self.num_leaf_descendents() * num_protos_per_descendant)

        if split_protos:
            self.num_protos_per_child = {}

        if split_protos and (min_protos > (self.num_leaf_descendents() * num_protos_per_descendant)):
            parts = split_value(min_protos, self.num_children())
            for i, child in enumerate(self.children):
                self.num_protos_per_child[child.name] = parts[i]
        elif split_protos and (min_protos < (self.num_leaf_descendents() * num_protos_per_descendant)):
            for i, child in enumerate(self.children):
                self.num_protos_per_child[child.name] = len(self.leaf_descendents_of_child[child.name]) * num_protos_per_descendant

        if not split_protos:
            raise NotImplementedError()

    def add_children(self, names, labels = None):
        if type(names) is not list:
            names = [names]
        if labels is None:
            labels = [i for i in range(len(self.children),len(self.children)+len(names))]
        names.sort()
        for i in range(len(names)):
            self.children.append( Node(names[i], parent=self, label = labels[i]))    
            self.children_to_labels.update({names[i] : labels[i]})

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child    

    def children_names(self):
        return([child.name for child in self.children])
    
    def is_leaf(self):
        if self.num_children() > 0:
            return False
        return True

    # def assign_children_names(self):
    #     self.children_names = self.children_names()

    # def assign_all_children_names(self):        
    #     active_nodes = []
    #     active_nodes += [self]
    #     while len(active_nodes) > 0:
    #         for node in active_nodes:
    #             node.assign_children_names()
    #             nodel.children_names.sort()
    #         new_active_nodes = [] 
    #         for node in active_nodes:
    #             new_active_nodes += node.children
    #         active_nodes = new_active_nodes           

    def get_node(self,name):                
        active_nodes = [self]
        while True:
            for node in active_nodes:
                if node.name == name:
                    return node
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes
            if len(active_nodes) == 0:
                print("node for " + name + " not found")
                break

    def get_node_attr(self,name,attr):
        node = self.get_node(name)
        return getattr(node,attr)

    def set_node_attr(self,name,attr,value):
        node = self.get_node(name)
        return setattr(node,attr,value)        
                
    def num_children(self):
        return(len(self.children))

    def class_to_num_children(self):
        class_to_num = {}
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                class_to_num.update({node.name : node.num_children()})
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return class_to_num

    def class_to_proto_shape(self, x_per_child = 1, dimension = 512):
        class_to_shape = {}
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:
                    class_to_shape.update({node.name : (x_per_child * node.num_children(),dimension,1,1)})
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return class_to_shape        

    def classes_with_children(self):
        classes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:
                    classes.append(node.name)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return classes

    def nodes_with_children(self):        
        nodes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:# and node.name != "root":
                    nodes.append(node)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return nodes        

    def add_children_to(self,name,children):
        node = self.get_node(name)
        node.add_children(children)

    def parents(self):
        parents = []
        parent = node.parent
        parents += parent
        while parent.parent is not None:
            parent = parent.parent
            parents += parent
        return parents


    def assign_descendents(self):
        active_nodes = []
        active_nodes += self.children
        descendents = set()
        while len(active_nodes) > 0:
            for node in active_nodes:
                descendents.add(node.name)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        self.descendents = descendents

    def assign_leaf_descendents(self):

        if self.is_leaf():
            # if child is itself a leaf node it is mapped to itself
            self.leaf_descendents = set([self.name])
            self.leaf_descendents_of_child = defaultdict(set) 
            return
            
        active_nodes = []
        active_nodes += self.children
        # set of all leaf descendents on both side
        leaf_descendents = set()
        # leaf descendents on each child node, if child is itself a leaf node it is mapped to itself
        leaf_descendents_of_child = defaultdict(set) 
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.is_leaf():
                    leaf_descendents.add(node.name)
                    leaf_descendents_of_child[self.closest_descendent_for(node.name).name].add(node.name)
            new_active_nodes = []
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        self.leaf_descendents = leaf_descendents
        self.leaf_descendents_of_child = leaf_descendents_of_child

    def assign_all_descendents(self):
        active_nodes = []
        active_nodes += [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                node.assign_descendents()
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes       

        self.assign_all_leaf_descendents()

    def assign_all_leaf_descendents(self):
        active_nodes = []
        active_nodes += [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                node.assign_leaf_descendents()
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                            


    def is_descendent(self, name):
        return name in self.descendents        
    
    def num_descendents(self):
        return len(self.descendents)
    
    def num_leaf_descendents(self):
        return len(self.leaf_descendents)


    # def closest_descendent_for(self,name):
    #     breakpoint()
    # 	if name in self.children_names(): 
    # 		return self.get_node(name)
    # 	else:
    #         return [child for child in self.children if name in child.descendents][0]

    def closest_descendent_for(self,name): 
        if name in self.children_names():
            return self.get_node(name)
        else:
            return [child for child in self.children if name in child.descendents][0]


    def has_logits(self):
        return self.num_children() > 1

    # def get_distribution(self):                
    #     if self.has_logits():
    #         return torch.nn.functional.softmax(self.logits,1)
    #     else:
    #         batch_size = self.logits.size(0)
    #         return torch.ones((batch_size,1))

        
    def distribution_over_furthest_descendents(self, net, batch_size, out, leave_out_classes=None, apply_overspecificity_mask=False, device='cuda', softmax_tau=1):
        if leave_out_classes is None:
            leave_out_classes = []

        # for child_idx, child in enumerate(self.children):
        #     if len(list(set(leave_out_classes) & set(self.leaf_descendents_of_child[child.name]))) > 0:
        #         print(self.name, child.name, child_idx)
        #         print('Without temp', F.softmax(torch.log1p(out[self.name]**2),1))
        #         print('With temp', F.softmax(torch.log1p(out[self.name]**2) / softmax_tau,1))
        #         pdb.set_trace()


        # if any([(child.is_leaf() and (child.name in leave_out_classes)) for child in self.children]):
        #     names = self.unwrap_names_of_joint(self.names_of_joint_distribution())
        #     left_out_descendant_name = [child for child in self.children if (child.is_leaf() and (child.name in leave_out_classes))][0].name
        #     bool_list = [name == left_out_descendant_name for name in names]
        #     return torch.tensor([int(value) for value in bool_list]).reshape(1, -1).repeat(batch_size,1).to(device)
        #     # return torch.ones(batch_size,1).to(device)
        
        if any([(child.leaf_descendents.issubset(set(leave_out_classes))) for child in self.children]):
            names = self.unwrap_names_of_joint(self.names_of_joint_distribution())
            left_out_descendant_name = [child for child in self.children if (child.is_leaf() and (child.name in leave_out_classes))][0].name
            bool_list = [name == left_out_descendant_name for name in names]
            # print(self.name, 'contains leave out class, last level possible')
            # pdb.set_trace()
            return torch.tensor([int(value) for value in bool_list]).reshape(1, -1).repeat(batch_size,1).to(device)


        if len(leave_out_classes) == 0:
            leave_out_classes = None

        if self.is_leaf():
            # print(self.name, 'leaf reached')
            # pdb.set_trace()
            return torch.ones(batch_size,1).to(device)
        else:
            if apply_overspecificity_mask:
                with torch.no_grad():
                    classification_weights = getattr(net, '_'+self.name+'_classification').weight
                    proto_presence = getattr(net, '_'+self.name+'_proto_presence')
                    proto_presence = F.gumbel_softmax(proto_presence, tau=0.5, hard=True, dim=-1)
                    masked_classification_weights = proto_presence[:, 1].unsqueeze(0) * classification_weights
                    all_protos_masked = False
                    for class_idx in range(masked_classification_weights.shape[0]):
                        if (masked_classification_weights[class_idx, :] <= 1e-3).all():
                            all_protos_masked = True
                            break
                    if all_protos_masked: # if even one of the class's protos are entirely masked then assume equal probability for each child class
                        
                        # if len(set(leave_out_classes) & self.leaf_descendents) > 0:
                        # label_to_child = {v:k for k,v in self.children_to_labels.items()}
                        # proto_count_each_child = { label_to_child[class_idx]:(masked_classification_weights[class_idx, :] > 1e-3).sum() for class_idx in range(masked_classification_weights.shape[0])}
                        # print(self.name, 'no proto')
                        # print(proto_count_each_child)
                        # pdb.set_trace()

                        # return torch.cat([torch.tensor([1./self.num_children()]*batch_size).view(batch_size,1).to(device) * self.children[i].distribution_over_furthest_descendents(net=net, batch_size=batch_size, out=out, leave_out_classes=leave_out_classes, \
                        #                                                                                                                                                             apply_overspecificity_mask=apply_overspecificity_mask, device=device,\
                        #                                                                                                                                                             softmax_tau=softmax_tau) for i in range(self.num_children())],1)
                        return torch.cat([torch.tensor([float(self.children[i].num_leaf_descendents()/self.num_leaf_descendents())]*batch_size).view(batch_size,1).to(device) * self.children[i].distribution_over_furthest_descendents(net=net, batch_size=batch_size, out=out, leave_out_classes=leave_out_classes, \
                                                                                                                                                                                    apply_overspecificity_mask=apply_overspecificity_mask, device=device,\
                                                                                                                                                                                    softmax_tau=softmax_tau) for i in range(self.num_children())],1)
                        # return torch.cat([torch.tensor([1.0]*batch_size).view(batch_size,1).to(device) * self.children[i].distribution_over_furthest_descendents(net=net, batch_size=batch_size, out=out, leave_out_classes=leave_out_classes, \
                        #                                                                                                                                                             apply_overspecificity_mask=apply_overspecificity_mask, device=device,\
                        #                                                                                                                                                             softmax_tau=softmax_tau) for i in range(self.num_children())],1)
                    # else:
                    #     for child_idx, child in enumerate(self.children):
                    #         if len(list(set(leave_out_classes) & set(self.leaf_descendents_of_child[child.name]))) > 0:
                    #             print(self.name, child.name, child_idx)
                    #             print('Without temp', F.softmax(torch.log1p(out[self.name]**2),1))
                    #             print('With temp', F.softmax(torch.log1p(out[self.name]**2) / softmax_tau,1))
                    #             pdb.set_trace()

            # # if len(set(leave_out_classes) & self.leaf_descendents) > 0:
            # label_to_child = {v:k for k,v in self.children_to_labels.items()}
            # # proto_count_each_child = { label_to_child[class_idx]:(masked_classification_weights[class_idx, :] > 1e-3).sum() for class_idx in range(masked_classification_weights.shape[0])}
            # prob_each_child = {label_to_child[i]:F.softmax(torch.log1p(out[self.name]**2) / softmax_tau,1)[:,i] for i in range(self.num_children())}
            # print(self.name)
            # # print(proto_count_each_child)
            # print(prob_each_child)
            # pdb.set_trace()
                    
            # try:
            return torch.cat([F.softmax(torch.log1p(out[self.name]**2) / softmax_tau,1)[:,i].view(batch_size,1) * self.children[i].distribution_over_furthest_descendents(net=net, batch_size=batch_size, out=out, leave_out_classes=leave_out_classes, \
                                                                                                                                                            apply_overspecificity_mask=apply_overspecificity_mask, device=device, softmax_tau=softmax_tau) \
                              for i in range(self.num_children())],1)            
            # except:
            #     pdb.set_trace()
        """
        torch.nn.functional.softmax(out[self.name],1)[:,0].view(batch_size,1)
        """
        # if not self.has_logits():
        #     return torch.ones(batch_size,1).cuda()
        # else:
        #     return torch.cat([torch.nn.functional.softmax(self.logits,1)[:,i].view(batch_size,1) * self.children[i].distribution_over_furthest_descendents(batch_size) \
        #                       for i in range(self.num_children())],1)            

    def names_of_joint_distribution(self):
        if self.num_children() == 1:
            return [self.children[0].name]
        elif self.num_children() == 0:
            return [self.name]
        else:
            return [child.names_of_joint_distribution() for child in self.children]


    def unwrap_names_of_joint(self,names):
        # # poorly written function for unwrapping nested lists up to depth 3 -- will cause bug if class hierarchy is too deep
        # new_list = []
        # for item in names:
        #     if type(item) is not list:
        #         new_list.append(item)
        #     else:
        #         for subitem in item:
        #             if type(subitem) is not list:
        #                 new_list.append(subitem)
        #             else:
        #                 for subsubitem in subitem:
        #                     if type(subsubitem) is not list:
        #                         new_list.append(subsubitem)
        #                     else:
        #                         for subsubsubitem in subsubitem:
        #                             if type(subsubsubitem) is not list:
        #                                 new_list.append(subsubsubitem)

        def contains_list(names):
            for name in names:
                if type(name) is list:
                    return True
            return False
        
        if not contains_list(names):
            return names
        
        new_list = []
        for item in names:
            if type(item) is not list:
                new_list += [item]
            else:
                new_list += self.unwrap_names_of_joint(item)
        return new_list


    def assign_unif_distributions(self):
        for node in self.nodes_with_children():
            node.unif = (torch.ones(node.num_children()) / node.num_children()).cuda()

    def assign_proto_dirs(self):
        for node in self.nodes_with_children():
            node.proto_dir = node.name + "_prototypes"


    def nodes_without_children(self):        
        nodes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if not node.has_logits():
                    nodes.append(node)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return nodes 


    def __str__(self):
        return "Node for " + self.name


    def save_visualization(self):
        save_path='/home/harishbabu/projects/interpretable-image/HPnet/figs'
        filename='tree'
        graph = graphviz.Digraph(comment='Tree Visualization')
        self._visualize(graph)
        graph.render(filename=os.path.join(save_path, filename), format='png', view=True)
        return graph
    
    def _visualize(self, graph):
        graph.node(self.name)
        for child in self.children:
            child._visualize(graph)
            graph.edge(self.name, child.name)

    def __str__(self):
        return self._print()
    
    def _print(self, depth=0):
        output = '{}{}\n'.format('\t' * depth, self.name)
        for child in self.children:
            output += child._print(depth=depth+1)
        return output

# for debugging, run node.py
if __name__ == "__main__":

    x = Node('cats',0)
    x.add_children(['leopard','tiger'])
    x.add_children(['lion','house cat'])
    lion = x.get_child('lion')
    lion.add_children('test')
    x.add_children_to('house cat',['tony','strappy'])
    x.add_children_to('tony',['paws','tail'])
    x.add_children_to('paws',['terrible'])
    #print(x.children[2].children[0].name)
    # print(x.children_names())
    # print(lion.children_names())
    # print(lion.parent.name)
    # print(x.get_node('test'))
    # setattr(x,"test",2)
    # print(x.test)
    # print(x.name)
    # print(x.children_to_labels)
    # print(x.children)
    print(x.get_child("lion").name)
    print(getattr(x,"children"))
    # print(x.class_to_num_children()) 

    full_list = x.names_of_joint_distribution()
    print(full_list)
    print(x.unwrap_names_of_joint(full_list))
    #print([item for sublist in full_list for item in sublist])
    
    # root = Node('root')
    # root.add_children(['vehicle','animal'])
    # root.add_children_to('vehicle',['airplane','automobile','ship','truck'])
    # root.add_children_to('animal',['bird','cat','deer','dog','frog','horse'])
    


