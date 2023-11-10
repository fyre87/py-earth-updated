# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

from ._record cimport PruningPassIteration
from ._util cimport gcv, apply_weights_2d
import numpy as np
import copy
from sklearn.linear_model import LinearRegression

from collections import defaultdict

GCV, RSS, NB_SUBSETS = "gcv", "rss", "nb_subsets"
FEAT_IMP_CRITERIA = (GCV, RSS, NB_SUBSETS)


# Returns a list of all the leaf nodes in an array
def get_array_of_leafs(curr_node):
    listy = []
    if curr_node.left_child == None and curr_node.right_child == None:
        return [curr_node]
    else:
        # Still have children to traverse
        listy = listy + get_array_of_leafs(curr_node.left_child) + get_array_of_leafs(curr_node.right_child)
    return listy

def get_full_GCV(leaf_array, main_node):
    GCV = 0
    for i in range(0, len(leaf_array)):
        GCV = GCV + leaf_array[i].node_GCV*(leaf_array[i].n / main_node.n)
    return GCV

def set_prunable(curr_node):
    # Only do this for the top node
    #if curr_node.parent == None:

    if (curr_node.left_child == None and curr_node.right_child != None) or (curr_node.left_child != None and curr_node.right_child == None):
        ValueError("Node only had 1 child. Doesn't make sense!")
    if curr_node.left_child != None and curr_node.right_child != None:
        set_prunable(curr_node.left_child)
        set_prunable(curr_node.right_child)

    #The top node has no parent. That one should be prunable everywhere!
    if curr_node.parent != None:
        for i in range(1, len(curr_node.Earth.basis_)):
            in_parent = check_in_parent(curr_node.Earth.basis_[i], curr_node.parent.Earth.basis_)
            if in_parent == True:
                #print("Gotta set this to unprunable: ")
                #print(curr_node.Earth.basis_[i])
                #print("See prunable thing: ", curr_node.Earth.basis_[i].is_prunable())
                curr_node.Earth.basis_[i].set_unprunable()
                #print("Now prunable thing: ", curr_node.Earth.basis_[i].is_prunable())
        
def check_in_parent(child_basis, parent_basis_list):
    # Start at 1 to avoid the constant basis function
    for i in range(1, len(parent_basis_list)):
        #print("Child: ", str(child_basis))
        #print("Parent: ", str(parent_basis_list[i]))
        if str(child_basis) == str(parent_basis_list[i]):
            return True
    return False

'''
cpdef FLOAT_t gcv(FLOAT_t mse, FLOAT_t basis_size, FLOAT_t data_size,
                FLOAT_t penalty):
    return mse * gcv_adjust(basis_size, data_size, penalty)

cpdef FLOAT_t gcv_adjust(FLOAT_t basis_size, FLOAT_t data_size,
                        FLOAT_t penalty):
    cdef FLOAT_t effective_parameters
    effective_parameters = basis_size + penalty * (basis_size - 1) / 2.0
    return 1.0 / ( ( (1.0 - (effective_parameters / data_size)) ** 2 ) )
'''
# Runs into an infinite loop when entries are 3, 6, 3
# effective_parameters = 3 + 3 * (3 - 1) / 2.0 = 6
# return mse* (1/ ( ( (1.0 - (6 / 6)) ** 2 ) )

#Temporary solution:::
#if (B.shape[1] + self.penalty * (B.shape[1]-1)/2.0) == leaf_array[i].n:
#    gcv_i = gcv(mse, B.shape[1], leaf_array[i].n, self.penalty+1)
#else:
#    gcv_i = gcv(mse, B.shape[1], leaf_array[i].n, self.penalty)


# CHANGE THIS: 
def calculate_GCV(mse, basis_size, data_size, penalty):
    if (basis_size + penalty * (basis_size-1)/2.0) == data_size:
        if basis_size == 1:
            # Then basis_size == data_size so MSE must equal 0!
            if mse != 0.0:
                raise ValueError("MSE should not be this high")
            return mse
        else:
            return gcv(mse, basis_size, data_size, penalty+1)
    else:
        return gcv(mse, basis_size, data_size, penalty)




cdef class PruningPasser:
    '''Implements the generic pruning pass as described by Friedman, 1991.'''
    def __init__(PruningPasser self, Basis basis,
                 cnp.ndarray[FLOAT_t, ndim=2] X,
                 cnp.ndarray[BOOL_t, ndim=2] missing,
                 cnp.ndarray[FLOAT_t, ndim=2] y,
                 cnp.ndarray[FLOAT_t, ndim=2] sample_weight, int verbose,
                 **kwargs):
        self.X = X
        self.missing = missing
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.y = y
        self.sample_weight = sample_weight
        self.verbose = verbose
        self.basis = basis
        self.B = np.ones(shape=(self.m, len(self.basis) + 1), dtype=float)
        self.penalty = kwargs.get('penalty', 3.0)
        if sample_weight.shape[1] == 1:
            y_avg = np.average(self.y, weights=sample_weight[:,0], axis=0)
        else:
            y_avg = np.average(self.y, weights=sample_weight, axis=0)

        # feature importance
        feature_importance_criteria = kwargs.get("feature_importance_type", [])
        if isinstance(feature_importance_criteria, basestring):
            feature_importance_criteria = [feature_importance_criteria]
        self.feature_importance = dict()
        for criterion in feature_importance_criteria:
            self.feature_importance[criterion] = np.zeros((self.n,))


    def best_combine_GCV(self, leaf_array):
        # Finds the best GCV if you subset
        best_iteration_GCV = float('inf')
        best_iteration_stored = None
        for i in range(0, len(leaf_array)-1):
            
            if leaf_array[i].parent == leaf_array[i+1].parent:
                # Then they are children of same parent and you can maybe combine!!!
                # Error: can easily make this faster by just checking if can_combine is false
                can_combine = True
                for j in range(0, len(leaf_array[i].Earth.basis_)):
                    if ((leaf_array[i].Earth.basis_[j].is_pruned() == False) and (leaf_array[i].Earth.basis_[j].is_prunable() == True)):
                        can_combine = False
                for j in range(0, len(leaf_array[i+1].Earth.basis_)):
                    if ((leaf_array[i+1].Earth.basis_[j].is_pruned() == False) and (leaf_array[i+1].Earth.basis_[j].is_prunable() == True)):
                        can_combine = False

                if can_combine == True:
                    # Calculate the GCV of doing so using the stored values in the nodes
                    curr_GCV = 0

                    # Add all the children's GCV up except the ones you are combining
                    for j in range(0, len(leaf_array)):
                        if j != i and j != i+1:
                            # Note: if the leaf node has only 1 observation then 
                            # leaf_array[k].node_GCV will be nan. Thus just set the mse or gcv to be 0
                            if leaf_array[j].n > 1:
                                #print("Node_GCV: ", leaf_array[j].node_GCV)
                                #print("Earth GCV: ", leaf_array[j].Earth.gcv_)
                                curr_GCV = curr_GCV + leaf_array[j].node_GCV*(leaf_array[j].n / self.X.shape[0])
                            else:
                                # Symbolic code
                                curr_GCV = curr_GCV + 0
                    # Add the combined GCV of the parent thing
                    #print("Parent_GCV: ", leaf_array[i].parent.node_GCV)
                    #print("Ready to add: ", leaf_array[i].parent.node_GCV*(leaf_array[i].parent.n / self.X.shape[0]))
                    curr_GCV = curr_GCV + leaf_array[i].parent.node_GCV*(leaf_array[i].parent.n / self.X.shape[0])
                    #print("curr_GCV: ", curr_GCV)
                    if curr_GCV < best_iteration_GCV:
                        best_iteration_GCV = curr_GCV
                        best_iteration_stored = i
                    
        #print("BEST: ", best_iteration_GCV)
        return best_iteration_GCV, best_iteration_stored
                




    def node_prune(PruningPasser self, main_node):
        leaf_array = get_array_of_leafs(main_node)
        #Error: This set prunable method doesn't work idk why???
        # It makes an infinite loop sometimes!
        set_prunable(main_node) # Set a bunch of things to be unprunable

        best_iteration_GCV = float('inf')
        best_iteration_stored = None

        # Keep pruning and storing the prunes until the whole tree is gone
        # Error: This while loop should maybe use plen instead, does pruning decrease the basis function number?

        #print("main_node.Earth.basis_: ")
        #print(main_node.Earth.basis_)
        #print("plen() of it: ", main_node.Earth.basis_.plen())

        #iter = 0
        while main_node.Earth.basis_.plen() > 1 or (main_node.left_child != None and main_node.right_child != None):
            # Store details of the next prune to do
            next_iteration_GCV = float('inf')
            next_iteration_stored = None
            
            GCV_combine, combine_iteration = self.best_combine_GCV(leaf_array)
            
            # Error! Make sure GCV combine isn't always infinity. That it actually works sometimes!
            if GCV_combine < next_iteration_GCV:
                next_iteration_GCV = GCV_combine
                next_iteration_stored = combine_iteration

            # See what the best way to prune basis functions is
            for i in range(0, len(leaf_array)):
                # Start at 1 since the first basis is the constant function. Don't want to prune that!
                for j in range(0, len(leaf_array[i].Earth.basis_)):
                    if leaf_array[i].Earth.basis_[j].is_pruned():
                        continue
                    if leaf_array[i].Earth.basis_[j].is_prunable() == False:
                        # The constant basis function is not prunable
                        #** Error: Hinge basis functions did not have a method for this already
                        # I added one in. However, how the heck were they pruned off before?
                        continue
                    
                    #print("made it here")
                    leaf_array[i].Earth.basis_[j].prune()
                    #print("Made it here!")
                    # Get the MSE and GCV post pruning of leaf i
                    B = np.empty(shape=(leaf_array[i].X.shape[0], leaf_array[i].Earth.basis_.plen()), order='F')
                    leaf_array[i].Earth.basis_.transform(leaf_array[i].X, leaf_array[i].missing, B)
                    
                    #if B.shape[0] == 1:
                    #    B = B[:, 0:1]
                    #else:
                    #    B = B[:, ~np.all(B == 0, axis = 0)]

                    # TODO: make this sklearn instead cuz singular matrix errors grrr
                    
                    
                    '''
                    try: 
                        #print("B: ", B)
                        #coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(B.T, B)), B.T), leaf_array[i].y)
                        #print("Printing coef")
                        #print(coef)
                        model = LinearRegression().fit(B, leaf_array[i].y)
                        #print(model.coef_)
                    except:
                        print("np.matmul(B.T, B): ")
                        print(np.matmul(B.T, B))
                        raise ValueError("Singular Matrix")
                    '''
                    
                    
                    #y_pred = (coef.T * B).sum(axis = 1)
                    #mse = ((((y_pred.reshape(y_pred.shape[0], 1) - leaf_array[i].y)**2).sum()) / leaf_array[i].y.shape[0])
                    model = LinearRegression().fit(B, leaf_array[i].y)
                    y_pred = model.predict(B)
                    mse = ((((y_pred - leaf_array[i].y)**2).sum()) / leaf_array[i].y.shape[0])
                    #print("mse: ", mse)
                    #print("B.shape[1]: ", B.shape[1])
                    #print("B: ")
                    #print(B)
                    #print("Leaf_array[i].n: ", leaf_array[i].n)
                    #print("self.penalty: ", self.penalty)
                    
                    gcv_i = calculate_GCV(mse, B.shape[1], leaf_array[i].n, self.penalty)
                    #print("gcv_i: ", gcv_i)

                    # Calculate the GCV post pruning for the whole tree
                    GCV_post_prune = 0
                    for k in range(0, len(leaf_array)):
                        if k != i:
                            # Note: if the leaf node has only 1 observation then 
                            # leaf_array[k].node_GCV will be nan. Thus just set the mse or gcv to be 0
                            if leaf_array[k].n > 1:
                                GCV_post_prune = GCV_post_prune + leaf_array[k].node_GCV*(leaf_array[k].n / main_node.n)
                            else:
                                # Symbolic
                                GCV_post_prune = GCV_post_prune + 0
                    #print("GCV_post_prune pre final add: ", GCV_post_prune)
                    GCV_post_prune = GCV_post_prune + gcv_i*(leaf_array[i].n / main_node.n)
                    #print("GCV_post_prune: ", GCV_post_prune)
                    leaf_array[i].Earth.basis_[j].unprune()

                    if GCV_post_prune < next_iteration_GCV:
                        next_iteration_GCV = GCV_post_prune
                        next_iteration_stored = [i, j]


            
            # Now you have found the best combine and the best next_iteration_GCV thing. 
            # Now do the pruning / combining
            #print("Length of leaf array pre combine/prune: ", len(leaf_array))
            #print(GCV_combine)
            #print(next_iteration_GCV)
            #print(next_iteration_stored)
            if type(next_iteration_stored) == type([]):
                # If it is a list then you just have to prune something!
                leaf_array[next_iteration_stored[0]].Earth.basis_[next_iteration_stored[1]].prune()

                # Then need to recalculate the node_GCV
                B = np.empty(shape=(leaf_array[next_iteration_stored[0]].X.shape[0], leaf_array[next_iteration_stored[0]].Earth.basis_.plen()), order='F')
                leaf_array[next_iteration_stored[0]].Earth.basis_.transform(leaf_array[next_iteration_stored[0]].X, leaf_array[next_iteration_stored[0]].missing, B)

                # Get the GCV post pruning to update node_GCV
                '''
                if B.shape[0] == 1:
                    B = B[:, 0:1]
                else:
                    B = B[:, ~np.all(B == 0, axis = 0)]
                '''
                
                #coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(B.T, B)), B.T), leaf_array[next_iteration_stored[0]].y)
                #y_pred = (coef.T * B).sum(axis = 1)
                model = LinearRegression().fit(B, leaf_array[next_iteration_stored[0]].y)
                y_pred = model.predict(B)
                mse = ((((y_pred - leaf_array[next_iteration_stored[0]].y)**2).sum()) / leaf_array[i].y.shape[0])
                # Neccesary to avoid divide by 0 errors: 
                leaf_array[next_iteration_stored[0]].node_GCV = calculate_GCV(mse, B.shape[1], leaf_array[next_iteration_stored[0]].n, self.penalty)

            elif next_iteration_stored != None:
                # Then need to combine a subset!
                # Make the parents children none
                leaf_array[next_iteration_stored].parent.left_child = None
                leaf_array[next_iteration_stored].parent.right_child = None

                # Then replace the two leafs with the one parent
                leaf_array[next_iteration_stored + 1] = leaf_array[next_iteration_stored + 1].parent
                del leaf_array[next_iteration_stored]
                # Now do poggers things::
            else:
                print("leaf_array: ")
                print(leaf_array)
                print("next_iteration_GCV: ")
                print(next_iteration_GCV)
                print("GCV_combine: ")
                print(GCV_combine)
                raise ValueError("Should not have got here. Error in the pruning stage")

            if next_iteration_GCV < best_iteration_GCV:
                best_iteration_GCV = next_iteration_GCV
                best_iteration_stored = copy.deepcopy(main_node) # Error: Need to copy this properly!!

            
            #print("Next iteration GCV: ", next_iteration_GCV)
            # Reset everything and loop again until done!
            next_iteration_GCV = float('inf')
            next_iteration_stored = None

        #print("best_iteration_GCV: ", best_iteration_GCV)
        return best_iteration_stored






















    cpdef run(PruningPasser self):
        # This is a totally naive implementation and could potentially be made
        # faster through the use of updating algorithms.  It is not clear that
        # such optimization would be worthwhile, as the pruning pass is not the
        # slowest part of the algorithm.
        cdef INDEX_t i
        cdef INDEX_t j
        cdef long v
        cdef INDEX_t basis_size = len(self.basis)
        cdef INDEX_t pruned_basis_size = self.basis.plen()
        cdef FLOAT_t gcv_
        cdef INDEX_t best_iteration
        cdef INDEX_t best_bf_to_prune
        cdef FLOAT_t best_gcv
        cdef FLOAT_t best_iteration_gcv
        cdef FLOAT_t best_iteration_mse
        cdef FLOAT_t mse, mse0, total_weight

        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef cnp.ndarray[BOOL_t, ndim = 2] missing = (
            <cnp.ndarray[BOOL_t, ndim = 2] > self.missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] y = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.y)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] sample_weight = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.sample_weight)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] weighted_y

        if self.verbose >= 1:
            print('Beginning pruning pass')

        # Initial solution
        mse = 0.
        mse0 = 0.
        total_weight = 0.
        for p in range(y.shape[1]):
            if sample_weight.shape[1] == 1:
                weighted_y = y[:,p] * np.sqrt(sample_weight[:,0])
                self.basis.weighted_transform(X, missing, B, sample_weight[:, 0])
                total_weight += np.sum(sample_weight[:,0])
                mse0 += np.sum(sample_weight[:,0] * (y[:,p] - np.average(y[:,p], weights=sample_weight[:,0])) ** 2)
            else:
                weighted_y = y[:,p] * np.sqrt(sample_weight[:,p])
                self.basis.weighted_transform(X, missing, B, sample_weight[:, p])
                total_weight += np.sum(sample_weight[:,p])
                mse0 += np.sum(sample_weight[:,p] * (y[:,p] - np.average(y[:,p], weights=sample_weight[:,p])) ** 2)
            if sample_weight.shape[1] == 1:
                self.basis.weighted_transform(X, missing, B, sample_weight[:, 0])
            else:
                self.basis.weighted_transform(X, missing, B, sample_weight[:, p])
            beta, mse_ = np.linalg.lstsq(B[:, 0:(basis_size)], weighted_y)[0:2]
            if mse_:
                pass
            else:
                mse_ = np.sum(
                    (np.dot(B[:, 0:basis_size], beta) - weighted_y) ** 2)
            mse += mse_

        # Create the record object
        self.record = PruningPassRecord(
            self.m, self.n, self.penalty, mse0 / total_weight, pruned_basis_size, mse / total_weight)
        gcv_ = self.record.gcv(0)
        best_gcv = gcv_
        best_iteration = 0

        if self.verbose >= 1:
            print(self.record.partial_str(slice(-1, None, None), print_footer=False))

        # init feature importance
        prev_best_iteration_gcv = None
        prev_best_iteration_mse = None

        # Prune basis functions sequentially
        for i in range(1, pruned_basis_size):
            first = True
            pruned_basis_size -= 1

            # Find the best basis function to prune
            for j in range(basis_size):
                bf = self.basis[j]
                if bf.is_pruned():
                    continue
                if not bf.is_prunable():
                    continue
                bf.prune()


                mse = 0.
                for p in range(y.shape[1]):
                    if sample_weight.shape[1] == 1:
                        weighted_y = y[:,p] * np.sqrt(sample_weight[:,0])
                        self.basis.weighted_transform(X, missing, B, sample_weight[:, 0])
                    else:
                        weighted_y = y[:,p] * np.sqrt(sample_weight[:,p])
                        self.basis.weighted_transform(X, missing, B, sample_weight[:, p])
                    beta, mse_ = np.linalg.lstsq(
                        B[:, 0:pruned_basis_size], weighted_y)[0:2]
                    if mse_:
                        pass
#                         mse_ /= np.sum(self.sample_weight)
                    else:
                        mse_ = np.sum((np.dot(B[:, 0:pruned_basis_size], beta) -
                                    weighted_y) ** 2) #/ np.sum(sample_weight)
                    mse += mse_# * output_weight[p]
                gcv_ = gcv(mse / np.sum(sample_weight), pruned_basis_size, self.m, self.penalty)

                if gcv_ <= best_iteration_gcv or first:
                    best_iteration_gcv = gcv_
                    best_iteration_mse = mse
                    best_bf_to_prune = j
                    first = False
                bf.unprune()

            # Feature importance
            if i > 1:
                # having selected the best basis to prune, we compute how much
                # that basis decreased the mse and gcv relative to the previous mse and gcv
                # respectively.
                mse_decrease = (best_iteration_mse - prev_best_iteration_mse)
                gcv_decrease = (best_iteration_gcv - prev_best_iteration_gcv)
                variables = set()
                bf = self.basis[best_bf_to_prune]
                for v in bf.variables():
                    variables.add(v)
                for v in variables:
                    if RSS in self.feature_importance:
                        self.feature_importance[RSS][v] += mse_decrease
                    if GCV in self.feature_importance:
                        self.feature_importance[GCV][v] += gcv_decrease
                    if NB_SUBSETS in self.feature_importance:
                        self.feature_importance[NB_SUBSETS][v] += 1
            # The inner loop found the best basis function to remove for this
            # iteration. Now check whether this iteration is better than all
            # the previous ones.
            if best_iteration_gcv <= best_gcv:
                best_gcv = best_iteration_gcv
                best_iteration = i

            prev_best_iteration_gcv = best_iteration_gcv
            prev_best_iteration_mse = best_iteration_mse
            # Update the record and prune the selected basis function
            self.record.append(PruningPassIteration(
                best_bf_to_prune, pruned_basis_size, best_iteration_mse / total_weight))
            self.basis[best_bf_to_prune].prune()

            if self.verbose >= 1:
                print(self.record.partial_str(slice(-1, None, None), print_header=False, print_footer=(pruned_basis_size == 1)))

        # Unprune the basis functions pruned after the best iteration
        self.record.set_selected(best_iteration)
        self.record.roll_back(self.basis)
        if self.verbose >= 1:
            print(self.record.final_str())

        # normalize feature importance values
        for name, val in self.feature_importance.items():
            if name == 'gcv':
                val[val < 0] = 0 # gcv can have negative feature importance correponding to an increase of gcv, set them to zero
            if val.sum() > 0:
                val /= val.sum()
            self.feature_importance[name] = val

    cpdef PruningPassRecord trace(PruningPasser self):
        return self.record

