# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

from ._util cimport gcv_adjust, log2, apply_weights_1d, apply_weights_slice
from ._basis cimport (Basis, BasisFunction, ConstantBasisFunction,
                      HingeBasisFunction, LinearBasisFunction,
                      MissingnessBasisFunction)
from ._record cimport ForwardPassIteration
from ._types import BOOL, INT
from ._knot_search cimport knot_search, MultipleOutcomeDependentData, PredictorDependentData, \
    KnotSearchReadOnlyData, KnotSearchState, KnotSearchWorkingData, KnotSearchData
import sys
import time
import copy
import _pickle as cPickle
from libc.math cimport sqrt, abs, log
from sklearn.linear_model import LinearRegression

import numpy as np
cnp.import_array()

from heapq import heappush, heappop
class FastHeapContent:

    def __init__(self, idx, mse=-np.inf, m=-np.inf, v=None):
        """
        This class defines an entry of the priority queue as defined in [1].
        The entry stores information about parent basis functions and is
        used by the priority queue in the forward pass
        to choose the next parent basis function to try.

        References
        ----------
        .. [1] Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993.

        """
        self.idx = idx
        self.mse = mse
        self.m = m
        self.v = v

    def __lt__(self, other):
        return self.mse < other.mse

cdef int MAXTERMS = 0
cdef int MAXRSQ = 1
cdef int NOIMPRV = 2
cdef int LOWGRSQ = 3
cdef int NOCAND = 4
stopping_conditions = {
    MAXTERMS: "Reached maximum number of terms",
    MAXRSQ: "Achieved RSQ value within threshold of 1",
    NOIMPRV: "Improvement below threshold",
    LOWGRSQ: "GRSQ too low",
    NOCAND: "No remaining candidate knot locations"
}

cdef class ForwardPasser:

    def __init__(ForwardPasser self, cnp.ndarray[FLOAT_t, ndim=2] X,
                 cnp.ndarray[BOOL_t, ndim=2] missing,
                 cnp.ndarray[FLOAT_t, ndim=2] y,
                 cnp.ndarray[FLOAT_t, ndim=2] sample_weight,
                 **kwargs):

        cdef INDEX_t i
        self.X = X
        self.missing = missing
        self.y = y
        self.Node = None

        # Assuming Earth.fit got capital W (the inverse of squared variance)
        # so the objective function is (sqrt(W) * residual) ^ 2)
        self.sample_weight = np.sqrt(sample_weight)
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.endspan       = kwargs.get('endspan', -1)
        self.minspan       = kwargs.get('minspan', -1)
        self.endspan_alpha = kwargs.get('endspan_alpha', .05)
        self.minspan_alpha = kwargs.get('minspan_alpha', .05)
        self.max_terms     = kwargs.get('max_terms', min(2 * self.n + self.m // 10, 400))
        self.allow_linear  = kwargs.get('allow_linear', True)
        self.max_degree    = kwargs.get('max_degree', 1)
        self.thresh        = kwargs.get('thresh', 0.001)
        self.penalty       = kwargs.get('penalty', 3.0)
        self.check_every   = kwargs.get('check_every', -1)
        self.min_search_points = kwargs.get('min_search_points', 100)
        self.xlabels       = kwargs.get('xlabels')
        self.use_fast = kwargs.get('use_fast', False)
        self.fast_K = kwargs.get("fast_K", 5)
        self.fast_h = kwargs.get("fast_h", 1)
        self.zero_tol = kwargs.get('zero_tol', 1e-12)
        self.allow_missing = kwargs.get("allow_missing", False)
        self.verbose = kwargs.get("verbose", 0)
        self.allow_subset = kwargs.get("allow_subset", True)
        if self.allow_missing:
            self.has_missing = np.any(self.missing, axis=0).astype(BOOL)

        self.fast_heap = []

        if self.xlabels is None:
            self.xlabels = ['x' + str(i) for i in range(self.n)]
        if self.check_every < 0:
            self.check_every = (<int > (self.m / self.min_search_points)
                                if self.m > self.min_search_points
                                else 1)

        weighted_mean = np.mean((self.sample_weight ** 2) * self.y)
        self.sst = np.sum((self.sample_weight * (self.y - weighted_mean)) ** 2)
        self.basis = Basis(self.n)
        self.basis.append(ConstantBasisFunction())
        if self.use_fast is True:
            content = FastHeapContent(idx=0)
            heappush(self.fast_heap, content)

        self.mwork = np.empty(shape=self.m, dtype=np.int64)

        self.B = np.ones(
            shape=(self.m, self.max_terms + 4), order='F', dtype=float)
        self.basis.transform(self.X, self.missing, self.B[:,0:1])

        if self.endspan < 0:
            self.endspan = round(3 - log2(self.endspan_alpha / self.n))

        self.linear_variables = np.zeros(shape=self.n, dtype=INT)
        self.init_linear_variables()

        # Removed in favor of new knot search code
        self.iteration_number = 0

        # Add in user selected linear variables
        for linvar in kwargs.get('linvars',[]):
            if linvar in self.xlabels:
                self.linear_variables[self.xlabels.index(linvar)] = 1
            elif linvar in range(self.n):
                self.linear_variables[linvar] = 1
            else:
                raise IndexError(
                    'Unknown variable selected in linvars argument.')

        # Initialize the data structures for knot search
        self.n_outcomes = self.y.shape[1]
        n_predictors = self.X.shape[1]
        n_weights = self.sample_weight.shape[1]
        self.workings = []
        self.outcome = MultipleOutcomeDependentData.alloc(self.y, self.sample_weight, self.m,
                                                          self.n_outcomes, self.max_terms + 4,
                                                          self.zero_tol)
        self.outcome.update_from_array(self.B[:,0])
        self.total_weight = 0.
        for i in range(self.n_outcomes):
            working = KnotSearchWorkingData.alloc(self.max_terms + 4)
            self.workings.append(working)
            self.total_weight += self.outcome.outcomes[i].weight.total_weight
        self.predictors = []
        for i in range(n_predictors):
            x = self.X[:, i].copy()
            x[missing[:,i]==1] = 0.
            predictor = PredictorDependentData.alloc(x)
            self.predictors.append(predictor)

        # Initialize the forward pass record
        self.record = ForwardPassRecord(
            self.m, self.n, self.penalty, self.outcome.mse(), self.xlabels)



    # Return basis functions
    cpdef Basis get_basis(ForwardPasser self):
        # Very important to use pickle, it is essentially doing a deep copy and avoids pointer issues
        data = cPickle.dumps(self.basis)
        basis_copy = cPickle.loads(data)
        return basis_copy

    def set_basis(ForwardPasser self, x):
        # Very important to use pickle, it is essentially doing a deep copy and avoids pointer issues
        data = cPickle.dumps(x)
        basis_copy = cPickle.loads(data)
        self.basis = basis_copy

    cpdef init_linear_variables(ForwardPasser self):
        cdef INDEX_t variable
        cdef cnp.ndarray[INT_t, ndim = 1] order
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef ConstantBasisFunction root_basis_function = self.basis[0]
        for variable in range(self.n):
            order = np.argsort(X[:, variable])[::-1].astype(np.int64)
            if root_basis_function.valid_knots(B[order, 0], X[order, variable],
                                               variable, self.check_every,
                                               self.endspan, self.minspan,
                                               self.minspan_alpha, self.n,
                                               self.mwork).shape[0] == 0:
                linear_variables[variable] = 1
            else:
                linear_variables[variable] = 0

    cpdef run(ForwardPasser self):
        if self.verbose >= 1:
            print('Beginning forward pass')
            print(self.record.partial_str(slice(-1, None, None), print_footer=False))

        if self.max_terms <= len(self.basis):
            # We done!
            return
        if self.max_terms > 1 and self.record.mse(0) != 0.:
            while True:
                self.next_pair()
                if self.stop_check():
                    if self.verbose >= 1:
                        print(self.record.partial_str(slice(-1, None, None), print_header=False, print_footer=True))
                        print(self.record.final_str())
                    break
                else:
                    if self.verbose >= 1:
                        print(self.record.partial_str(slice(-1, None, None), print_header=False, print_footer=False))
                if self.Node.left_child != None and self.Node.right_child != None:
                    # We have subset so we stop adjusting this forward passer

                    return 
                self.iteration_number += 1
        return

    cdef stop_check(ForwardPasser self):
        last = self.record.__len__() - 1
        if self.record.iterations[last].get_size() > self.max_terms:
            self.record.stopping_condition = MAXTERMS
            return True
        rsq = self.record.rsq(last)
        if rsq > 1 - self.thresh:
            self.record.stopping_condition = MAXRSQ
            return True
        if last > 0:
            previous_rsq = self.record.rsq(last - 1)
            if rsq - previous_rsq < self.thresh:
                self.record.stopping_condition = NOIMPRV
                return True
        if self.record.grsq(last) < -10:
            self.record.stopping_condition = LOWGRSQ
            return True
        if self.record.iterations[last].no_further_candidates():
            self.record.stopping_condition = NOCAND
            return True
        if self.record.mse(last) == self.zero_tol:
            self.record.stopping_condition = NOIMPRV
            return True
        return False


    cpdef orthonormal_update(ForwardPasser self, b):
        # Update the outcome data
        linear_dependence = False
        return_codes = []
        return_code = self.outcome.update_from_array(b)
        if return_code == -1:
            raise ValueError('This should not have happened.')
        if return_code == 1:
            linear_dependence = True
        return linear_dependence

    cpdef orthonormal_downdate(ForwardPasser self):
        self.outcome.downdate()

    def trace(self):
        return self.record

    # CHANGE THIS: 
    def subset_GCV(ForwardPasser self):
        
        optimal_GCV = float('inf')
        optimal_split = float('inf')
        optimal_var = float('inf')

        # Error: without this it splits infinitely a lot
        # Current failsafe to prevent infinite subsetting

        if self.X.shape[0] <= 10:
            #Then don't subset
            return optimal_GCV, optimal_var, optimal_split
        

        # This part may be inefficient due to numpy copying. 
        # Might wanna try direct self.X instead of new var?
        # Error: Is this unneccesary?? I just don't want pointers to be messed up. 
        X_sub = np.copy(self.X)
        missing_sub = np.copy(self.missing)
        y_sub = np.copy(self.y)
        sample_weight_sub = np.copy(self.sample_weight)

        #start_time = time.time()

        model1 = LinearRegression(fit_intercept = False)
        model2 = LinearRegression(fit_intercept = False)

        for variable in range(self.n):


            #np.random.seed(seed = 1)
            #print("about to make quantiles")
            #Quantiles = [np.quantile(self.X[:, variable], q=0), np.quantile(self.X[:, variable], q=0.1), np.quantile(self.X[:, variable], q=0.2), np.quantile(self.X[:, variable], q=0.3), np.quantile(self.X[:, variable], q=0.4), np.quantile(self.X[:, variable], q=0.5), np.quantile(self.X[:, variable], q=0.6), np.quantile(self.X[:, variable], q=0.7), np.quantile(self.X[:, variable], q=0.8), np.quantile(self.X[:, variable], q=0.9)]
            #print(np.unique(self.X[:, variable])[:-1])
            #print("about to print quantiles")
            #print(Quantiles)
            #print(np.unique(Quantiles))

            
            for split_val in np.unique(self.X[:, variable])[:-1]:
            #Error: For some reason looping through the quantiles makes it crash
            # but looping through everything doesn't???
            #for split_val in Quantiles:
            #for split_val in np.unique(Quantiles):

                
                # Alternative method to avoid orthonormal_updates
                # Go to length of self.basis cuz otherwise has too many rows and matrix is not invertible
                x1 = copy.deepcopy(self.B[X_sub[:, variable] <= split_val, 0:len(self.basis)]) 
                y1 = copy.deepcopy(y_sub[X_sub[:, variable] <= split_val, :])
                x2 = copy.deepcopy(self.B[X_sub[:, variable] >  split_val, 0:len(self.basis)])
                y2 = copy.deepcopy(y_sub[X_sub[:, variable] >  split_val, :])



                # Must chop out every column with 1 unique value in it. 
                # Cuz those ones make the matrix not invertible
                # Maybe need to check if number of vars exceeds number of observations??
                # But which variables to keep in that case??
                # a[:, ~np.all(a[1:] == a[:-1], axis=0)]



                #******
                # IDEAS: Use QR decomposition or use ridge regression. 

                '''
                if x1.shape[0] == 1:
                    x1 = x1[:, 0:1]
                else:
                    #x1[:, np.append(True, ~np.all(x1[:, 1:] == x1[0, 1:], axis=0))]
                    #x1 = x1[:, ~np.all(x1 == 0, axis = 0)]
                    x1 = x1[:, np.append(True, ~np.all(x1[:, 1:] == x1[0, 1:], axis=0))]

                if x2.shape[0] == 1:
                    x2 = x2[:, 0:1]
                else:
                    #x2 = x2[:, ~np.all(x2 == 0, axis = 0)]
                    x2 = x2[:, np.append(True, ~np.all(x2[:, 1:] == x2[0, 1:], axis=0))]
                try:
                    coef1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x1.T, x1)), x1.T), y1)
                    coef2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), x2.T), y2)
                except:
                    print(self.basis)
                    print(x1)
                    print(x1[:, np.append(True, ~np.all(x1[:, 1:] == x1[0, 1:], axis=0))])
                    print(x2)
                    print(x2[:, np.append(True, ~np.all(x2[:, 1:] == x2[0, 1:], axis=0))])
                    print(y1)
                    print(y2)
                    print(len(y1) + len(y2))
                    print(len(x1) + len(x2))
                    raise ValueError("Exception occurred, probably singular matrix")

                y_pred1 = (coef1.T * x1).sum(axis = 1)
                y_pred2 = (coef2.T * x2).sum(axis = 1)

                se1 = (y_pred1 - y1.reshape(y_pred1.shape[0]))**2
                se2 = (y_pred2 - y2.reshape(y_pred2.shape[0]))**2
                GCV = (se1.sum() + se2.sum())/(x1.shape[0] + x2.shape[0])
                # y_pred1 is in the form (n, ). y1 is in the form (n, 1). 
                # Can't be subtracted without reshaping.
                '''
                
                #'''
                # Alternative using SKlearn, avoids any nasty dumb stuff. 
                # But also very slow
                '''
                file1 = open("/Users/WilliamJamesonPattie_/Desktop/Cool Text", "a")
                file1.write("           SUB_GCV_loop : About to fit models\n")
                file1.write(str(x1) + "\n")
                file1.write(str(y1) + "\n")
                file1.close()
                '''

                model1.fit(x1, y1)
                model2.fit(x2, y2)
                y_pred1 = model1.predict(x1)
                y_pred2 = model2.predict(x2)
                #'''
                se1 = (y_pred1 - y1)**2
                se2 = (y_pred2 - y2)**2
                GCV = (se1.sum() + se2.sum())/(x1.shape[0] + x2.shape[0])
                    # Replace the old GCV with the new GCV
                if GCV < optimal_GCV:
                    optimal_GCV = GCV
                    optimal_var = variable
                    optimal_split = split_val

        return optimal_GCV, optimal_var, optimal_split


    cdef next_pair(ForwardPasser self):
        cdef INDEX_t variable
        cdef INDEX_t parent_idx
        cdef INDEX_t parent_degree
        cdef INDEX_t nonzero_count
        cdef BasisFunction parent
        cdef cnp.ndarray[INDEX_t, ndim = 1] candidates_idx
        cdef FLOAT_t knot
        cdef FLOAT_t mse
        cdef INDEX_t knot_idx
        cdef FLOAT_t knot_choice
        cdef FLOAT_t mse_choice
        cdef FLOAT_t mse_choice_cur_parent
        cdef int variable_choice_cur_parent
        cdef int knot_idx_choice
        cdef INDEX_t parent_idx_choice
        cdef BasisFunction parent_choice
        cdef BasisFunction new_parent
        cdef BasisFunction new_basis_function
        parent_basis_content_choice = None
        parent_basis_content = None
        cdef INDEX_t variable_choice
        cdef bint first = True
        cdef bint already_covered
        cdef INDEX_t k = len(self.basis)
        cdef INDEX_t endspan
        cdef bint linear_dependence
        cdef bint dependent
        # TODO: Shouldn't there be weights here?
        cdef FLOAT_t gcv_factor_k_plus_1 = gcv_adjust(k + 1, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_2 = gcv_adjust(k + 2, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_3 = gcv_adjust(k + 3, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_4 = gcv_adjust(k + 4, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_
        cdef FLOAT_t mse_
        cdef INDEX_t i
        cdef bint eligible
        cdef bint covered
        cdef bint missing_flag
        cdef bint choice_needs_coverage

        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef cnp.ndarray[BOOL_t, ndim = 2] missing = (
            <cnp.ndarray[BOOL_t, ndim = 2] > self.missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[BOOL_t, ndim = 1] has_missing = (
            <cnp.ndarray[BOOL_t, ndim = 1] > self.has_missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] b
        cdef cnp.ndarray[FLOAT_t, ndim = 1] p
        cdef bint variable_can_be_linear

        if self.use_fast:
            nb_basis = min(self.fast_K, k, len(self.fast_heap))
        else:
            nb_basis = k

        content_to_be_repushed = []
        for idx in range(nb_basis):
            # Iterate over parents
            if self.use_fast:
                # retrieve the next basis function to try as parent
                parent_basis_content = heappop(self.fast_heap)
                content_to_be_repushed.append(parent_basis_content)
                parent_idx = parent_basis_content.idx
                mse_choice_cur_parent = -1
                variable_choice_cur_parent = -1
            else:
                parent_idx = idx

            parent = self.basis.get(parent_idx)
            if not parent.is_splittable():
                continue

            if self.use_fast:
                # each "fast_h" iteration, force to pass through all the variables,
                if self.iteration_number - parent_basis_content.m >= self.fast_h:
                    variables = range(self.n)
                    parent_basis_content.m = self.iteration_number
                # in the opposite case, just use the last chosen variable
                else:
                    if parent_basis_content.v is not None:
                        variables = [parent_basis_content.v]
                    else:
                        variables = range(self.n)
            else:
                variables = range(self.n)

            parent_degree = parent.effective_degree()

            for variable in variables:
                # Determine whether this variable can be linear
                variable_can_be_linear = self.allow_linear and not parent.has_linear(variable)

                # Determine whether missingness needs to be accounted for.
                if self.allow_missing and has_missing[variable]:
                    missing_flag = True
                    eligible = parent.eligible(variable)
                    covered = parent.covered(variable)
                else:
                    missing_flag = False

                # Make sure not to exceed max_degree (but don't count the
                # covering missingness basis function if required)
                if self.max_degree >= 0:
                    if parent_degree >= self.max_degree:
                        continue

                # If there is missing data and this parent is not
                # an eligible parent for this variable with missingness
                # (because it includes a non-missing factor for the variable)
                # then skip this variable.
                if missing_flag and not eligible:
                    continue

                # Add the linear term to B
                predictor = self.predictors[variable]

#                 # If necessary, protect from missing data
#                 if missing_flag:
#                     B[missing[:, variable]==1, k] = 0.
#                     b = B[:, k]
#                     # Update the outcome data
#                     linear_dependence = self.orthonormal_update(b)

                if missing_flag and not covered:
                    p = B[:, parent_idx] * (1 - missing[:, variable])
                    b = B[:, parent_idx] * (1 - missing[:, variable])
                    self.orthonormal_update(b)
                    b = B[:, parent_idx] * missing[:, variable]
                    self.orthonormal_update(b)
                    q = k + 3
                else:
                    p = self.B[:, parent_idx]
                    q = k + 1

                b = p * predictor.x
                if missing_flag and not covered:
                    b[missing[:, variable] == 1] = 0
                linear_dependence = self.orthonormal_update(b)

                # If a new hinge function does not improve the gcv over the
                # linear term then just the linear term will be retained
                # (if variable_can_be_linear).  Calculate the gcv with just the linear
                # term in order to compare later.  Note that the mse with
                # another term never increases, but the gcv may because it
                # penalizes additional terms.
                mse_ = self.outcome.mse()
                if missing_flag and not covered:
                    gcv_ = gcv_factor_k_plus_3 * mse_
                else:
                    gcv_ = gcv_factor_k_plus_1 * mse_

                if linear_variables[variable]:
                    mse = mse_
                    knot_idx = -1
                else:
                    # Find the valid knot candidates
                    candidates, candidates_idx = predictor.knot_candidates(p, self.endspan,
                                                                           self.minspan,
                                                                           self.minspan_alpha,
                                                                           self.n, set(parent.knots(variable)))
                    # Choose the best candidate (if no candidate is an
                    # improvement on the linear term in terms of gcv, knot_idx
                    # is set to -1
                    if len(candidates_idx) > 0:
#                         candidates = np.array(predictor.x)[candidates_idx]

                        # Find the best knot location for this parent and
                        # variable combination
                        # Assemble the knot search data structure
                        constant = KnotSearchReadOnlyData(predictor, self.outcome)
                        search_data = KnotSearchData(constant, self.workings, q)

                        # Run knot search
                        knot, knot_idx, mse = knot_search(search_data, candidates, p, q,
                                                          self.m, len(candidates), self.n_outcomes,
                                                          self.verbose)
                        mse /= self.total_weight
                        knot_idx = candidates_idx[knot_idx]

                        # If the hinge function does not decrease the gcv then
                        # just keep the linear term (if variable_can_be_linear is True)
                        if variable_can_be_linear:
                            if missing_flag and not covered:
                                if gcv_factor_k_plus_4 * mse >= gcv_:
                                    mse = mse_
                                    knot_idx = -1
                            else:
                                if gcv_factor_k_plus_2 * mse >= gcv_:
                                    mse = mse_
                                    knot_idx = -1
                    else:
                        if variable_can_be_linear:
                            mse = mse_
                            knot_idx = -1
                        else:
                            # Do an orthonormal downdate and skip to the next
                            # iteration
                            if missing_flag and not covered:
                                self.orthonormal_downdate()
                                self.orthonormal_downdate()
                            self.orthonormal_downdate()
                            continue

                # Do an orthonormal downdate
                if missing_flag and not covered:
                    self.orthonormal_downdate()
                    self.orthonormal_downdate()
                self.orthonormal_downdate()

                # Update the choices
                if mse < mse_choice or first:
                    if first:
                        first = False
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    if self.use_fast is True:
                        parent_basis_content_choice = parent_basis_content
                    variable_choice = variable
                    dependent = linear_dependence
                    if missing_flag and not covered:
                        choice_needs_coverage = True
                    else:
                        choice_needs_coverage = False

                if self.use_fast is True:
                    if (mse_choice_cur_parent == -1) or \
                       (mse < mse_choice_cur_parent):
                        mse_choice_cur_parent = mse
                        variable_choice_cur_parent = variable
            if self.use_fast is True:
                if mse_choice_cur_parent != -1:
                    parent_basis_content.mse = mse_choice_cur_parent
                    parent_basis_content.v = variable_choice_cur_parent

        if self.use_fast is True:
            for content in content_to_be_repushed:
                heappush(self.fast_heap, content)


        if self.allow_subset == True:
            GCV_subset, optimal_var, optimal_val = self.subset_GCV()

            # Then return a tuple. No other return statement returns anything so you can tell if you subsetted
            if GCV_subset < mse_choice:

                # Longshot idea, what if I either make this a free function (not in self)
                # Or I put it all in here? Could that change pointer memory things
                # Cuz now its in the C code or whatever?
                forward_left, forward_right = self.create_optimal_forward_passers(optimal_val, optimal_var)
                #self.Node.GCV_pre_split = GCV_subset
                self.Node.split_var = optimal_var
                self.Node.split_val = optimal_val

                X_sub = np.copy(self.X)
                missing_sub = np.copy(self.missing)
                y_sub = np.copy(self.y)
                sample_weight_sub = np.copy(self.sample_weight)
                #Set left child node
                self.Node.left_child = Node()
                forward_left.set_node(self.Node.left_child)
                self.Node.left_child.parent = self.Node
                self.Node.left_child.set_forward_passer(forward_left)
                self.Node.left_child.X = X_sub[X_sub[:, int(optimal_var)] <= optimal_val, :]
                self.Node.left_child.y = y_sub[X_sub[:, int(optimal_var)] <= optimal_val, :]
                self.Node.left_child.missing = missing_sub[X_sub[:, int(optimal_var)] <= optimal_val, :]
                self.Node.left_child.sample_weight = sample_weight_sub[X_sub[:, int(optimal_var)] <= optimal_val, :]
                self.Node.left_child.n = self.Node.left_child.X.shape[0]

                #Set right child node
                self.Node.right_child = Node()
                forward_right.set_node(self.Node.right_child)
                self.Node.right_child.parent = self.Node
                self.Node.right_child.set_forward_passer(forward_right)
                self.Node.right_child.X = X_sub[X_sub[:, int(optimal_var)] > optimal_val, :]
                self.Node.right_child.y = y_sub[X_sub[:, int(optimal_var)] > optimal_val, :]
                self.Node.right_child.missing = missing_sub[X_sub[:, int(optimal_var)] > optimal_val, :]
                self.Node.right_child.sample_weight = sample_weight_sub[X_sub[:, int(optimal_var)] > optimal_val, :]
                self.Node.right_child.n = self.Node.right_child.X.shape[0]

                #Then finish
                return
            else:
                # Did not subset so update the GCV
                # Error: Wtf is gcv_??? Why did I put this??
                self.Node.GCV_ = gcv_
        # Make sure at least one candidate was checked
        if first:
            self.record[len(self.record) - 1].set_no_candidates(True)
            return

        # Add the new basis functions
        label = self.xlabels[variable_choice]
        if self.use_fast is True:
            parent_basis_content_choice.m = -np.inf
        if choice_needs_coverage:
            new_parent = parent_choice.get_coverage(variable_choice)
            if new_parent is None:
                new_basis_function = MissingnessBasisFunction(parent_choice, variable_choice,
                                               True, label)
                new_basis_function.apply(X, missing, B[:, len(self.basis)])
                self.orthonormal_update(B[:, len(self.basis)])
                if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                    content = FastHeapContent(idx=len(self.basis))
                    heappush(self.fast_heap, content)
                self.basis.append(new_basis_function)
                new_parent = new_basis_function

                new_basis_function = MissingnessBasisFunction(parent_choice, variable_choice,
                                               False, label)
                new_basis_function.apply(X, missing, B[:, len(self.basis)])
                self.orthonormal_update(B[:, len(self.basis)])
                if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                    content = FastHeapContent(idx=len(self.basis))
                    heappush(self.fast_heap, content)
                self.basis.append(new_basis_function)
        else:
            new_parent = parent_choice
        if knot_idx_choice != -1:
            # Add the new basis functions
            new_basis_function = HingeBasisFunction(new_parent,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     False, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, FastHeapContent(idx=len(self.basis)))
            self.basis.append(new_basis_function)

            new_basis_function = HingeBasisFunction(new_parent,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     True, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, content)
            self.basis.append(new_basis_function)

        elif not dependent and knot_idx_choice == -1:
            # In this case, only add the linear basis function (in addition to
            # covering missingness basis functions if needed)
            new_basis_function = LinearBasisFunction(new_parent, variable_choice, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, content)
            self.basis.append(new_basis_function)
        else:  # dependent and knot_idx_choice == -1
            # In this case there were no acceptable choices remaining, so end
            # the forward pass
            self.record[len(self.record) - 1].set_no_candidates(True)
            return

        # Compute the new mse, which is the result of the very stable
        # orthonormal updates and not the mse that comes directly from
        # the knot search
        cdef FLOAT_t final_mse = self.outcome.mse()

        # Update the build record
        self.record.append(ForwardPassIteration(parent_idx_choice,
                                                variable_choice,
                                                knot_idx_choice, final_mse,
                                                len(self.basis)))


    def set_node(ForwardPasser self, Node):
            self.Node = Node

    def create_optimal_forward_passers(ForwardPasser self, optimal_val, optimal_var):
        X_sub = np.copy(self.X)
        missing_sub = np.copy(self.missing)
        y_sub = np.copy(self.y)
        sample_weight_sub = np.copy(self.sample_weight)


        forward1_optimal = ForwardPasser(X_sub[X_sub[:, int(optimal_var)] <= optimal_val, :], missing_sub[X_sub[:, int(optimal_var)] <= optimal_val, :], 
                            y_sub[X_sub[:, int(optimal_var)] <= optimal_val, :], sample_weight_sub[X_sub[:, int(optimal_var)] <= optimal_val, :])
        forward2_optimal = ForwardPasser(X_sub[X_sub[:, int(optimal_var)] > optimal_val, :], missing_sub[X_sub[:, int(optimal_var)] > optimal_val, :], 
                            y_sub[X_sub[:, int(optimal_var)] > optimal_val, :], sample_weight_sub[X_sub[:, int(optimal_var)] > optimal_val, :])

        forward1_optimal.set_basis(self.get_basis())
        forward2_optimal.set_basis(self.get_basis())
            
        forward1_optimal.B = self.B[X_sub[:, int(optimal_var)] <= optimal_val, :]
        forward2_optimal.B = self.B[X_sub[:, int(optimal_var)] > optimal_val, :]


        if len(forward1_optimal.basis) > len(forward1_optimal.B[0]):
            raise ValueError("WOULD HAVE FAILED!")
        if len(forward2_optimal.basis) > len(forward2_optimal.B[0]):
            raise ValueError("WOULD HAVE FAILED!")
        '''
        file1 = open("/Users/WilliamJamesonPattie_/Desktop/Cool Text", "a")
        file1.write("About to orthonormal update\n")
        file1.write("Forward1 basis length: " + str(len(forward1_optimal.basis)) + "\n")
        file1.write("Forward1 max_terms: " + str(forward1_optimal.max_terms) + "\n")
        file1.write("Forward2 basis length: " + str(len(forward2_optimal.basis)) + "\n")
        file1.write("Forward2 max_terms: " + str(forward2_optimal.max_terms) + "\n")
        '''
        
        for i in range(0, len(forward1_optimal.basis)):
            try: 
                forward1_optimal.orthonormal_update(forward1_optimal.B[:, i])
            except Exception as e:
                print("Exception occurred:", e)
        for i in range(0, len(forward2_optimal.basis)):
            try: 
                forward2_optimal.orthonormal_update(forward2_optimal.B[:, i])
            except Exception as e:
                print("Exception occurred:", e)
        
        
        #forward1_optimal.record = copy.deepcopy(self.record)
        #forward2_optimal.record = copy.deepcopy(self.record)
        '''
        file1.write("Done with orthonormal update\n")
        file1.close()
        '''

        #forward1_optimal.orthonormal_update(forward1_optimal.B[:, len(self.basis)])
        #for i in range(0, len(forward2_optimal.basis) ): #+ 1):
            #if i > len(forward2_optimal.B[0])-1:
            #    print("WOULD HAVE GOTTEN OUT OF BOUNDS ERROR!!")
            #    print(i)
            #    print(len(forward2_optimal.basis))
            #    print(forward2_optimal.B)
            #else:

        # I don't understand why AT ALL but maybe this works??
        #forward2_optimal.orthonormal_update(forward2_optimal.B[:, len(self.basis)])

        return forward1_optimal, forward2_optimal


    ######################
    ##### NODE CLASS #####
    ######################
class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_var = None
        self.split_val = None
        self.forward_passer = None
        self.node_GCV = None
        self.Earth = None
        self.n = None

        self.parent = None

        self.X = None
        self.y = None
        self.sample_weight = None
        self.missing = None
    
    #def set_forward_passer_params(ForwardPasser self, cnp.ndarray[FLOAT_t, ndim=2] X,
                 #cnp.ndarray[BOOL_t, ndim=2] missing,
                 #cnp.ndarray[FLOAT_t, ndim=2] y,
                 #cnp.ndarray[FLOAT_t, ndim=2] sample_weight):
        #self.forward_passer = ForwardPasser(X, missing, y , sample_weight, Node = self)

    def set_forward_passer(self, forward_passer):
        self.forward_passer = forward_passer

    def run_forward_pass(self):

        self.forward_passer.run()
        if self.left_child == None and self.right_child == None:
            # Then running the forward passer created no children. Thus stop!
            return
        else:
            self.left_child.run_forward_pass()
            self.right_child.run_forward_pass()
