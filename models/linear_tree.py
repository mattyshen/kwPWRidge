import numpy as np

from sklearn.metrics import mean_squared_error

class LinearNode():
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None, threshold=None, value=None, 
                true_branch=None, false_branch=None, beta=None, num_samples=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree
        self.beta = beta                    # Beta estimate
        self.num_samples = num_samples      # Number of samples contained in Node


# Super class of RegressionTree and ClassificationTree
class LinearTree(object):
    """Super class of RegressionTree and ClassificationTree.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    def __init__(self, min_samples_split=2, min_r2_gain=0, seed=2023,
                 alpha = 1, lam = 1, depth_shrink = 0, max_depth=float("inf"), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_r2_gain = min_r2_gain
        self.max_depth = max_depth
        self.loss = loss
        self.seed = seed 
        self.alpha = alpha
        self.lam = lam
        self.depth_shrink = depth_shrink
        self.feature_indices = None
        self.sample_indices = None
    
    def solve(self, X, y):
        print(X.shape,y.shape)
        return np.linalg.inv(X.T@X + self.alpha * np.eye(X.shape[1])) @ X.T @ y

    def gauss_update(self, Xy, beta, V_t = None, add=True, R_init=None):
        if R_init is not None and len(beta) == 0:
            Xy = R_init.copy()
        y = Xy[:, Xy.shape[1]-1].reshape(-1, 1)
        X = Xy[:, :Xy.shape[1]-1]
        if len(beta) == 0:
            V = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1]))
            beta = V @ X.T @ y
        else:
            V = V_t.copy()
            if add:
                for i in range(X.shape[0]):
                    x = X[i, :].reshape(-1, 1)

                    V -= V@x@x.T@V / (1 + (x.T@V@x).item())
                    beta += V@x * (y[i] - (x.T@beta).item())
            else:
                for i in range(X.shape[0]):
                    x = X[i, :].reshape(-1, 1)
                    
                    h = (x.T@V@x).item()
                    beta -= V @ x * (y[i] - (x.T@beta).item()) / (1-h)
                    V += V@x@x.T@V / (1 - h)
        return beta.copy(), V.copy()

    def rss(self, X, y, beta):
        return len(y) * mean_squared_error(y, X @ beta)
        #return np.sum((y.reshape(-1, ) - (X @ beta).reshape(-1, ))**2)

    def fit(self, X, y, loss=None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis = 1)
        
        cur_beta = self.solve(X, y)
        cur_rss = self.rss(X, y, cur_beta)
        
        self.root = self._build_tree(X, y, cur_beta, cur_rss)
        self._shrink_tree()
        self.loss=None

    def _build_tree(self, X, y, cur_beta, cur_rss, current_depth=0):
        n_samples, n_features = X.shape

        if cur_rss == 0:
            return LinearNode(value=np.mean(y), beta=cur_beta, num_samples=len(y))
        
        if np.var(y) == 0:
            mean_beta = np.zeros(n_features).reshape(-1, 1)
            mean_beta[0] = np.mean(y)
            return LinearNode(value=np.mean(y), beta=mean_beta, num_samples=len(y))

        smallest_rss = cur_rss
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        assert len(y.shape) == 2 and y.shape[1] == 1, "y needs to be of shape (n, 1). please fix."

        Xy = np.concatenate((X, y), axis=1)
    
        if n_samples >= self.min_samples_split and current_depth < self.max_depth:
            for feature_i in range(1, n_features):
                Xy = Xy[np.argsort(Xy[:,feature_i])]

                sort_unq_vals = np.sort(np.unique(Xy[:, feature_i]))
                sort_unq_vals = sort_unq_vals[:-1] + 0.5*np.diff(sort_unq_vals)

                indexes = np.array([np.where(Xy[:, feature_i] < val)[0][-1] for val in sort_unq_vals]) + 1

                L_beta, R_beta = np.array([]), np.array([])
                L_V, R_V = None, None
            
                prev_i = 0
                for i in indexes:
                    
                    Xy1 = Xy[:i, :]
                    Xy2 = Xy[i:, :]

                    L_beta, L_V = self.gauss_update(Xy[prev_i:i, :], L_beta.copy(), L_V, True)
                    R_beta, R_V = self.gauss_update(Xy[prev_i:i, :], R_beta.copy(), R_V, False, Xy2)

                    L_rss = self.rss(Xy1[:,:n_features], Xy1[:,n_features:], L_beta)
                    R_rss = self.rss(Xy2[:,:n_features], Xy2[:,n_features:], R_beta)

                    if L_rss + R_rss < smallest_rss:

                        #print(f'new split accepted! feature: {feature_i}, old best rss: {smallest_rss}, new best rss: {cur_rss}')
                        smallest_rss = (L_rss + R_rss).copy()
                        best_criteria = {"feature_i": feature_i, "threshold": sort_unq_vals[np.argmax(indexes == i)]}
                        L_rss_best = L_rss.copy()
                        R_rss_best = R_rss.copy()
                        L_best = L_beta.copy()
                        R_best = R_beta.copy()
                        best_sets = {
                            "leftX": Xy1[:, :n_features],   # X of left subtree
                            "lefty": Xy1[:, n_features:],   # y of left subtree
                            "rightX": Xy2[:, :n_features],  # X of right subtree
                            "righty": Xy2[:, n_features:],   # y of right subtree
                            "leftBeta": L_best,
                            "rightBeta": R_best,
                            "leftRSS": L_rss_best,
                            "rightRSS": R_rss_best
                            }

                    prev_i = i
                    
            r2_gain = (cur_rss - smallest_rss)/np.sum((y-np.mean(y))**2)
            
            # print(f'r2 gain: {r2_gain}')
            # print(f'best split point: {best_criteria["threshold"]}')
            # print(f'current depth: {current_depth}')
            
            if r2_gain > self.min_r2_gain:
                #print(r2_gain)
                true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], best_sets['leftBeta'], best_sets['leftRSS'], current_depth + 1)
                false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], best_sets['rightBeta'], best_sets['rightRSS'], current_depth + 1)

                return LinearNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"], value=np.mean(y), true_branch=true_branch, false_branch=false_branch, beta=cur_beta,num_samples=len(y))
            else:
                return LinearNode(value=np.mean(y), beta=cur_beta, num_samples=len(y))
        else:
            return LinearNode(value=np.mean(y), beta=cur_beta, num_samples=len(y))


    def predict_value(self, x, tree=None, linear_honesty = False):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """
        if tree is None:
            tree = self.root

        # If we have no nodes (LinearNodes) below, we are at a leaf node
        if tree.true_branch is None:
            # If linear honesty
            if linear_honesty:
                return tree.value
            # if true linear tree
            else:
                y_hat = (x.reshape(1, -1) @ tree.beta)
                return y_hat.item()

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]
        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value < tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        
        # Test subtree
        return self.predict_value(x, tree = branch, linear_honesty = linear_honesty)

    def predict(self, X, linear_honesty = False):
        """ Classify samples one by one and return the set of labels """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis = 1)
        #print('individ predicts')
        y_pred = [self.predict_value(sample, linear_honesty = linear_honesty) for sample in X]
        return y_pred

    def _shrink_tree(self, tree=None, local=None, parent_val=None, parent_beta=None, parent_num=None, cum_sum=0, cum_beta = np.array([]), depth=None):
        
        if tree is None and len(cum_beta) == 0 and depth is None:
            tree = self.root
            cum_beta = np.zeros_like(tree.beta)
            depth = 0
        is_leaf = tree.true_branch is None
        n_samples = tree.num_samples
        val = tree.value.copy()
        beta = tree.beta.copy()
        
        if (parent_val is None and parent_beta is None and parent_num is None) or (depth <= self.depth_shrink):
            cum_sum = val
            cum_beta = beta
        else:
            denom = parent_num
            if local == "r2":
                denom *= tree.r2
            if local == "oobr2":
                denom *= tree.oobr2
            val_new = (val - parent_val)/(1 + self.lam / denom)
            beta_new = (beta - parent_beta)/(1 + self.lam / denom) #beta, beta_new should be px1
            cum_sum += val_new
            cum_beta += beta_new

        tree.value = cum_sum
        tree.beta = cum_beta

        if not is_leaf:
            self._shrink_tree(tree=tree.true_branch, local=local, parent_val=val, parent_beta=beta, parent_num=n_samples, cum_sum=cum_sum.copy(), cum_beta=cum_beta.copy(), depth=depth+1)
            self._shrink_tree(tree=tree.false_branch, local=local, parent_val=val, parent_beta=beta, parent_num=n_samples, cum_sum=cum_sum.copy(), cum_beta=cum_beta.copy(), depth=depth+1)
        return tree

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.true_branch is None:
            #print(f'v: {tree.value}, b: {tree.beta}')
            print(f'v: {tree.value}')
        # Go deeper down the tree
        else:
            # Print test
            #print ("f: %s, t: %s, b: %s? " % (tree.feature_i, tree.threshold, tree.beta))
            print ("f: %s, t: %s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)
