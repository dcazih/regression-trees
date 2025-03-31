# Regression Trees Midterm

import numpy as np
import time
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

###################################### PART 1 #################################
# Class to represent a node in the regression tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, depth=0):
        # feature to split at
        self.feature = feature
        # threshold for the split
        self.threshold = threshold
        self.left = left
        self.right = right
        # mean target value
        self.value = value
        self.depth = depth

# Class for a regression tree    
class RegressionTree:
    def __init__(self, X, y, max_depth=None, min_leaf_size=None, control='depth'):
        # Max depth of tree
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.control = control
        # build tree from root
        self.root = self._build_tree(np.array(X), np.array(y), depth=0)

    # Function to calculate the sum of squared errors for a feature
    def _sum_squared_errors(self, y):
        # If no samples
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        # Sum of squared differences from mean
        sum_squared_errors = np.sum((y - mean) ** 2)
        return sum_squared_errors
    
    # Function to find the best feature to split at (Step 2 in Notes Algorithm)
    def _best_split(self, X, y):
        best_sum_squared_errors = self._sum_squared_errors(y)
        best_feature = None
        best_threshold = None

        # Loop through all features
        for feature in range(X.shape[1]):
            # Sort values by feature
            sorted = X[:, feature].argsort()
            X_sorted = X[sorted]
            y_sorted = y[sorted]

            # Try all features to split at
            for i in range(1, len(y_sorted)):
                # If values don't change just skip it
                if X_sorted[i, feature] == X_sorted[i - 1, feature]:
                    continue
                
                # Comput threshold between the two points
                threshold = (X_sorted[i, feature] + X_sorted[i - 1, feature]) / 2
                left_y = y_sorted[:i]
                right_y = y_sorted[i:]
                # Total error after split
                sum_squared_errors_split = self._sum_squared_errors(left_y) + self._sum_squared_errors(right_y)
                # Update if this is the new best split feature
                if sum_squared_errors_split < best_sum_squared_errors:
                    best_sum_squared_errors = sum_squared_errors_split
                    best_feature = feature
                    best_threshold = threshold
        # best split
        return best_feature, best_threshold
    
    # Function for building the regression tree (step 1 in Notes Algorith)
    def _build_tree(self, X, y, depth):
        # checks if all points have same variablees
        if len(set(map(tuple, X))) == 1:
            return Node(value=np.mean(y), depth=depth)
        # Check if max depth is reached
        if self.control == 'depth' and self.max_depth is not None and depth >= self.max_depth:
            return Node(value=np.mean(y), depth=depth)
        # Check if node size is below leaf size limit
        if self.control == 'leaf_size' and self.min_leaf_size is not None and len(y) <= self.min_leaf_size:
            return Node(value=np.mean(y), depth=depth)
        # Find best split for current node
        feature, threshold = self._best_split(X, y)
        # No error reduction
        if feature is None:
            return Node(value=np.mean(y), depth=depth)
        # Split data based on threshold
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        # Don't split into empty children
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return Node(value=np.mean(y), depth=depth)
        
        # Build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # return node with split
        return Node(feature, threshold, left_child, right_child, depth=depth)
    
    # Function to traverse tree for prediction and path (Step 3 in Notes Algorithm)
    def _traverse(self, node, x):
        # Check if node is a leaf
        if node.value is not None:
            return node.value
        
        # Decide left or right traversal based on feature
        if x[node.feature] <= node.threshold:
            return self._traverse(node.left, x)
        else:
            return self._traverse(node.right, x)

    # Function to predict value for input
    def predict(self, x):
        return self._traverse(self.root, np.array(x))
    
    # Function to display the decision path of agiven input value
    def decision_path(self, x):
        path = []
        node = self.root
        # While there are still nodes
        while node.value is None:
            feature = node.feature
            threshold = node.threshold
            # Print the feature that is selected
            if x[feature] <= threshold:
                path.append(f"X[{feature}] <= {threshold}")
                node = node.left
            else:
                path.append(f"X[{feature}] > {threshold}")
                node = node.right
        return path
    
###############################################################################
###################################### PART 2 #################################

# Part a
# 100 uniformly distributed samples on the domain x [-3, 3]
X = np.linspace(-3, 3, 100).reshape(-1, 1)
# Continous y = 0.8 sin(x - 1) function
y = 0.8 * np.sin(X - 1).ravel()
# 80 20 split with Scikit-Learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# keep track of time cost of building the tree
start_time1 = time.time()
tree = RegressionTree(X_train, y_train)
end_time1 = time.time()
# predict on the test set and keep track of the error
y_pred1 = [tree.predict(x) for x in X_test]
error1 = mean_squared_error(y_test, y_pred1)
build_time1 = end_time1 - start_time1
# helper function for testing the height of the regression tree
def get_tree_height(node):
    # Basic recursion function to count tree levels
    if node is None or node.value is not None:
        return 0
    return 1 + max(get_tree_height(node.left), get_tree_height(node.right))
# call the helper, keep track of height
tree_height1 = get_tree_height(tree.root)
# Print Results
print("Part a - Limitless Tree")
print("Tree Height:", tree_height1)
print("Test Error:", error1)
print("Time Cost:", build_time1)

# Part b
# Test another tree that is half the height
half_height = int(tree_height1 * 0.5)
# Do the same as part a with this new tree
# keep track of time cost of building the tree
start_time2 = time.time()
tree_half = RegressionTree(X_train, y_train, max_depth=half_height)
end_time2 = time.time()
# predict on the test set and keep track of the error
y_pred2 = [tree_half.predict(x) for x in X_test]
error2 = mean_squared_error(y_test, y_pred2)
build_time2 = end_time2 - start_time2
# call the helper, keep track of height
tree_height2 = get_tree_height(tree_half.root)
# Print Results
print("Part b - Half height tree")
print("Tree Height:", tree_height2)
print("Test Error:", error2)
print("Time Cost:", build_time2)

# Part c
# Leaf size limits
leaf_sizes = [2, 4, 8]
# Loop through leaf sizes to test each limit
for size in leaf_sizes:
    start_time3 = time.time()
    tree_leaf = RegressionTree(X_train, y_train, min_leaf_size=size, control='leaf_size')
    end_time3 = time.time()
    # predict on the test set and keep track of the error
    y_pred3 = [tree_leaf.predict(x) for x in X_test]
    error3 = mean_squared_error(y_test, y_pred3)
    build_time3 = end_time3 - start_time3
    # call the helper, keep track of height
    tree_height3 = get_tree_height(tree_leaf.root)
    # Print Results
    print("Part c - Leaf Size Limit ", size)
    print("Tree Height:", tree_height3)
    print("Test Error:", error3)
    print("Time Cost:", build_time3)

###############################################################################
###################################### PART 3 #################################
## Generate Data
# Sample number
N = 500
# Inputs
xk = np.random.uniform(0, 100, N)
vk = np.full(N, 10.0)
# Shape of (N,2)
X = np.stack((xk, vk), axis=1)
# Outputs
xk_plus1 = xk + 0.1 * vk
vk_plus1 = np.full(N, 10.0)
# Use Regression Tree to make predictions
tree_x = RegressionTree(X, xk_plus1)  
tree_v = RegressionTree(X, vk_plus1) 
# Helper function to predict the next state on the tree
def predict_next_state(xk, vk):
    state = [xk, vk]
    x_next = tree_x.predict(state)
    v_next = tree_v.predict(state)
    return [x_next, v_next]

# Print some tester results to see how good the predictions are
# Can change values to test different states
print("Current state: (99, 10)")
predicted_state = predict_next_state(99, 10)
print("Predicted next state:", predicted_state)
next_state = [99 + 0.1 * 10, 10]
print("True next state:     ", next_state)
error = np.array(predicted_state) - np.array(next_state)
print("Error of Prediction: ", error)





###############################################################################
###################################### PART 4 #################################


#### PART A
print("\n\nPART 4")
print("\nCase Study 1")
### Generate Sample Data for Dynamic System
x1 = np.random.uniform(-5, 5, N)
x2 = np.random.uniform(-5, 5, N)

x1_next = 0.9 * x1 - 0.2 * x2
x2_next = 0.2 * x1 + 0.9 * x2

X = np.column_stack((x1, x2))     
Y = np.column_stack((x1_next, x2_next))  

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


### Create Regression Trees
# reg1: learns the mapping (x(1),x(2))→ x′(1)(x(1), x(2))→ x′(1)
# reg2: learns the mapping (x(1),x(2))→ x′(2)(x(1), x(2))→ x′(2)
tree_depth = 15
leaf_sizes = 4
reg_tree_x1 = RegressionTree (X_train, Y_train[:, 0], tree_depth, leaf_sizes)
reg_tree_x2 = RegressionTree(X_train, Y_train[:, 1], tree_depth, leaf_sizes)


### Perform Testing
# Predict each dimension on test samples
y1_pred = np.array([reg_tree_x1.predict(x) for x in X_test])
y2_pred = np.array([reg_tree_x2.predict(x) for x in X_test])

# take mean squared error for each dimension and print
mse_x1 = mean_squared_error(Y_test[:, 0], y1_pred)
mse_x2 = mean_squared_error(Y_test[:, 1], y2_pred)
print(f"MSE of x'(1): {mse_x1:.4f}")
print(f"MSE of x'(2): {mse_x2:.4f}")

### Perform  Simulation
# State Predictors: Model and True Predictors
def predict_next_state(x1, x2):
    return reg_tree_x1.predict([x1, x2]), reg_tree_x2.predict([x1, x2])

def true_next_state(x1, x2):
    return (0.9 * x1 - 0.2 * x2, 0.2 * x1 + 0.9 * x2)

T = 20 # steps T = 0, 1, ..., 20

x1_0, x2_0 = (0.5, 1.5) # set initial conditions

# Set initial states
predicted_trajectory = np.zeros((T + 1, 2))
predicted_trajectory[0] = [x1_0, x2_0]

true_trajectory = np.zeros((T+1, 2))
true_trajectory[0] = [x1_0, x2_0]

# Run Simulation
for t in range(T):
    x1_pred, x2_pred = predict_next_state(predicted_trajectory[t,0], predicted_trajectory[t,1])
    predicted_trajectory[t+1] = [x1_pred, x2_pred]

    x1_true, x2_true = true_next_state(true_trajectory[t,0],
                                       true_trajectory[t,1])
    true_trajectory[t+1] = [x1_true, x2_true]

# Plot  final states 
print("Final predicted state:", predicted_trajectory[-1])
print("Final true state:     ", true_trajectory[-1])
plt.figure(figsize=(6,6))
plt.plot(true_trajectory[:,0], true_trajectory[:,1], 'o--', label='True Trajectory')
plt.plot(predicted_trajectory[:,0], predicted_trajectory[:,1], 's--', label='Predicted Trajectory')
plt.xlabel('x(1)')
plt.ylabel('x(2)')
plt.title('Phase Plot of True vs. Predicted Trajectory')
plt.legend()
plt.grid(True)
plt.show()


#### PART B
### Generate Sample Data for Program State Prediction
N = 1000
x_vals = np.random.uniform(-3, 3, N)
z_vals = np.random.uniform(0, 15, N)

# Simulate the program to get next states
def oneStep_in_program(x, z):
    new_x = 0 if x > 1 else x + 0.2
    new_z = z + new_x
    return new_x, new_z

# Generate next states
next_states = np.array([oneStep_in_program(x, z) for x, z in zip(x_vals, z_vals)])
X = np.column_stack((x_vals, z_vals))
Y = next_states

# Split data
X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(X, Y, test_size=0.2)


### Create Regression Trees
tree_depth = 15
leaf_sizes = 4
reg_tree_x = RegressionTree(X_train_p, Y_train_p[:, 0], tree_depth, leaf_sizes)
reg_tree_z = RegressionTree(X_train_p, Y_train_p[:, 1], tree_depth, leaf_sizes)


### Run Simulation
# Predict each dimension on test samples x and z
def predict_program_state(x, z):
    return reg_tree_x.predict([x, z]), reg_tree_z.predict([x, z])

# Initialize parameters
T = 20
x0, z0 = 2.0, 0.0
predicted_trajectory = np.zeros((T+1, 2))
predicted_trajectory[0] = [x0, z0]
true_trajectory = np.zeros((T+1, 2))
true_trajectory[0] = [x0, z0]

# Run simulation
for t in range(T):
    x_pred, z_pred = predict_program_state(predicted_trajectory[t,0], predicted_trajectory[t,1])
    predicted_trajectory[t+1] = [x_pred, z_pred] 

    x_true, z_true = oneStep_in_program(true_trajectory[t,0], true_trajectory[t,1])
    true_trajectory[t+1] = [x_true, z_true]

# Plot results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)# x trajectory
plt.plot(range(T+1), true_trajectory[:,0], 'o-', label='True x')
plt.plot(range(T+1), predicted_trajectory[:,0], 's--', label='Predicted x')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('x Variable Trajectory')
plt.legend() 
plt.grid(True)
plt.subplot(1,2,2)# z trajectory
plt.plot(range(T+1), true_trajectory[:,1], 'o-', label='True z')
plt.plot(range(T+1), predicted_trajectory[:,1], 's--', label='Predicted z')
plt.xlabel('Iteration')
plt.ylabel('z value')
plt.title('z Variable Trajectory')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#### PART C

def tune_hyperparameters(X_train, Y_train, X_test, Y_test):
    print("\n=========================Tuning Hyperparameters=========================")
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_leaf_size': [1, 2, 4, 8, 16]
    }
    results = []

    for depth in param_grid['max_depth']:
        for leaf_size in param_grid['min_leaf_size']:
            start_time = time.time()
            
            # Train models in both dimensions
            model_x = RegressionTree(X_train, Y_train[:,0], max_depth=depth, 
                                    min_leaf_size=leaf_size)
            model_z = RegressionTree(X_train, Y_train[:,1], max_depth=depth,
                                    min_leaf_size=leaf_size)
            
            build_time = time.time() - start_time
            
            # Make predictions
            y_pred_x = [model_x.predict(x) for x in X_test]
            y_pred_z = [model_z.predict(x) for x in X_test]
            
            # find MSE
            mse_x = mean_squared_error(Y_test[:,0], y_pred_x)
            mse_z = mean_squared_error(Y_test[:,1], y_pred_z)
            avg_mse = (mse_x + mse_z)/2
            
            print(f"Depth: {str(depth).ljust(4)}, Leaf: {str(leaf_size).ljust(2)} "f"| MSE: {avg_mse:.4f} (x: {mse_x:.4f}, z: {mse_z:.4f}) "f"| Time: {build_time:.2f}s")
    
    # Show best parameters
    print("\nBest Parameters:")
    print(f"Depth: {depth}, Leaf Size: {leaf_size}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Build Time: {build_time:.2f}s")
 

# Run tuning for both systems
print("\n Dyamical System Prediction")
tune_hyperparameters(X_train, Y_train, X_test, Y_test)

print("\n Program State Prediction")
tune_hyperparameters(X_train_p, Y_train_p, X_test_p, Y_test_p)