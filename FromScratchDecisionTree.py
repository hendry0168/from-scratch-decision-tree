"""

A standalone Python script that:
1. Connects to SSMS to load HR data.
2. Trains/tests a custom Decision Tree classifier.
3. Validates performance and outputs metrics.

python FromScratchDecisionTree.py
"""

# import pyodbc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class FromScratchDecisionTree:

    def __init__(self, max_depth = 6, min_gain=0.0, depth = 1):
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.depth = depth
        self.left = None
        self.right = None
    
    def fit(self, data, target):
        if self.depth <= self.max_depth: print(f"processing at Depth: {self.depth}")
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)
        if self.depth <= self.max_depth:
            self.__validate_data()
            self.impurity_score = self.__calculate_impurity_score(self.data[self.target])
            self.criteria, self.split_feature, self.information_gain = self.__find_best_split()
            if self.criteria is not None and self.information_gain > 0: self.__create_branches()
        else: 
            print("Stopping splitting as Max depth reached")
    
    def __create_branches(self):
        if self.information_gain > self.min_gain:
            self.left = FromScratchDecisionTree(max_depth = self.max_depth, 
                                    depth = self.depth + 1)
            self.right = FromScratchDecisionTree(max_depth = self.max_depth, 
                                    depth = self.depth + 1)
            left_rows = self.data[self.data[self.split_feature] <= self.criteria] 
            right_rows = self.data[self.data[self.split_feature] > self.criteria] 
            self.left.fit(data = left_rows, target = self.target)
            self.right.fit(data = right_rows, target = self.target)
        else:
            print(f"Stopping split at depth {self.depth}: information gain {self.information_gain:.4f} <= min_gain {self.min_gain:.4f}")
                 
    def __calculate_impurity_score(self, data):
       if data is None or data.empty: return 0
       p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist() 
       return p_i * (1 - p_i) * 2
    
    def __find_best_split(self):
        print(f"\n=== Depth {self.depth} ===")
        print(f"Current node size: {len(self.data)}")
        print(f"Class distribution: {self.data[self.target].value_counts().to_dict()}")
        best_split = {}
        for col in self.independent:
            information_gain, split = self.__find_best_split_for_column(col)
            if split is None: continue
            print(f"  Column {col}: best split at {split} with IG={information_gain:.4f}")
            if not best_split or best_split["information_gain"] < information_gain:
                best_split = {"split": split, "col": col, "information_gain": information_gain}
        
        # Debugging Prints help you see where splits are failing
        if best_split:
            print(f"✅ BEST SPLIT: {best_split['col']} at {best_split['split']:.2f} (IG: {best_split['information_gain']:.4f})")
        else:
            print("❌ NO VALID SPLIT FOUND")
            
        return best_split.get("split"), best_split.get("col"), best_split.get("information_gain")

    # def __find_best_split_for_column(self, col):
    #     x = self.data[col]
    #     unique_values = x.unique()
    #     if len(unique_values) == 1: return None, None
    #     information_gain = None
    #     split = None
    #     for val in unique_values:
    #         left = x <= val
    #         right = x > val
    #         left_data = self.data[left]
    #         right_data = self.data[right]
    #         left_impurity = self.__calculate_impurity_score(left_data[self.target])
    #         right_impurity = self.__calculate_impurity_score(right_data[self.target])
    #         score = self.__calculate_information_gain(left_count = len(left_data),
    #                                                   left_impurity = left_impurity,
    #                                                   right_count = len(right_data),
    #                                                   right_impurity = right_impurity)
    #         if information_gain is None or score > information_gain: 
    #             information_gain = score 
    #             split = val
    #     return information_gain, split

    def __find_best_split_for_column(self, col):
        x = self.data[col]
        unique_values = x.unique()
        if len(unique_values) == 2: 
            val = 0.5  # Split between 0 and 1
            left = self.data[x <= val]
            right = self.data[x > val]
            
            if not left.empty and not right.empty:
                gain = self.__calculate_information_gain(
                    len(left), self.__calculate_impurity_score(left[self.target]),
                    len(right), self.__calculate_impurity_score(right[self.target])
                )
                return gain, val
            return None, None
        else:
            # Original numeric split logic
            best_gain = -1
            best_split = None
            for val in unique_values:
                left = x <= val
                right = x > val
                left_data = self.data[left]
                right_data = self.data[right]
                left_impurity = self.__calculate_impurity_score(left_data[self.target])
                right_impurity = self.__calculate_impurity_score(right_data[self.target])
                current_gain = self.__calculate_information_gain(left_count = len(left_data),
                                                        left_impurity = left_impurity,
                                                        right_count = len(right_data),
                                                        right_impurity = right_impurity)
                if best_gain is None or current_gain > best_gain: 
                    best_gain = current_gain 
                    split = val
        return best_gain, split
    
    def __calculate_information_gain(self, left_count, left_impurity, right_count, right_impurity):
        return self.impurity_score - ((left_count/len(self.data)) * left_impurity + \
                                      (right_count/len(self.data)) * right_impurity)

    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for _, row in data.iterrows()])

    def __validate_data(self):
        non_numeric_columns = self.data[self.independent].select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
        if(len(set(self.independent).intersection(set(non_numeric_columns))) != 0):
            raise RuntimeError("Not all columns are numeric")
        
        self.data[self.target] = self.data[self.target].astype("category")
        if(len(self.data[self.target].cat.categories) != 2):
            raise RuntimeError("Implementation is only for Binary Classification")

    def __flow_data_thru_tree(self, row):
        if self.is_leaf_node: return self.probability
        tree = self.left if row[self.split_feature] <= self.criteria else self.right
        return tree.__flow_data_thru_tree(row)
        
    @property
    def is_leaf_node(self): return self.left is None

    @property
    def probability(self): 
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()

def evaluate_model(tree, X_test, y_test):
    """Calculate and print performance metrics"""
    y_pred = [1 if prob[1] > 0.5 else 0 for prob in tree.predict(X_test)]
    
    # print("\nModel Performance:")
    print(f"Custom Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Custom Tree Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # # Plot feature importance
    # if hasattr(tree, 'split_feature'):
    #     plt.barh([tree.split_feature], [tree.information_gain])
    #     plt.title("Feature Importance")
    #     plt.xlabel("Information Gain")
    #     plt.show()
    