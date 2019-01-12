#from sklearn.datasets import load_breast_cancer
# def replace_function(data_set, column, change_matrix):
#     y = data_set[column]
#     for record in change_matrix:
#         y = y.replace(record[0], record[1])
    
#     return y.values

# #change_matrix = [["malignant", 0], ["benign", 1]]
# change_matrix = [[0, "malignant"], [1, "benign"]]


# data = load_breast_cancer()
# print(data.target_names)
# print(data.target_names)
# Y = pd.DataFrame(data.target)
# Y = replace_function(Y, 0, change_matrix)
# #print(Y)

# X = pd.DataFrame(data.data)
# X['30'] = Y
# print(X)
# X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 0)

# X_train = pd.DataFrame(X_train)
# Y_train = replace_function(pd.DataFrame(Y_train), 0, change_matrix)
# X_train['30'] = Y_train
# X_train.to_csv('traindata.csv', sep=',',index=False, header=False)

# X_test = pd.DataFrame(X_test)
# X_test.to_csv('x_testdata.csv',sep=",", index=False, header=False)
# Y_test = replace_function(pd.DataFrame(Y_test), 0, change_matrix)
# Y_test = pd.DataFrame(Y_test)
# Y_test.to_csv('y_testdata.csv', sep=",", index=False, header=False)