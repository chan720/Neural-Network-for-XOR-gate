from sklearn.neural_network import MLPClassifier
X = [
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1]
]

Y = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0]

clf=MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,5,2),max_iter=1000)
clf.fit(X,Y)
result=clf.predict(X)
print("Result: ", result)
print("Performence: ",clf.score(X,Y)*100,"%")

# print("\nWeights Between Input Layerand first Hidden")
# print(clf.coefs_[0])
# print("\nWeights Between Input Layerand second Hidden")
# print(clf.coefs_[1])
# print("\nWeights Between Input Layerand third Hidden")
# print(clf.coefs_[2])
# print ("Bias for Hidden Layer")
