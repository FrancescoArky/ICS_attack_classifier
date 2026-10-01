from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def decision_tree_classifier(features, X_train_final, y_train, X_val_final, y_val):
    model = DecisionTreeClassifier(random_state=42)
    
    model.fit(X_train_final, y_train)
    
    y_pred = model.predict(X_val_final)
    
    accuracy = accuracy_score(y_val, y_pred)
    
    print("Accuracy:", accuracy)
    
    print(classification_report(y_val, y_pred))
    
    cm = confusion_matrix(y_val, y_pred)
    
    print(cm)
    
    importances = model.feature_importances_
    
    for feature, importance in zip(features, importances):
        print(feature, importance)