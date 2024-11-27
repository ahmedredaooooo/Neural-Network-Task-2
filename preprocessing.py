import pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess():
    selected_class = ['A','B','C']
    data = pd.read_csv("DataSet/birds.csv")



    # fill gender col with most frequent
    data['gender'] = data['gender'].fillna(data['gender'].mode()[0])

    # Encode gender col
    le_gender = LabelEncoder()
    data['gender'] = le_gender.fit_transform(data['gender'])


    # Scale numeric cols from -1 to 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data[['body_mass', 'beak_length', 'beak_depth', 'fin_length']] = np.round(
        scaler.fit_transform(data[['body_mass', 'beak_length', 'beak_depth', 'fin_length']]), 4
    )

    selected_features = ['gender'  ,'body_mass'  ,'beak_length'  ,'beak_depth'  ,'fin_length']
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    for selected_class in ['A','B','C']:
        class_data = data[data['bird category'] == selected_class].sample(frac=1, random_state=42).reset_index(
            drop=True)

        # Select only the two specified features and drop 'bird category' for x and retain for y
        x_class = class_data[selected_features]
        y_class = class_data['bird category']

        # split each class to 30 record for train and 20 for test
        x_train_class = x_class.iloc[:30]
        x_test_class = x_class.iloc[30:50]
        y_train_class = y_class.iloc[:30]
        y_test_class = y_class.iloc[30:50]

        x_train_list.append(x_train_class)
        x_test_list.append(x_test_class)
        y_train_list.append(y_train_class)
        y_test_list.append(y_test_class)


    x_train = pd.concat(x_train_list).reset_index(drop=True)
    x_test = pd.concat(x_test_list).reset_index(drop=True)
    y_train = pd.concat(y_train_list).reset_index(drop=True)
    y_test = pd.concat(y_test_list).reset_index(drop=True)



    print("\nNumber of rows for each class in y_train:")
    print(y_train.value_counts())
    print("\nNumber of rows for each class in y_test:")
    print(y_test.value_counts())


    print("\nFinal concatenated training and testing sets:")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")



    x_train, y_train = (x_train.sample(frac=1, random_state=42).reset_index(drop=True),
                        y_train.sample(frac=1, random_state=42).reset_index(drop=True))
    x_test, y_test = (x_test.sample(frac=1, random_state=42).reset_index(drop=True),
                      y_test.sample(frac=1, random_state=42).reset_index(drop=True))

    # encode them cause they are A | B | C with same mapping
    unique_classes = y_train.unique()
    class_mapping = {unique_classes[0]: 0, unique_classes[1]: 1, unique_classes[2]:2 }

    # Apply the mapping to y_train and y_test
    y_train = y_train.replace(class_mapping)
    y_test = y_test.replace(class_mapping)

    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test

