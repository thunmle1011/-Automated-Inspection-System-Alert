# from pyod.models.knn import KNN
from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from apyori import apriori
import psycopg2
import yaml
# from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import IsolationForest
from apyori import apriori
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib
matplotlib.use('TkAgg')  # Use the 'agg' backend
import matplotlib.pyplot as plt


def read_data_from_database():
    # Create Connection
    with open('config.yml', 'r') as file:
        data = yaml.safe_load(file)
    con = psycopg2.connect(database=data["db"],
                           user=data["usr"],
                           host=data["hst"],
                           password=data["pwd"],
                           port=data["prt"])


    # Load datasets to dataframe
    # Create a cursor object
    cur = con.cursor()
    # Execute query
    cur.execute("""
                    SELECT * FROM export."AS400_tblProceso";
                    """)
    df_proceso = pd.DataFrame(cur.fetchall())
    df_proceso.columns = [desc[0] for desc in cur.description]
    df_proceso.head()

    cur.execute("""
                    SELECT * FROM export."CustomerClaims";
                    """)
    df_customer_claims = pd.DataFrame(cur.fetchall())
    df_customer_claims.columns = [desc[0] for desc in cur.description]
    df_customer_claims.head()

    cur.execute("""
                    SELECT * FROM export."NAS_catFinishCodes";
                    """)
    df_finish_codes = pd.DataFrame(cur.fetchall())
    df_finish_codes.columns = [desc[0] for desc in cur.description]
    df_finish_codes.head()

    cur.execute("""
                    SELECT * FROM export."NAS_catLines";
                    """)
    df_lines = pd.DataFrame(cur.fetchall())
    df_lines.columns = [desc[0] for desc in cur.description]
    df_lines.head()

    cur.execute("""
                    SELECT * FROM export."NAS_catSteelGrades";
                    """)
    df_steel_grades = pd.DataFrame(cur.fetchall())
    df_steel_grades.columns = [desc[0] for desc in cur.description]
    df_steel_grades.head()



    cur.execute("""
                    SELECT * FROM export."NAS_catWorkCodes";
                    """)
    df_work_codes = pd.DataFrame(cur.fetchall())
    df_work_codes.columns = [desc[0] for desc in cur.description]
    df_work_codes.head()


    cur.execute("""
                    SELECT * FROM export."Nas_catDefectCodes";
                    """)
    df_defect_codes = pd.DataFrame(cur.fetchall())
    df_defect_codes.columns = [desc[0] for desc in cur.description]
    df_defect_codes.head()

    cur.execute("""
                    SELECT * FROM export."Steinb_AP4_Coils";
                    """)
    df_coils = pd.DataFrame(cur.fetchall())
    df_coils.columns = [desc[0] for desc in cur.description]
    df_coils.head()



    cur.execute("""
                    SELECT * FROM export."Steinb_AP4_DefectClasses";
                    """)
    df_defect_classes = pd.DataFrame(cur.fetchall())
    df_defect_classes.columns = [desc[0] for desc in cur.description]
    df_defect_classes.head()


    cur = con.cursor()
    cur.execute("""
                    SELECT * FROM export."Steinb_AP4_Defects";
                    """)
    df_defects = pd.DataFrame(cur.fetchall())
    df_defects.columns = [desc[0] for desc in cur.description]
    df_defects.head()



    cur.execute("""
                    SELECT * FROM export."cm_tblFLInspection";
                    """)
    df_fl_inspection = pd.DataFrame(cur.fetchall())
    df_fl_inspection.columns = [desc[0] for desc in cur.description]
    df_fl_inspection.head()


    cur.execute("""
                    SELECT * FROM export."cm_tblFLQualityInspections";
                    """)
    df_fl_quality_inspection = pd.DataFrame(cur.fetchall())
    df_fl_quality_inspection.columns = [desc[0] for desc in cur.description]
    df_fl_quality_inspection.head()


    cur.execute("""
                    SELECT * FROM export."cm_tblFlatCoils";
                    """)
    df_flat_coils = pd.DataFrame(cur.fetchall())
    df_flat_coils.columns = [desc[0] for desc in cur.description]
    df_flat_coils.head()


    cur.execute("""
                    SELECT * FROM export."cm_tblFlatInspectionMappedDefects";
                    """)
    df_flat_inspection_mapped_defects = pd.DataFrame(cur.fetchall())
    df_flat_inspection_mapped_defects.columns = [desc[0] for desc in cur.description]
    df_flat_inspection_mapped_defects.head()

    cur.execute("""
                    SELECT * FROM export."cm_tblFlatInspectionProcesses";
                    """)
    df_flat_inspection_processes = pd.DataFrame(cur.fetchall())
    df_flat_inspection_processes.columns = [desc[0] for desc in cur.description]
    df_flat_inspection_processes.head()

    # Close cursor and communication with the database
    cur.close()
    return (df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes,
            df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils,
            df_flat_inspection_mapped_defects, df_flat_inspection_processes)

 #read dat from csv

def read_csv_data():
    df_proceso = pd.read_csv('proceso.csv')
    df_customer_claims = pd.read_csv('customer_claims.csv')
    df_finish_codes = pd.read_csv('cat_finish_codes.csv')
    df_lines = pd.read_csv('cat_lines.csv')
    df_steel_grades = pd.read_csv('cat_steel_grades.csv')
    df_work_codes = pd.read_csv('cat_work_codes.csv')
    df_defect_codes = pd.read_csv('cat_defect_codes.csv')
    df_coils = pd.read_csv('flat_coils.csv')
    df_defect_classes = pd.read_csv('AP4_defect_classes.csv')
    df_defects = pd.read_csv('AP4_defects.csv')
    df_fl_inspection = pd.read_csv('inspection.csv')
    df_fl_quality_inspection = pd.read_csv('quality_inspections.csv')
    df_flat_coils = pd.read_csv('flat_coils.csv')
    df_flat_inspection_mapped_defects = pd.read_csv('flat_inspection_mapped_defects.csv')
    df_flat_inspection_processes = pd.read_csv('flat_inspection_processes.csv')
    return  (df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes,
             df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils,
             df_flat_inspection_mapped_defects, df_flat_inspection_processes)

def print_datasets_summary(df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes,
                df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils,
                df_flat_inspection_mapped_defects, df_flat_inspection_processes):
    # List of datasets
    datasets = [df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes,
                df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils,
                df_flat_inspection_mapped_defects, df_flat_inspection_processes]

    # Display basic information and summary statistics for each dataset
    for dataset_name, df in enumerate(datasets, start=1):
        print(f"Dataset: {dataset_name}")
        print(f"Shape: {df.shape}")
        print(f"Info:\n{df.info()}")
        print(f"Summary Statistics:\n{df.describe()}\n")

def preprocessing(df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes,
                df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils,
                df_flat_inspection_mapped_defects, df_flat_inspection_processes):
    # Removing Unnecessary Columns
    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['StartDate', 'StartYear', 'StartMonth', 'StartDay', 'StartHour', 'StartMinute', 'StartSecond',
                       'LastProcessDate', 'LastProcessYear', 'LastProcessMonth', 'LastProcessDay', 'FinishHour',
                       'FinishMinute', 'FinishSecond', 'TotalHoursInLine', 'TotalMinutesInLine', 'TotalSecondsInLine',
                       'EndTime', 'FinishTime', 'ProductionDate', 'CrewID', 'ShiftID', 'isActive', 'isModified']
    df_proceso = df_proceso.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_proceso.dropna()
    df_proceso.drop_duplicates()
    df_proceso.head()


    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['CastDate', 'CustomerNumber', 'ClaimCreateDate', 'QCApprovedDate', 'ClosedDate', 'LastInspectedDate',
                       'GeneralComment1']
    df_customer_claims = df_customer_claims.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_customer_claims.dropna()
    df_customer_claims.drop_duplicates()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['Description', 'isActive']
    df_finish_codes = df_finish_codes.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_finish_codes.dropna()
    df_finish_codes.drop_duplicates()
    df_finish_codes.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['isActive']
    df_lines = df_lines.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_lines.dropna()
    df_lines.drop_duplicates()
    df_lines.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['Description', 'isActive']
    df_steel_grades = df_steel_grades.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_steel_grades.dropna()
    df_steel_grades.drop_duplicates()
    df_steel_grades.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['Description', 'isActive']
    df_work_codes = df_work_codes.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_work_codes.dropna()
    df_work_codes.drop_duplicates()
    df_work_codes.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['Description', 'isFlat', 'isActvie', 'ShortDescription']
    df_defect_codes = df_defect_codes.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_defect_codes.dropna()
    df_defect_codes.drop_duplicates()
    df_defect_codes.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['StartTime', 'EndTime', 'Description', 'PdiRecvTime', 'InternalStatus']
    df_coils = df_coils.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_coils.dropna()
    df_coils.drop_duplicates()
    df_coils.head()

    # Drop columns that are not required for analysis or modeling

    # Drop missing valus and duplicates
    df_defect_classes.dropna()
    df_defect_classes.drop_duplicates()
    df_defect_classes.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['Grade', 'PeriodId', 'PositionCD', 'PositionRCD', 'PositionMD', 'Side', 'SizeCD', 'SizeMD',
                       'CameraNo', 'MergedTo', 'Confidence', 'RoiX0', 'RoiX1', 'RoiY0', 'RoiY1', 'OriginalClass', 'PP_ID',
                       'PostCL', 'MergerPP', 'OnlineCPP', 'OfflineCPP', 'Rollerid', 'InternalStatus', 'CL_PROD_CLASS',
                       'CL_TEST_CLASS', 'AbsPosCD']
    df_defects = df_steel_grades.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_defects.dropna()
    df_defects.drop_duplicates()
    df_defects.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['InspectionDate', 'InspectionDateInt', 'InspectionTime', 'InspectionTimeInt', 'PackProductCode',
                       'Percent1AQualityExt', 'Percent1BQualityExt', 'Percent2QualityExt', 'PercentScrapQualityExt',
                       'Percent1AQualityIntCAP', 'Percent1BQualityIntCAP', 'Percent2QualityIntCAP',
                       'PercentScrapQualityIntCAP', 'Percent1AQualityExtCAP', 'Percent1BQualityExtCAP',
                       'Percent2QualityExtCAP', 'PercentScrapQualityExtCAP', 'Percent1AQualityIntHAP',
                       'Percent1BQualityIntHAP', 'Percent2QualityIntHAP', 'PercentScrapQualityIntHAP',
                       'Percent1AQualityExtHAP', 'Percent1BQualityExtHAP', 'Percent2QualityExtHAP',
                       'PercentScrapQualityExtHAP', 'CreateDate', 'CreateTime', 'ChangeDate', 'ChangeTime', 'isActive',
                       'InspectionDateTime']
    df_fl_inspection = df_fl_inspection.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_fl_inspection.dropna()
    df_fl_inspection.drop_duplicates()
    df_fl_inspection.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['Percent1AQualityExt', 'Percent1BQualityExt', 'Percent2QualityExt', 'PercentScrapQualityExt',
                       'CreateDate', 'CreateTime', 'ChangeDate', 'ChangeTime', 'isActive']
    df_fl_quality_inspection = df_fl_quality_inspection.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_fl_quality_inspection.dropna()
    df_fl_quality_inspection.drop_duplicates()
    df_fl_quality_inspection.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['ProductionDate', 'ShiftID', 'CrewID', 'StartTime', 'EndTime', 'isActive']
    df_flat_coils = df_flat_coils.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_flat_coils.dropna()
    df_flat_coils.drop_duplicates()
    df_flat_coils.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['InspectionMappedDefectID', 'SideID', 'FaceID', 'QualityID']
    df_flat_inspection_mapped_defects = df_flat_inspection_mapped_defects.drop(columns=columns_to_drop)
    # Drop missing valus and duplicates
    df_flat_inspection_mapped_defects.dropna()
    df_flat_inspection_mapped_defects.drop_duplicates()
    df_flat_inspection_mapped_defects.head()

    # Drop columns that are not required for analysis or modeling
    columns_to_drop = ['ProcessStartTime', 'InspectionStartTime', 'InspectionEndTime', 'ApprovedTime', 'UserID', 'active',
                       'Observations']
    df_flat_inspection_processes = df_flat_inspection_processes.drop(columns=columns_to_drop)
    # Drop missing values and duplicates
    df_flat_inspection_processes.dropna()
    df_flat_inspection_processes.drop_duplicates()
    df_flat_inspection_processes.head().head()




# 1 - Classify whether the human inspectors agreed or disagreed with the customer claim defect identification.
#def question1():
    # add code





# 2 - Classify whether a defect claimed by a customer was caught or missed during the production process.
#def question2():
    #add code





# 3- Predict the length of defect coils
def question3():
    q3_df = pd.read_csv("question_3.csv")
    dates = ['InspectionStartTime', 'InspectionEndTime', 'ApprovedTime', 'ProductionDate', 'StartTime', 'EndTime',
             'ProcessStartTime']
    q3_df = q3_df.drop(columns=dates)

    print(q3_df.columns)

    q3_df = pd.get_dummies(q3_df)

    target = 'defectlength'

    # Get all column names
    all_cols = q3_df.columns

    # Get remaining columns (excluding target)
    predictors = set(all_cols).difference({target})
    # Convert predictors set to a list
    predictors = list(predictors)

    # Prepare the data
    X = q3_df[predictors]
    y = q3_df[target]

    # Define a pipeline with scaling and logistic regression
    logistic_regression = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2', C=1.0))

    # Define classifiers
    classifiers = {
        'Logistic Regression': logistic_regression,
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'k-Nearest Neighbors': KNeighborsClassifier()
    }

    # Define training-test splits
    splits = [0.8, 0.6]

    # Define number of folds for cross-validation
    cv_folds = [5, 10]

    # Iterate over classifiers
    for split in splits:
        print(f"\nTraining-Test Split: {split}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split, random_state=42)

        # Choose a regression model and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"Root Mean Squared Error: {rmse:.2f}")

    # Define the number of bins
    num_bins = 10

    # Define labels for the bins
    bin_labels = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 'Bin_8', 'Bin_9', 'Bin_10']

    # Cut the 'Length' variable into 10 bins with labels
    length_bins = pd.cut(y, bins=num_bins, labels=bin_labels)
    # Iterate over classifiers
    for clf_name, clf in classifiers.items():
        print(f"\nClassifier: {clf_name}")
        for cv_fold in cv_folds:
            print(f"\nNumber of Folds: {cv_fold}")
            # Perform cross-validation
            cv_scores = cross_val_score(clf, X, length_bins, cv=cv_fold)
            print("Cross-validation scores:", cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())

# 4- Predict the NetWeight of defect coils
def question4():
    q4_df = pd.read_csv("proceso_netweight.csv")

    print(q4_df.columns)

    q4_df = pd.get_dummies(q4_df)

    target = 'NetWeight'

    # Get all column names
    all_cols = q4_df.columns

    # Get remaining columns (excluding target)
    predictors = set(all_cols).difference({target})
    # Convert predictors set to a list
    predictors = list(predictors)

    # Prepare the data
    X = q4_df[predictors]
    y = q4_df[target]

    # Define a pipeline with scaling and logistic regression
    logistic_regression = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2', C=1.0))

    # Define classifiers
    classifiers = {
        'Logistic Regression': logistic_regression,
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'k-Nearest Neighbors': KNeighborsClassifier()
    }

    # Define training-test splits
    splits = [0.8, 0.6]

    # Define number of folds for cross-validation
    cv_folds = [5, 10]

    # Iterate over classifiers
    for split in splits:
        print(f"\nTraining-Test Split: {split}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split, random_state=42)

        # Choose a regression model and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"Root Mean Squared Error: {rmse:.2f}")

    # Define the number of bins
    num_bins = 10

    # Define labels for the bins
    bin_labels = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 'Bin_8', 'Bin_9', 'Bin_10']

    # Cut the 'Length' variable into 10 bins with labels
    length_bins = pd.cut(y, bins=num_bins, labels=bin_labels)
    # Iterate over classifiers
    for clf_name, clf in classifiers.items():
        print(f"\nClassifier: {clf_name}")
        for cv_fold in cv_folds:
            print(f"\nNumber of Folds: {cv_fold}")
            # Perform cross-validation
            cv_scores = cross_val_score(clf, X, length_bins, cv=cv_fold)
            print("Cross-validation scores:", cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())


def question5():
    # 5- Predict DefectCount on defect code
    df = pd.read_csv('defect_count.csv')

    data_encoded = pd.get_dummies(df, columns=['DefectCode'], drop_first=True)

    # Splitting the dataset into training and testing sets
    X = data_encoded.drop(columns=['totaldefectcount'])
    y = data_encoded['totaldefectcount']

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'k-Nearest Neighbors': KNeighborsClassifier()
    }

    # Define training-test splits
    splits = [0.8, 0.6, 0.5]

    # Define number of folds for cross-validation
    cv_folds = [5, 10]

    # Iterate over classifiers
    for split in splits:
        print(f"\nTraining-Test Split: {split}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split, random_state=42)

        # Choose a regression model and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"Root Mean Squared Error: {rmse:.2f}")

    # Define the number of bins
    num_bins = 10

    # Define labels for the bins
    bin_labels = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 'Bin_8', 'Bin_9', 'Bin_10']

    # Cut the 'NetWeight' variable into 10 bins with labels
    count_bins = pd.cut(y, bins=num_bins, labels=bin_labels)
    # Iterate over classifiers
    for clf_name, clf in classifiers.items():
        print(f"\nClassifier: {clf_name}")
        for cv_fold in cv_folds:
            print(f"\nNumber of Folds: {cv_fold}")
            # Perform cross-validation
            cv_scores = cross_val_score(clf, X, count_bins, cv=cv_fold)
            print("Cross-validation scores:", cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())

def question6():
    # 6- Association, are the defect codes related
    #read data
    df = pd.read_csv('../associationRuleData.csv')

    # Convert defect codes to binary columns
    df['defectcode'] = df['defectcode'].str.split(' - ').str[0]  # Extracting the defect code
    df = pd.get_dummies(df, columns=['defectcode'])

    # Group by CoilNumber and sum the binary columns
    df = df.groupby('CoilNumber').sum().reset_index()

    # Convert counts to binary values (1 for any count > 0, 0 otherwise)
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: 1 if x > 0 else 0)
    df.to_csv('question6.csv', index=False)

    data = pd.read_csv('../question6.csv')

    records = []

    # drop the total column
    data = data.drop(['CoilNumber'], axis=1)

    defect = list(data.columns)

    defectCode = data.to_numpy().astype(str)

    defectCode.shape

    records = [[] for _ in range(df.shape[0])]

    # the create a list of lists from the data set
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if defectCode[i, j] == '1':
                records[i].append(defect[j])


    print("*" * 60)
    association_rules = apriori(records, min_support=0.0005, min_confidence=0.2, min_lift=3, min_length=3)

    rules = list(association_rules)

    for rule in rules:
        # first index of the inner list
        # Contains base item and add item
        pair = rule[0]
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])

        # second index of the inner list
        print("Support: " + str(rule[1]))

        # third index of the list located at 0th
        # of the third index of the inner list

        print("Confidence: " + str(rule[2][0][2]))
        print("Lift: " + str(rule[2][0][3]))
        print("*" * 60)

def anomaly_detection_knn():
    # Anomoly Detection
    # read data
    df = pd.read_csv('customerclaims_weights.csv')
    X = df[['CustomerClaimDefectWeight']]

    # Detect outliers using KNN
    knn = KNN(n_neighbors=3, contamination=0.05)  # Assuming 10% of the data are outliers
    knn.fit(X)
    outlier_labels = knn.predict(X)
    outlier_scores = knn.decision_scores_

    # Create a new DataFrame with the outlier information
    outlier_df = pd.DataFrame({
        'ClaimId': df['ClaimId'],
        'CustomerClaimDefectWeight': df['CustomerClaimDefectWeight'],
        'Outlier_Score': outlier_scores,
        'Outlier': ['Outlier' if label == 1 else 'Normal' for label in outlier_labels]
    })

    print(outlier_df.head())
    print(outlier_df.info())

    # Count the number of outliers
    outlier_count = outlier_df['Outlier'].value_counts().get('Outlier', 0)

    # Print value counts of the 'Outlier' column
    print("Value counts of 'Outlier' column:")
    print(outlier_df['Outlier'].value_counts())

    print("Number of Outliers:", outlier_count)

    # Map outlier labels to colors
    colors = {'Normal': 'blue', 'Outlier': 'red'}
    outlier_df['Color'] = outlier_df['Outlier'].map(colors)

    # Save the results to a file
    outlier_df.to_csv('customer_outlier_results_knn.csv', index=False)

    # Visualize the outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(outlier_df.index, outlier_df['CustomerClaimDefectWeight'], c=outlier_df["Color"])
    plt.title("Outliers in the CustomerClaimDefectWeight")
    plt.xlabel("Observation Index")
    plt.ylabel("CustomerClaimDefectWeight")
    plt.show()


def anomaly_detection_isolationforest():
    # Detect outliers IsolationForest approach

    df = pd.read_csv('customerclaims_weights.csv')
    X = df[['CustomerClaimDefectWeight']]

    # Detect outliers using Isolation Forest
    iso_forest = IsolationForest(contamination=0.05)  # Assuming 1% of the data are outliers
    y_pred = iso_forest.fit_predict(X)

    # Create a new DataFrame with the outlier information
    outlier_df = pd.DataFrame({
        'ClaimId': df['ClaimId'],
        'CustomerClaimDefectWeight': df['CustomerClaimDefectWeight'],
        'Outlier_Score': -iso_forest.decision_function(X),
        'Outlier': ['Outlier' if label == -1 else 'Normal' for label in y_pred]
    })

    # Save the results to a file
    outlier_df.to_csv('customer_outlier_results_isolationforest.csv', index=False)

    # Count the number of outliers
    outlier_count = outlier_df['Outlier'].value_counts().get('Outlier', 0)

    # Print value counts of the 'Outlier' column
    print("Value counts of 'Outlier' column:")
    print(outlier_df['Outlier'].value_counts())

    print("Number of Outliers:", outlier_count)

    # Map outlier labels to colors
    colors = {'Normal': 'blue', 'Outlier': 'red'}
    outlier_df['Color'] = outlier_df['Outlier'].map(colors)

    # Save the results to a file
    outlier_df.to_csv('customer_outlier_results_knn.csv', index=False)

    # Visualize the outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(outlier_df.index, outlier_df['CustomerClaimDefectWeight'], c=outlier_df["Color"])
    plt.title("Outliers in the CustomerClaimDefectWeight")
    plt.xlabel("Observation Index")
    plt.ylabel("CustomerClaimDefectWeight")
    plt.show()



# ANN Modeling
#def ann_modeling():
    #add code











# function to display menu
def displayMenuBar1():
    print("----------------Menu----------------")
    print("1. Read data from database")
    print("2. Read data from csv files")
    print("3. Quit")
    print("------------------------------------")

def displayMenuBar2():
    print("----------------Menu----------------")
    print("1. Display data summary")
    print("2. Question 1")
    print("3. Question 2")
    print("4. Question 3")
    print("5. Question 4")
    print("6. Question 5")
    print("7. Question 6")
    print("8. Anomaly detection using KNN")
    print("9. Anomaly detection using IsolationForest")
    print("10. ANN Modeling")
    print("11. Quit")
    print("------------------------------------")
#
# # main program
#
# ### load data
# # call display menu function
# displayMenuBar1()
#
# # promp user to enter choice
# choice = int(input("Enter your choice: "))
#
# # corresponding tasks with choices
# while(choice!=3):
#     if (choice==1):
#         df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes, df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils, df_flat_inspection_mapped_defects, df_flat_inspection_processes = read_data_from_database()
#     elif (choice==2):
#         df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes, df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils, df_flat_inspection_mapped_defects, df_flat_inspection_processes = read_csv_data()
#     else:
#         print("Invalid choice")
#     print()
#     displayMenuBar1()
#     choice = int(input("Enter your choice: "))
#
#
# # display menu for tasks
# displayMenuBar2()
#
# # promp user to enter choice
# choice = int(input("Enter your choice: "))
#
# # corresponding tasks with choices
# while(choice!=11):
#     if (choice == 1):
#         print_datasets_summary(df_proceso, df_customer_claims, df_finish_codes, df_lines, df_steel_grades, df_work_codes, df_defect_codes, df_coils, df_defect_classes, df_defects, df_fl_inspection, df_fl_quality_inspection, df_flat_coils, df_flat_inspection_mapped_defects, df_flat_inspection_processes)
#     elif (choice == 2):
#         question1()
#     elif (choice == 3):
#         question2()
#     elif (choice == 4):
#         question3()
#     elif (choice == 5):
#         question4()
#     elif (choice == 6):
#         question5()
#     elif (choice == 7):
#         question6()
#     elif (choice == 8):
#         anomaly_detection_knn()
#     elif (choice == 9):
#         anomaly_detection_isolationforest()
#     elif (choice == 10):
#         ann_modeling()
#     else:
#         print("Invalid choice")
#     print()
#     displayMenuBar2()
#     choice = int(input("Enter your choice: "))


question6()