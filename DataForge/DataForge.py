from sklearn.metrics import r2_score
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# from sklearn 
from sklearn.metrics import accuracy_score, mean_squared_error,precision_score, recall_score, f1_score, classification_report


# func for classification:

def classification(X,y,target):
    return 1
     

    
   



    


def main():


            st.warning("This wont work when target columns have empty values and string values")

            Result_technique= st.radio("What you want to do?",["Classification","Regression"])
    # if Result_technique=="Classification":
         
            
            st.title("Classification And Prediction")
            upload_file=st.file_uploader("Upload  CSV")



            if upload_file is not None:
                data=pd.read_csv(upload_file)
                st.header('1. Random 10 Data Values from uploaded file')
                st.dataframe(data.sample(10))
                st.markdown("""---""")
                target=st.sidebar.selectbox("Select Target Column [which need to be predicted]",data.columns)

                X=data.drop(target,axis=1)
                y=data[target]
                select_cols=st.sidebar.multiselect("Select Columns To Drop [Which Are Not Relevant For Prediction]",X.columns)
                st.subheader("After dropping selected columns:")
                if select_cols:
                    X=X.drop(columns=select_cols)
                st.dataframe(X.sample(10))
                # CHECK FOR NA:
                st.write("Processing For Empty Values")
                col_with_nan=X.columns[X.isnull().any()].to_list()
                st.write(col_with_nan)
                if(col_with_nan):

                    for col in col_with_nan:
                        fill = st.radio(f"Pick a method for handling NaN in column '{col}'", ["mean", "median", "drop"])
                # 
                        # for i in col:

                        if fill=="mean":
                            X[col]=X[col].fillna(data[col].mean())
                        elif fill=="median":
                            X[col]=X[col].fillna(data[col].median())
                        elif fill=='drop':
                            X=X.dropna(subset=[col])
                    st.write("After Applying mentioned Technique")
                    st.dataframe(data)

                st.markdown("""---""")









                # target=st.sidebar.selectbox("Select Target Column [which need to be predicted]",data.columns)

                # X=data.drop(target,axis=1)
                # y=data[target]

                # st.bar_chart()

            # col -select

                # st.sidebar.subheader("Select Column")

                # select_cols=st.sidebar.multiselect("Select Columns To Drop [Which Are Not Relevant For Prediction]",X.columns)
                # st.subheader("After dropping selected columns:")
                # if select_cols:
                #     X=X.drop(columns=select_cols)
                # st.dataframe(X.sample(10))


                text_columns=X.select_dtypes(include='object').columns

                X=pd.get_dummies(X,columns=text_columns,drop_first=True)


                test_size= st.sidebar.slider('Select test size',0.2,0.3)
                X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=test_size,random_state=42)
                st.header("2. Basic Summary of Data")
                st.write("2.1 Values containing target-values",data[target].value_counts())


                st.write("2.2 Description Of data",data.describe())

                st.write("Want to scale out data?")

                option= st.radio("scaling",["yes","no"])

                if option =="yes":

                    scalling=st.radio("select Method to scale out data",["standard scaling","MinMaxScaler"])



                    if scalling=="standard scaling":

                            scaler = StandardScaler()
                            scaled_train_data=scaler.fit_transform(X_train)
                            scaled_test_data=scaler.fit_transform(X_test)
                            st.dataframe(scaled_train_data[:5])

                    if scalling=="MinMaxScaler":
                            scaler = MinMaxScaler()
                            scaled_train_data=scaler.fit_transform(X_train)
                            scaled_test_data=scaler.fit_transform(X_test)
                            st.dataframe(scaled_train_data[:5])


                st.header("3.Results")
                if Result_technique=="Classification":
                    st.title("Classification")
             # define all classifiers
                    st.subheader("3.1 Random Forest")
            #  r    f-classifier
                    rf_classifier = RandomForestClassifier()
                    rf_classifier.fit(X_train,y_train)
                    rf_pred=rf_classifier.predict(X_test)
                    acc_rf=accuracy_score(y_test,rf_pred)

                    st.write(f"Accuracy: {acc_rf.round(2)*100}%")
                    report_dict=classification_report(y_test,rf_pred,output_dict=True)
                    report_df=pd.DataFrame(report_dict).transpose()
                    st.write(f"Report")
                    st.table(report_df)
                    st.markdown("""---""")

                    st.subheader("3.2 Logistic Regression")
                    # lr-classifier
                    lr_classifier = LogisticRegression()
                    lr_classifier.fit(X_train,y_train)
                    lr_pred=lr_classifier.predict(X_test)
                    acc_lr=accuracy_score(y_test,lr_pred)
                    st.write(f"Accuracy: {acc_lr.round(2)*100}%")
                    report_dict=classification_report(y_test,lr_pred,output_dict=True)
                    report_df=pd.DataFrame(report_dict).transpose()
                    st.write(f" Report")
                    st.table(report_df)
                    st.markdown("""---""")

                    # knn 
                    st.subheader("3.3 KNN Classifier")

                    knn_classifier = KNeighborsClassifier()
                    knn_classifier.fit(X_train,y_train)
                    knn_pred=knn_classifier.predict(X_test)
                    acc_knn=accuracy_score(y_test,knn_pred)
                    st.write(f"Accuracy: {acc_knn.round(2)*100}%")
                    report_dict=classification_report(y_test,knn_pred,output_dict=True)
                    report_df=pd.DataFrame(report_dict).transpose()
                    st.write(f" Report")
                    st.table(report_df)
                    st.markdown("""---""")

                    # ada-boost
                    st.subheader("3.4 AdaBoost Classifier")
                    ab_classifier = AdaBoostClassifier()
                    ab_classifier.fit(X_train, y_train)
                    ab_pred = ab_classifier.predict(X_test)
                    acc_ab = accuracy_score(y_test, ab_pred)
                    st.write(f"Accuracy: {acc_ab.round(2)*100}%")
                    report_dict = classification_report(y_test, ab_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.write(f" Report")
                    st.table(report_df)
                    st.markdown("""---""")




                    st.cache_data.clear()

                elif Result_technique=="Regression":
                    st.title("Regression ")
                     
                    # lr
                    st.subheader("3.5 Linear Regression")
                    lr_reg=LinearRegression()
                    lr_reg.fit(X_train,y_train)
                    lr_pred=lr_reg.predict(X_test)
                    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": lr_pred}).sample(10))
                    # classification(X,y,target)

    





       
if __name__ == "__main__":

    main()




