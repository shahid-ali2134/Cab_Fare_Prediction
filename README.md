# Cab Fare Prediction
In the bustling landscape of urban mobility, the ability to predict cab fares accurately holds paramount importance for enhancing operational efficiency and customer satisfaction. This project dives into the intricacies of fare estimation for Uber rides, leveraging a rich dataset that captures a wide array of factors influencing fare costs. The uber.csv dataset encompasses critical information such as pickup and drop-off coordinates, trip durations, distances, times of day, and actual fares charged. By dissecting this data, our aim is to unravel the dynamic and often complex interplay of variables that determine fare prices in real-time urban transport scenarios.

Armed with machine learning and deep learning techniques, this initiative seeks to build predictive models capable of forecasting Uber fares with high precision. Through the meticulous analysis of the dataset, we aspire to identify patterns and correlations that go beyond the apparent. The insights gleaned from this endeavor will not only facilitate more accurate fare predictions but also contribute to the broader objectives of optimizing route planning, managing demand, and elevating the overall user experience in the realm of ride-sharing services.


In our endeavor to create a highly accurate and reliable cab fare prediction system, we will focus on implementing and evaluating two distinct models:
1. Random Forest Model
2. Neural Network


## Models Overview
### Random Forest Model
The Random Forest model is chosen for its robustness and ability to handle the complexities of the dataset through an ensemble approach, effectively capturing the nonlinear relationships between the variables affecting fare prices. It stands out for its ease of use, interpretability, and excellent performance on diverse datasets.

### Neural Network
The Neural Network model represents a more sophisticated deep learning approach, designed to delve deeper into the data's intricacies. By leveraging a layered architecture, this model aims to uncover and learn from the abstract patterns and high-level interactions within the data that are not immediately apparent, offering the potential for superior predictive performance in terms of fare estimation.


## Installation

To run the models and evaluate their performance in the 'Cab Fare Prediction' project, you will need to install the following Python libraries:

1. **pandas:** Essential for data manipulation and analysis, allowing us to work with the dataset in a flexible manner.
2. **numpy:** Provides support for numerical computations, which is crucial for handling data transformations and model inputs.
3. **matplotlib and seaborn:** These libraries are used for creating visualizations, such as accuracy plots and feature importance graphs, to interpret the models' performance and data distributions effectively.
4. **scikit-learn:** A key library for machine learning, used here for Random Forest implementation, data splitting, preprocessing with StandardScaler, and evaluating model performance through metrics like confusion matrix and classification report.
5. **scipy:** While not explicitly listed in your format, the inclusion of from scipy.stats import chi2_contingency suggests statistical tests or analysis may be performed, thus it's beneficial for users to have scipy installed as well.
To install these libraries, you can use the following pip command in your terminal or command prompt:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
## Dataset Overview
For this project, we leverage a dataset to train models capable of predicting cab fares accurately. This dataset, rich in ride-related attributes, provides a foundation for analyzing and understanding factors that influence fare prices, enabling the development of predictive models.

### Features Description
While the specific contents of the uber.csv dataset are not directly reviewed here, a typical Uber dataset may include features like:

1. **Trip ID:** A unique identifier for each ride.
2. **Pickup/Drop-off Coordinates:** The geographical locations of ride start and end points, respectively.
3. **Pickup/Drop-off Time:** Timestamps detailing when the ride began and concluded.
4. **Distance:** The total distance covered during the ride.
5. **Fare Amount:** The charged fare for the ride, which serves as the target variable for our prediction models.
Additional features might encompass details such as ride duration, traffic conditions, type of service (e.g., UberX, UberPool), and weather conditions at the time of the ride, each contributing valuable insights into fare determination.

### Target Variable
The primary objective of our analysis is to predict the Fare Amount, making it the target variable. This continuous variable represents the cost of the ride, influenced by various factors including distance, time of day, ride type, and external conditions.


## Data Pre-Processing
To ensure the integrity and quality of the data used for our cab fare prediction models, we undertook a series of preprocessing steps on the Uber dataset. These steps are crucial for handling missing values, correcting data inconsistencies, and refining our dataset for optimal model performance.

### Handling Missing and Incorrect Values, Geographical Filtering
- Converted fare_amount to numeric, setting incorrect entries to NaN and replacing zero values with NaN to indicate missing data.
- Missing passenger_count values were filled with 0, converted to integers, and zeros were replaced with NaN for clarity.
- Zero values in location coordinates (pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude) were replaced with NaN to signify missing data.
- Removed rows where pickup and dropoff locations are identical, as these do not contribute to meaningful fare prediction.
```
# Handling Missing and Incorrect Values, Geographical Filtering

df['fare_amount'] = df['fare_amount'].apply(pd.to_numeric, errors='coerce')
df['fare_amount'] = df['fare_amount'].replace({0: np.nan})
df['passenger_count'] = df['passenger_count'].fillna(0)
df['passenger_count'] = df['passenger_count'].astype(int)
df['passenger_count'] = df['passenger_count'].replace({0: np.nan})
df['pickup_longitude'] = df['pickup_longitude'].replace({0: np.nan})
df['pickup_latitude'] = df['pickup_latitude'].replace({0: np.nan})
df['dropoff_longitude'] = df['dropoff_longitude'].replace({0: np.nan})
df['dropoff_latitude'] = df['dropoff_latitude'].replace({0: np.nan})

df = df[np.logical_and(df['pickup_longitude'] != df['dropoff_longitude'],
                       df['pickup_latitude'] != df['dropoff_latitude'])]
```

### Data Distribution Visualization
- Visualized the distribution of key features including fare_amount, and pickup/dropoff coordinates using seaborn's distribution plots to understand data spread and identify outliers.
- Missing Values Analysis and Imputation
- Calculated and sorted missing values by percentage to prioritize handling.
- Imputed missing values in fare_amount and location coordinates with their mean to maintain data integrity.
- For passenger_count, missing values were filled with the mode, reflecting the most common scenario.
```
# Data Distribution Visualization

sns.set(style='darkgrid', palette='Set1')
plt.figure(figsize=(20, 20))
plt.subplot(321)
_ = sns.distplot(df['fare_amount'], bins=50)
plt.subplot(322)
_ = sns.distplot(df['pickup_longitude'], bins=50)
plt.subplot(323)
_ = sns.distplot(df['pickup_latitude'], bins=50)
plt.subplot(324)
_ = sns.distplot(df['dropoff_longitude'], bins=50)
plt.subplot(325)
_ = sns.distplot(df['dropoff_latitude'], bins=50)
plt.show()
```
![1](https://github.com/shahid-ali2134/CabFarePrediction/assets/88580273/658741f7-e34d-4213-8336-cf3cf8958d64)


```
# Calculate Missing Values
missing_val = pd.DataFrame(df.isnull().sum())
missing_val = missing_val.reset_index()
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'count'})
missing_val['Missing_percentage'] = (missing_val['count']/len(df)*100)
missing_val = missing_val.sort_values('Missing_percentage', ascending=False).reset_index(drop=True)

# Impute Missing Values
df['fare_amount'] = df['fare_amount'].fillna(df['fare_amount'].mean())
df['pickup_longitude'] = df['pickup_longitude'].fillna(df['pickup_longitude'].mean())
df['pickup_latitude'] = df['pickup_latitude'].fillna(df['pickup_latitude'].mean())
df['dropoff_longitude'] = df['dropoff_longitude'].fillna(df['dropoff_longitude'].mean())
df['dropoff_latitude'] = df['dropoff_latitude'].fillna(df['dropoff_latitude'].mean())
df['passenger_count'] = df['passenger_count'].fillna(int(df['passenger_count'].mode()))
```


### Outlier Detection and Handling
- Identified and treated outliers in numerical features based on IQR method, setting values beyond a calculated range to NaN.
- Imputed these missing values post-outlier detection with mean for location coordinates and mode for passenger_count to ensure data consistency.
```
# Detect and handle outliers for numerical columns
coutliers = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
for col in coutliers:
    q75, q25 = np.percentile(df[col], [75 ,25])
    iqr = q75 - q25
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    df.loc[df[col] < minimum, col] = np.nan
    df.loc[df[col] > maximum, col] = np.nan

# Impute missing values post-outlier detection
df['pickup_longitude'] = df['pickup_longitude'].fillna(df['pickup_longitude'].mean())
df['pickup_latitude'] = df['pickup_latitude'].fillna(df['pickup_latitude'].mean())
df['dropoff_longitude'] = df['dropoff_longitude'].fillna(df['dropoff_longitude'].mean())
df['dropoff_latitude'] = df['dropoff_latitude'].fillna(df['dropoff_latitude'].mean())
df['passenger_count'] = df['passenger_count'].fillna(int(df['passenger_count'].mode()))
```



### Feature Engineering
- Utilized the Haversine formula to calculate distances between pickup and dropoff points, adding valuable spatial information for our models.
```
# Haversine formula to calculate distance between pickup and dropoff points
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return earth_radius * 2 * np.arcsin(np.sqrt(a))
```

### Correlation Analysis
- Performed correlation analysis among numerical features to identify potential relationships and redundancies, visualized through a heatmap.
```
# Correlation analysis among numerical features
numeric = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
df_corr = df.loc[:, numeric]
corr = df_corr.corr()

# Visualization with heatmap
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
```
![2](https://github.com/shahid-ali2134/CabFarePrediction/assets/88580273/1ef47ce7-20bc-41f9-8d81-1fe193ac14d3)



### Final Data Cleaning
- Removed entries with identical pickup and dropoff locations as well as irregular fare_amount and passenger_count values, ensuring the dataset only contains valid and meaningful records.
```
# Remove entries with identical pickup and dropoff locations and irregular values
df = df[np.logical_and(df['pickup_longitude'] != df['dropoff_longitude'], df['pickup_latitude'] != df['dropoff_latitude'])]
df.loc[df['fare_amount'] < 0, 'fare_amount'] = np.nan
df.loc[df['fare_amount'] > 30, 'fare_amount'] = np.nan
df.loc[df['passenger_count'] > 8, 'passenger_count'] = np.nan
df = df.dropna()
```

## Splitting Data for Testing and Training
We use the train_test_split method from scikit-learn to partition the dataset. This method allows us to randomly divide the data into training and testing sets, with the testing set comprising 20% of the total data. This proportion ensures that we have sufficient data for both training our models and validating their performance.

The code snippet below demonstrates how to perform this split and print the shapes of the resulting datasets, providing insight into the distribution of data across the training and testing sets:
```bash
# Split the data into features (X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Print the shape of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
)
```
## Model Training
Each model was trained individually on the training dataset and eventually tested with the new unseen testing data, here's how it's done,

### Random Forest Model:
The model is instantiated with RandomForestRegressor from scikit-learn, specifying **n_estimators=100** to create a forest of 100 trees. This parameter is crucial as it balances the model's complexity with computational efficiency. We also set **random_state=42** to ensure reproducibility of our results.

Following instantiation, the model is trained (fit) on the training dataset, comprising the preprocessed features (X_train) and labels (y_train). This step involves the Random Forest algorithm learning the intricate patterns and relationships in the data to predict cab fares accurately.



```bash
# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create an instance of Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

```

### Neural Network :
The model is constructed using TensorFlow's Keras API, which provides a more accessible interface for building and training deep learning models. Our neural network consists of:

- An input layer with 64 neurons and a ReLU (Rectified Linear Unit) activation function, designed to process the input features (X_train.shape[1], denotes the number of features).
- A hidden layer with 32 neurons, also using the ReLU activation function, to further abstract the relationships in the data.
- An output layer with a single neuron, as this is a regression task aiming to predict a continuous value—the fare amount.
**Compilation and Training:** The model is compiled with the Adam optimizer, a popular choice for deep learning tasks due to its efficient computation and adaptive learning rate capabilities. The loss function specified is 'mean_squared_error', appropriate for regression problems, and we track mean absolute error (MAE) and mean squared error (MSE) as metrics during training.

Training is conducted over 10 epochs with a batch size of 32, and we use 20% of the training data as a validation set. This approach helps monitor the model's performance on unseen data during training, aiding in the detection and mitigation of overfitting.

```
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create a neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1))  # Output layer for regression


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Make predictions on the scaled test data
y_pred_nn = model.predict(X_test).flatten()

```

## Results and Observations
After training, the model makes predictions on the unseen test dataset (X_test), allowing us to assess its performance through various metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), the R-squared (R²) score, and the Mean Absolute Percentage Error (MAPE). These metrics provide a comprehensive view of the model's accuracy, error rate, and the proportion of variance in the fare amount that is predictable from the features, respectively.

### Comparative Analysis:
A comprehensive table that summarizes the performance metrics for each model is shown below. This table will encapsulate key performance metrics such as MSE, MAE, r2 score, and MAPE, providing a clear, concise comparison of each model's effectiveness in predicting cab fare.

$$
\begin{array}{|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{MSE} & \textbf{MAE} & \textbf{r2 Score} & \textbf{MAPE} \\
\hline
RF  & 6.7 & 1.74 & 0.74 & 19.8 \\
\hline
Neural Network & 10.6 & 2.19 & 0.6 & 25.5 \\
\hline
\end{array}$$

**MSE (Mean Squared Error):** The RF model exhibits a lower MSE of 6.7 compared to the Neural Network's 10.6. This indicates that, on average, the squared difference between the predicted and actual fare amounts is smaller for the RF model, suggesting higher accuracy in predicting cab fares.

**MAE (Mean Absolute Error):** Similarly, the RF model outperforms the Neural Network with a lower MAE of 1.74 against 2.19. This metric shows that the average absolute difference between predicted and actual values is smaller for the RF model, further indicating its superior accuracy in fare prediction.

**r2 Score:** The RF model achieves a higher R-squared score of 0.74, in contrast to the Neural Network's 0.6. The R-squared score measures the proportion of variance in the dependent variable that is predictable from the independent variables. A higher R-squared value for the RF model means it is better at capturing the variance in cab fare prices from the given features.

**MAPE (Mean Absolute Percentage Error):** The RF model also demonstrates a better performance with a lower MAPE of 19.8%, compared to the Neural Network's 25.5%. This implies that the RF model has a lower average percentage error in its fare predictions, making it more reliable for estimating cab fares.

The comparative analysis of the RF and Neural Network models based on the provided metrics suggests that the Random Forest model generally performs better in predicting cab fares. It not only achieves lower errors as indicated by MSE and MAE but also exhibits a higher capability in explaining the variance in fare prices through the R-squared score and demonstrates a lower percentage error in predictions as shown by MAPE.

This analysis underscores the importance of choosing the right model based on performance metrics relevant to the task at hand. While the RF model shows superior performance in this scenario, the selection between models should also consider factors like training time, model complexity, and scalability depending on the application's specific requirements and constraints.

## Conclusion
The 'Cab Fare Prediction' project represents a significant step forward in harnessing the power of machine learning to tackle the complex challenge of predicting cab fares. By leveraging a dataset of Uber rides, we successfully developed and compared two models: a Random Forest and a Neural Network. Our analysis revealed that the Random Forest model outperformed the Neural Network across several key performance metrics, including MSE, MAE, r2 Score, and MAPE. This outcome highlights the efficacy of ensemble methods in dealing with predictive modeling tasks, especially when faced with the intricacies of real-world data. The project underscored the critical role of data preprocessing and feature engineering in enhancing model performance and provided insights into the selection and evaluation of predictive models based on their accuracy and reliability.

Looking ahead, this endeavor opens up avenues for further refinement and exploration. Optimizing the Neural Network architecture, integrating additional contextual data, and implementing real-time prediction capabilities are promising directions that could elevate the project's impact. Ultimately, this work lays a foundational blueprint for future advancements in predictive analytics within the urban mobility sector, showcasing the potential of machine learning to improve service delivery and customer satisfaction in the ride-sharing ecosystem.

