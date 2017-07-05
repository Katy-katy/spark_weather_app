# Spark Weather App 


to build:
mvn package

to run (assuming that spark is in “Applications” folder): 
/Applications/spark-1.6.1/bin/spark-submit --class "SimpleApp" --master local[4] target/simple- project-1.0.jar

My spark app reads raw data from “AirQualityUCI.csv”. Then it make data cleaning and preparation and writing the clean data to “AirQualityUCI_WithLabels.csv.”
Label is the temperature 24 hours late.
“AirQualityUCI_WithLabels.csv" does not include the rows form original file that has “-200” in the “T” column since it looks like as a default value for missing data. Also it does not include the row that could have “-200” as a label. All commas in the numbers are replaced by dots since then we will use Double.parseDouble(“some_string”) and it does not work with strings that contain commas.
Then my app reads data from “AirQualityUCI_WithLabels.csv” into
I decided to use LinearRegressionWithSGD model since we have a regression problem. I did not do feature normalization, but I tried some different step size in range 0.01 - 0.0000001 and got the best result using 0.001. I also tried different number of iterations and some different sets of features.
To train my algorithm I used 70% of data. Then I tested on my testing set (30% of data). I used MSE as my metric.
I got the best result MSE = 8.0 on testing set (since I used random split the MSE is little bit different every run - about 8.0-8.6).
I used only two features - temperature and difference of temperatures for previous 24 hours. For example if we want to predict temperature on May, 8th, 6.00 am, we use temperature on May, 7th, 6.00 am and difference between the temperatures on May, 7th, 6.00 am and May, 6th, 6.00 am.
I like this project! Actually, it was my first experience with MLLIB and Java for machine learning (usually I use python with Scikit-Learn, but I have experience with Java for mobile apps). I think,
