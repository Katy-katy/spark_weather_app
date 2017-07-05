import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import scala.Tuple2;
import java.io.*;
import org.apache.spark.mllib.regression.LassoModel;
import org.apache.spark.mllib.regression.LassoWithSGD;
import org.apache.spark.mllib.regression.RidgeRegressionModel;
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD;

public class SimpleApp {
  public static void main(String[] args) throws IOException{
    String dataFile = "AirQualityUCI.csv"; // Should be some file on your system
    String clenedDataFile = "AirQualityUCI_WithLabels.csv";
    //data cleanig and preporation
    //creating of an array of labels (column "T")
    String[] labels = new String[10000];
    int ind = 0;

    BufferedReader br_l = new BufferedReader(new FileReader(dataFile));
    String row = "";
    while ((row = br_l.readLine()) != null) {
      String[] cols = row.split(";");
      if (cols.length == 0) break;
      if (cols[0].contains("Date")) continue;
      labels[ind] = cols[12].replaceAll(",", ".");
      ind = ind + 1;
    }

    // appending the labels column to the original data table (for every row
    // we append the label that equal to the temp on the next day at the
    // same time)
    BufferedReader br = new BufferedReader(new FileReader(dataFile));
    BufferedWriter bw = null;
    FileWriter fw = null;
    fw = new FileWriter(clenedDataFile);
		bw = new BufferedWriter(fw);

    String l = "";
    int ind2 = 24;
    while ((l = br.readLine()) != null) {
      String clean_line = l.replaceAll(",", ".");
      String[] cols = clean_line.split(";");
      if (cols.length == 0) break;
      if (cols[0].contains("Date")) continue;
      if (labels[ind2]==null) break;
      if (cols[12].contains("-200") || labels[ind2].contains("-200")){
        ind2 = ind2 + 1;
        continue;
      }
      double diff;
      if (ind2 - 48 >= 0){
          diff = Double.parseDouble(cols[12]) - Double.parseDouble(labels[ind2-48]);
      }
      else{
          diff = 0.0;
      }
      bw.write(diff + ";" + clean_line + labels[ind2] + "\n");
      ind2 = ind2 + 1;
    }

    SparkConf conf = new SparkConf().setAppName("Weather Prediction");
    JavaSparkContext sc = new JavaSparkContext(conf);
    JavaRDD<String> data = sc.textFile(clenedDataFile).cache();
      
    JavaRDD <String> filteredData = data.filter( x -> x.split(";").length == 18);

    JavaRDD<LabeledPoint> parsedData = filteredData.map(line -> {
      String[] parts = line.split(";");
      String[] features =  {parts[0], parts[13]};
        //String[] features =  {parts[0], parts[13], parts[14]};
       // String[] features =  {parts[13]};
      
      double[] v = new double[features.length];
      for (int i = 0; i < features.length; i++) {
        v[i] = Double.parseDouble(features[i]);
      }
      return new LabeledPoint(Double.parseDouble(parts[17]), Vectors.dense(v));
    });
    parsedData.cache();

    JavaRDD<LabeledPoint>[] tmp = parsedData.randomSplit(new double[]{0.7, 0.3});
    JavaRDD<LabeledPoint> training = tmp[0]; // training set
    JavaRDD<LabeledPoint> test = tmp[1]; // test set

    // Building the model
    int numIterations = 100;
    double stepSize = 0.001;// MSE_train = 8.3, MSE_test = 8.6
  
    LinearRegressionModel model =
      LinearRegressionWithSGD.train(JavaRDD.toRDD(training), numIterations, stepSize);  
      
      /*
    LassoModel model =
      LassoWithSGD.train(JavaRDD.toRDD(training), numIterations);
     */
      
  /*   RidgeRegressionModel model =
      RidgeRegressionWithSGD.train(JavaRDD.toRDD(training), numIterations, stepSize, 10.0); */
      
    // Evaluate model on trening examples and compute trening error
    JavaRDD<Tuple2<Double, Double>> valuesAndPreds_train = training.map(
      new Function<LabeledPoint, Tuple2<Double, Double>>() {
        public Tuple2<Double, Double> call(LabeledPoint point) {
          double prediction = model.predict(point.features());
          return new Tuple2<>(prediction, point.label());
        }
      }
      );
      double MSE_train = new JavaDoubleRDD(valuesAndPreds_train.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
            System.out.println(" " + pair._1() + " " + pair._2());
          return Math.pow(pair._1() - pair._2(), 2.0);
        }
      }
      ).rdd()).mean();
      System.out.println("train Mean Squared Error = " + MSE_train);

    // Evaluate model on testning examples and compute testing error
    JavaRDD<Tuple2<Double, Double>> valuesAndPreds = test.map(
      new Function<LabeledPoint, Tuple2<Double, Double>>() {
        public Tuple2<Double, Double> call(LabeledPoint point) {
          double prediction = model.predict(point.features());
          return new Tuple2<>(prediction, point.label());
        }
      }
      );
      double MSE = new JavaDoubleRDD(valuesAndPreds.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
          return Math.pow(pair._1() - pair._2(), 2.0);
        }
      }
      ).rdd()).mean();
      System.out.println("test Mean Squared Error = " + MSE);
    sc.stop();
  }
}
