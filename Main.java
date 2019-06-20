import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.sql.*;
import scala.Array;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.List;
import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;


public class Main {

    public static void main(String[] args) {

        //Create a SparkContext to initialize
        SparkConf conf = new SparkConf().setMaster("local").setAppName("Cure Algorithm");
        SparkSession ss =new SparkSession.Builder()
                .config(conf)
                .appName("Cure Algorithm").master("local")
                .getOrCreate();
         // Create a Java version of the Spark Context
         JavaSparkContext sc = JavaSparkContext.fromSparkContext(ss.sparkContext());

        String files = "src/main/data/*.txt";
        //System.out.println(files);
        JavaRDD<String> lines = sc.textFile(files);

        JavaRDD<Tuple2<Double, Double>> parsedData;
        parsedData = lines.map((line) -> line.split(","))
                .map((line) -> new Tuple2<>(parseDouble(line[0]), parseDouble(line[1])))
                .cache();

        JavaPairRDD<Double, Double> pairsRDD = parsedData.mapToPair((line) -> new Tuple2<>(line._1(), line._2()));


        //random sampling our data into one java pair rdd

        JavaPairRDD<Double, Double> sampledData = pairsRDD.sample(true, 0.001);
        sampledData = sampledData.mapToPair((line) -> new Tuple2<>(line._1(), line._2()));
        sampledData.coalesce(1).saveAsTextFile("resultsjava");


        String files2 = "C:\\Users\\dell\\Desktop\\curejavaspark\\src\\main\\java\\predictions.csv";
        //remove header and broadcast it
        JavaRDD<String> lines2 = sc.textFile(files2);
        String header = lines2.first();
        Broadcast<String> bheader = sc.broadcast(header);
        JavaRDD<String> lines2wh = lines2.filter(line -> !line.equals(bheader.value()));

        //create java rdd for prediction file [(x,y),prediction]
        JavaRDD<Tuple2<Tuple2<Double, Double>, Integer>> predictionsfile;
        predictionsfile = lines2wh.map((line) -> line.split(","))
                .map((line) -> new Tuple2<>(new Tuple2<>(parseDouble(line[1]), parseDouble(line[2])), parseInt(line[3])))
                .cache();


        //create java pair rdd and Group By Cluster the predictionfile rdd
        //we have a triple (cluster,all the points of these cluster (x,y))
        JavaPairRDD<Integer, ArrayList<Tuple2<Double, Double>>> finalpredictfile = predictionsfile.groupBy(line -> line._2())
                .mapValues(line -> {
                    ArrayList<Tuple2<Double, Double>> returnarray = new ArrayList<>();
                    for (Tuple2<Tuple2<Double, Double>, Integer> tripleta : line) {
                        returnarray.add(tripleta._1());
                    }
                    return returnarray;
                });


        //we calculate the reps (4 reps per cluster)
        //cluster 1: the four reps
        //cluster 2: the 4 reps
        //....
        JavaPairRDD<Integer, ArrayList<Tuple2<Double, Double>>> reps = finalpredictfile.mapValues(list -> representatives(list, 4));

        //cluster 1, the four reps
        //cluster 1, the four reps
        //...
        JavaPairRDD<Integer,Tuple2<Double,Double>> flatreps= reps.flatMapValues(list -> list);

        List<Tuple2<Integer,Tuple2<Double,Double>>> flatrepsserial = flatreps.collect();

        //flatrepserial: cluster 4,[rep]
                         //cluster 4,[rep]
        //broadcast flatrepsserial
        Broadcast<List<Tuple2<Integer,Tuple2<Double,Double>>>> bflatrepsserial = sc.broadcast(flatrepsserial);

        //in predictions now we have points x,y and prediction for all the dataset
        JavaPairRDD<Double[],Integer> predictions = parsedData.mapToPair(point -> new Tuple2<Tuple2<Double,Double>,Integer>(point,functclust(point,bflatrepsserial.value())))
                .mapToPair(row -> new Tuple2<Double[],Integer>(new Double[]{row._1()._1(),row._1()._2()},row._2()));

//############Silhouette#######################

        //Dataset<Row> df = ss.createDataset(JavaPairRDD.toRDD(predictions), Encoders., Encoders.INT()).toDF("features","prediction");
       // df.show(4);

         //Evaluate clustering by computing Silhouette score
      //  ClusteringEvaluator evaluator = new ClusteringEvaluator();

       // double silhouette = evaluator.evaluate(df);
       // System.out.println("Silhouette with squared euclidean distance = " + silhouette);
        System.out.println("The cure algorithm is done (success)");

    }

  //  ###############END OF MAIN#######################

    public static Integer functclust(Tuple2<Double,Double> point, List<Tuple2<Integer,Tuple2<Double,Double>>> reps){

        String s;
        Double mindist = Double.MAX_VALUE;
        Integer mincluster = -1;
        for (Tuple2<Integer,Tuple2<Double,Double>>p : reps) {
            Double crdist = distance(p._2(), point);
            if (crdist<mindist) {
                mindist=crdist;
                mincluster=p._1();
            }
        }
        return mincluster;
    }


    public static ArrayList<Tuple2<Double, Double>> representatives(ArrayList<Tuple2<Double, Double>> points, int numberofrepr) {

        //find the mean of cluster
        Double x = 0.0;
        Double y = 0.0;

        //first repr
        for (Tuple2<Double, Double> point : points) {
            x += point._1();
            y += point._2();
        }
        Double meanx = x / points.size();
        Double meany = y / points.size();
        ArrayList<Tuple2<Double, Double>> replist = new ArrayList<>();
        Double maxdist = 0.0;
        Tuple2<Double, Double> maxpoint = null;
        for (Tuple2<Double, Double> p : points) {
            Double crdist = distance(p, new Tuple2<Double, Double>(meanx, meany));
            if (crdist>maxdist) {
                maxdist=crdist;
                maxpoint=p;
            }
        }
        replist.add(maxpoint);

        //for the rest representatives
        for (int i=1;i<numberofrepr;i++) {
            Double maxdist2 = 0.0;
            Tuple2<Double, Double> maxpoint2 = null;
            for (Tuple2<Double, Double> p : points) {
                Double crdist = distance(p, replist.get(replist.size()-1));
                if (crdist>maxdist2) {
                    maxdist2=crdist;
                    maxpoint2=p;
                }
            }
            replist.add(maxpoint2);
        }

        return shrinkrepr(replist,0.2);
    }

        public static Double distance (Tuple2 < Double, Double > p1, Tuple2 < Double, Double > p2){
            return Math.sqrt(Math.pow(Math.abs(p1._1() - p2._1()), 2) + Math.pow(Math.abs(p1._2() - p2._2()), 2));
        }

        public static ArrayList<Tuple2<Double,Double>> shrinkrepr(ArrayList<Tuple2<Double,Double>> pointssh, Double a){
            Double x2 = 0.0;
            Double y2 = 0.0;

            for (Tuple2<Double, Double> point : pointssh) {
                x2 += point._1();
                y2 += point._2();
            }
            Double meanx2 = x2 / pointssh.size();
            Double meany2 = y2 / pointssh.size();

            ArrayList<Tuple2<Double, Double>> replistsh = new ArrayList<>();
            for (Tuple2<Double, Double> p : pointssh) {
                Double crx =p._1()+a*(meanx2-p._1());
                Double cry=p._2()+a*(meanx2-p._2());
                replistsh.add(new Tuple2<Double,Double>(crx,cry));
            }

        return replistsh;
        }
}


