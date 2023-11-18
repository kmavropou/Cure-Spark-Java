import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.List;
import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;

public class Main {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("Cure Algorithm");

        try (SparkSession ss = SparkSession.builder().config(conf).appName("Cure Algorithm").master("local").getOrCreate();
             JavaSparkContext sc = JavaSparkContext.fromSparkContext(ss.sparkContext())) {

            String projectRoot = System.getProperty("user.dir");
            String dataPath = projectRoot + "/src/main/data/*.txt";
            String predictionsPath = projectRoot + "/cure-algorithm/predictions.csv";

            JavaRDD<String> lines = sc.textFile(dataPath);

            JavaRDD<Tuple2<Double, Double>> parsedData = lines.map(line -> line.split(","))
                    .map(array -> new Tuple2<>(parseDouble(array[0]), parseDouble(array[1])))
                    .cache();

            JavaPairRDD<Double, Double> sampledData = parsedData.sample(true, 0.001);

            sampledData.coalesce(1).saveAsTextFile("resultsjava");

            JavaRDD<String> lines2 = sc.textFile(predictionsPath);
            String header = lines2.first();
            Broadcast<String> bheader = sc.broadcast(header);
            JavaRDD<String> lines2wh = lines2.filter(line -> !line.equals(bheader.value()));

            JavaRDD<Tuple2<Tuple2<Double, Double>, Integer>> predictionsfile = lines2wh.map(line -> line.split(","))
                    .map(array -> new Tuple2<>(new Tuple2<>(parseDouble(array[1]), parseDouble(array[2])), parseInt(array[3])))
                    .cache();

            JavaPairRDD<Integer, ArrayList<Tuple2<Double, Double>>> finalpredictfile = predictionsfile
                    .groupBy(Tuple2::_2)
                    .mapValues(list -> new ArrayList<>(list.map(Tuple2::_1).toJavaList()));

            JavaPairRDD<Integer, ArrayList<Tuple2<Double, Double>>> reps = finalpredictfile.mapValues(list -> representatives(list, 4));
            JavaPairRDD<Integer, Tuple2<Double, Double>> flatreps = reps.flatMapValues(List::iterator);

            List<Tuple2<Integer, Tuple2<Double, Double>>> flatrepsserial = flatreps.collect();

            Broadcast<List<Tuple2<Integer, Tuple2<Double, Double>>>> bflatrepsserial =
                    sc.broadcast(JavaConverters.asScalaBuffer(flatrepsserial).toList());

            JavaPairRDD<Double[], Integer> predictions = parsedData
                    .mapToPair(point -> new Tuple2<>(new Double[]{point._1(), point._2()}, functclust(point, bflatrepsserial.value())));

            System.out.println("The cure algorithm is done (success)");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Integer functclust(Tuple2<Double, Double> point, List<Tuple2<Integer, Tuple2<Double, Double>>> reps) {
        Double mindist = Double.MAX_VALUE;
        Integer mincluster = -1;

        for (Tuple2<Integer, Tuple2<Double, Double>> p : reps) {
            Double crdist = distance(p._2(), point);
            if (crdist < mindist) {
                mindist = crdist;
                mincluster = p._1();
            }
        }
        return mincluster;
    }

    public static ArrayList<Tuple2<Double, Double>> representatives(ArrayList<Tuple2<Double, Double>> points, int numberofrepr) {
        Double x = points.map(Tuple2::_1).reduce(Double::sum) / points.size();
        Double y = points.map(Tuple2::_2).reduce(Double::sum) / points.size();

        Tuple2<Double, Double> mean = new Tuple2<>(x, y);

        Tuple2<Double, Double> maxPoint = points.maxBy(point -> distance(point, mean));

        ArrayList<Tuple2<Double, Double>> replist = new ArrayList<>();
        replist.add(maxPoint);

        for (int i = 1; i < numberofrepr; i++) {
            Tuple2<Double, Double> maxPoint2 = points.maxBy(point -> distance(point, replist.get(replist.size() - 1)));
            replist.add(maxPoint2);
        }

        return shrinkrepr(replist, 0.2);
    }

    public static Double distance(Tuple2<Double, Double> p1, Tuple2<Double, Double> p2) {
        return Math.sqrt(Math.pow(Math.abs(p1._1() - p2._1()), 2) + Math.pow(Math.abs(p1._2() - p2._2()), 2));
    }

    public static ArrayList<Tuple2<Double, Double>> shrinkrepr(ArrayList<Tuple2<Double, Double>> pointssh, Double a) {
        Double meanx2 = pointssh.map(Tuple2::_1).reduce(Double::sum) / pointssh.size();
        Double meany2 = pointssh.map(Tuple2::_2).reduce(Double::sum) / pointssh.size();

        return new ArrayList<>(pointssh.map(point ->
                new Tuple2<>(point._1() + a * (meanx2 - point._1()), point._2() + a * (meany2 - point._2()))).toJavaList());
    }
}
