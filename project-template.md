import java.io.\*;  
import java.util.SplittableRandom;  
import java.lang.Math;  
import java.io.BufferedReader;  
import java.io.File;  
import java.io.FileReader;

public class Tests {

   // Use we use 'static' for all methods to keep things simple, so we can call those methods main

   static void Assert (boolean res) // We use this to test our results \- don't delete or modify\!  
   {  
       if(\!res)  {  
           System.*out*.print("Something went wrong.");  
           System.*exit*(0);  
       }  
   }

   // Copy your vector operations here:  
   static double \[\] mult(double a, double \[\] V) {

       double \[\] result \= new double \[V.length\];  
       for(int i \= 0; i \< V.length; i++)  
           result\[i\] \= a\*V\[i\];  
       return result;  
   }  
   static double \[\] add(double a, double \[\] V) {

       double \[\] result \= new double \[V.length\];

       for(int i \= 0; i \< V.length; i++)  
           result\[i\] \= a \+ V\[i\];  
       return result;

   }  
   static double \[\] sub(double a, double \[\] V) {

       double \[\] result \= new double \[V.length\];

       for(int i \= 0; i \< V.length; i++)  
           result\[i\] \= V\[i\] \- a;

       return result;  
   }

   static double \[\] add(double \[\] U, double \[\] V) {

       *Assert*(U.length \== V.length);  
       double \[\] result \= new double \[U.length\];

       for(int i \= 0; i \< U.length; i++)  
           result\[i\] \= U\[i\] \+ V\[i\];  
       return result;  
   }  
   static double \[\] sub(double \[\] U, double \[\] V) {

       return *add*(U, *mult*(-1, V));

   }  
   static double dot(double \[\] U, double \[\] V) {

       *Assert*(U.length \== V.length);  
       double result \= 0;  
       for(int i \= 0; i \< U.length; i++)  
           result \+= U\[i\]\*V\[i\];

       return result;

   }

   static int *NumberOfFeatures* \= 15;

   static double\[\] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {

       double\[\] feature \= new double\[*NumberOfFeatures*\];  
       feature\[0\] \= id;  // We use the movie id as a numeric attribute.

       switch (genre) {  
           case "Action": feature\[1\] \= 1; break;  
           case "Fantasy": feature\[2\] \= 1; break;  
           case "Romance": feature\[3\] \= 1; break;  
           case "Sci-Fi": feature\[4\] \= 1; break;  
           case "Adventure": feature\[5\] \= 1; break;  
           case "Horror": feature\[6\] \= 1; break;  
           case "Comedy": feature\[7\] \= 1; break;  
           case "Thriller": feature\[8\] \= 1; break;  
           default: *Assert*(false);  
       }

       //added new features  
       feature\[9\] \= *normalize*(runtime, 91, 128);  
       feature\[10\] \= *normalize*(year, 1900, 2025);  
       feature\[11\] \= *normalize*(imdb, 6.1, 8.45);  
       feature\[12\] \= *normalize*(rt, 67, 92);  
       feature\[13\] \= *normalize*(budget, 20, 161);  
       feature\[14\] \= *normalize*(boxOffice, 31, 504);

       // That is all. We don't use any other attributes for prediction.  
       return feature;  
   }

   static double normalize(double value, double min, double max) {  
       return (value \- min) / (max \- min);  
   }  
   static double\[\] *featureWeights* \= {0, 1.0, 1.1, 0.9, 1.5, 1.5, 1.1, 1.1, 1.0, 0.3, 0.5, 2.4, 1.8, 1.5, 1.2};  
   // We are using the dot product to determine similarity:  
   static double similarity(double\[\] u, double\[\] v) {  
       return *weightedEuclideanDistance*(u, v);  
   }  
   static double weightedEuclideanDistance(double\[\] u, double\[\] v) {  
       double sum \= 0;  
       for (int i \= 0; i \< u.length; i++) {  
           double diff \= u\[i\] \- v\[i\];  
           sum \+= *featureWeights*\[i\] \* Math.*pow*(diff, 2);  
       }  
       return Math.*sqrt*(sum);  
   }

   // We have implemented KNN classifier for the K=1 case only. You are welcome to modify it to support any K

   static int knnClassify(double\[\]\[\] trainingData, int\[\] trainingLabels, double\[\] testFeature, int K) {  
       // Initialize arrays to store the K nearest neighbors and their distances.  
       int\[\] nearestLabels \= new int\[K\];  
       double\[\] nearestDistances \= new double\[K\];

       // Fill distances with the worst possible similarity (large negative value).  
       for (int i \= 0; i \< K; i++) {  
           nearestDistances\[i\] \= \-Double.*MAX\_VALUE*;  
       }

       // Iterate over all training data points.  
       for (int i \= 0; i \< trainingData.length; i++) {  
           // Calculate similarity (or distance) between the test feature and current training data.  
           double currentSimilarity \= *similarity*(testFeature, trainingData\[i\]);

           // Check if the current point should replace any of the K nearest neighbors.  
           for (int j \= 0; j \< K; j++) {  
               if (currentSimilarity \> nearestDistances\[j\]) {  
                   // Shift less similar neighbors down the list.  
                   for (int m \= K \- 1; m \> j; m--) {  
                       nearestDistances\[m\] \= nearestDistances\[m \- 1\];  
                       nearestLabels\[m\] \= nearestLabels\[m \- 1\];  
                   }  
                   // Insert the current point into the K nearest neighbors.  
                   nearestDistances\[j\] \= currentSimilarity;  
                   nearestLabels\[j\] \= trainingLabels\[i\];  
                   break; // Break as we only replace one position.  
               }  
           }  
       }

       // Count occurrences of each label among the K nearest neighbors.  
       int\[\] labelCounts \= new int\[10\]; // Assuming labels are in the range \[0, 9\].  
       for (int label : nearestLabels) {  
           labelCounts\[label\]++;  
       }

       // Find the label with the maximum count (majority label).  
       int majorityLabel \= \-1;  
       int maxCount \= \-1;  
       for (int label \= 0; label \< labelCounts.length; label++) {  
           if (labelCounts\[label\] \> maxCount) {  
               maxCount \= labelCounts\[label\];  
               majorityLabel \= label;  
           }  
       }

       return majorityLabel; // Return the majority label as the prediction.  
   }

   // Naive Bayes Helper Variables  
   static double\[\]\[\] *means*;         // Mean for each feature per label  
   static double\[\]\[\] *variances*;     // Variance for each feature per label  
   static double\[\] *priors*;          // Prior probabilities of each label  
   static int *numLabels* \= 2;        // Number of possible labels ("Like it")

   static void calculateMeanVariance(double\[\]\[\] features, int\[\] labels) {  
       int\[\] labelCounts \= new int\[*numLabels*\];  
       *means* \= new double\[*numLabels*\]\[*NumberOfFeatures*\];  
       *variances* \= new double\[*numLabels*\]\[*NumberOfFeatures*\];

       // Initialize arrays to zero  
       for (int label \= 0; label \< *numLabels*; label++) {  
           for (int feature \= 0; feature \< *NumberOfFeatures*; feature++) {  
               *means*\[label\]\[feature\] \= 0;  
               *variances*\[label\]\[feature\] \= 0;  
           }  
       }

       // Calculate sums for means  
       for (int i \= 0; i \< features.length; i++) {  
           int label \= labels\[i\];  // Gets label of current feature  
           labelCounts\[label\]++;   // Increment count for current label  
           for (int j \= 0; j \< *NumberOfFeatures*; j++) {  
               *means*\[label\]\[j\] \+= features\[i\]\[j\]; // Feature value for the current label  
           }  
       }

       // Calculate means  
       for (int label \= 0; label \< *numLabels*; label++) {  
           for (int j \= 0; j \< *NumberOfFeatures*; j++) {  
               if (labelCounts\[label\] \> 0) { // So no division by 0  
                   *means*\[label\]\[j\] /= labelCounts\[label\]; // Calculate mean  
               }  
           }  
       }

       // Calculate variance  
       for (int i \= 0; i \< features.length; i++) {  
           int label \= labels\[i\]; // gets label of current feature  
           for (int j \= 0; j \< *NumberOfFeatures*; j++) {  
               double diff \= features\[i\]\[j\] \- *means*\[label\]\[j\]; // Difference from mean  
               *variances*\[label\]\[j\] \+= diff \* diff; // Accumulate squared difference  
           }  
       }

       // Finalize variance calculation  
       for (int label \= 0; label \< *numLabels*; label++) {  
           for (int j \= 0; j \< *NumberOfFeatures*; j++) {  
               if (labelCounts\[label\] \> 0) { // So no division by 0  
                   *variances*\[label\]\[j\] /= labelCounts\[label\]; // Calculate variance  
               }  
           }  
       }

       // Calculate priors  
       *priors* \= new double\[*numLabels*\]; // Initialize priors array  
       for (int label \= 0; label \< *numLabels*; label++) {  
           *priors*\[label\] \= (double) labelCounts\[label\] / features.length; // P(label) \= count(label) / total feature  
       }  
   }

   // Naive Bayes Prediction  
   static int predictNaiveBayes(double\[\] feature) {

       double\[\] posteriors \= new double\[*numLabels*\]; // Array to store posterior probabilities for each label

       // Calculate probability for each label  
       for (int label \= 0; label \< *numLabels*; label++) {

           posteriors\[label\] \= Math.*log*(*priors*\[label\]); // Start with log of prior probability for numerical stability

           for (int j \= 0; j \< *NumberOfFeatures*; j++) {

               double diff \= feature\[j\] \- *means*\[label\]\[j\]; // Difference from mean  
               double variance \= *variances*\[label\]\[j\]; // Variance for the current feature and label

               if (variance \> 0) { // So no division by 0

                   posteriors\[label\] \+= \-0.5 \* Math.*log*(2 \* Math.*PI* \* variance) \- (diff \* diff) / (2 \* variance); //Constant term of Gaussian \- Exponent term of Gaussian

               }  
           }  
       }

       // Find the label with the highest posterior probability  
       int bestLabel \= 0;  
       for (int label \= 1; label \< *numLabels*; label++) {  
           if (posteriors\[label\] \> posteriors\[bestLabel\]) {  
               bestLabel \= label;  
           }  
       }

       return bestLabel;  
   }

   static void loadData(String filePath, double\[\]\[\] dataFeatures, int\[\] dataLabels) throws IOException {  
       try (BufferedReader br \= new BufferedReader(new FileReader(filePath))) {  
           String line;  
           int idx \= 0;  
           br.readLine(); // skip header line  
           while ((line \= br.readLine()) \!= null) {  
               String\[\] values \= line.split(",");  
               //MovieID, Title, Genre, Year, Director, Lead Actor, RT (%),IMDB, Box Office Revenue (in million $), Budget, Runtime, Like it  
               double id \= Double.*parseDouble*(values\[0\]);  
               String genre \= values\[2\];  
               double runtime \= Double.*parseDouble*(values\[10\]);  
               double year \= Double.*parseDouble*(values\[3\]);  
               double imdb \= Double.*parseDouble*(values\[7\]);  
               double rt \= Double.*parseDouble*(values\[6\]);  
               double budget \= Double.*parseDouble*(values\[9\]);  
               double boxOffice \= Double.*parseDouble*(values\[8\]);

               dataFeatures\[idx\] \= *toFeatureVector*(id, genre, runtime, year, imdb, rt, budget, boxOffice);  
               dataLabels\[idx\] \= Integer.*parseInt*(values\[11\]); // Assuming the label is the last column and is numeric  
               idx++;  
           }  
       }  
   }

   public static void main(String\[\] args) {

       double\[\]\[\] trainingData \= new double\[100\]\[\];  
       int\[\] trainingLabels \= new int\[100\];  
       double\[\]\[\] testingData \= new double\[100\]\[\];  
       int\[\] testingLabels \= new int\[100\];  
       try {  
           // You may need to change the path:  
           *loadData*("G:\\\\Other computers\\\\My laptop\\\\Uni\\\\Year2\\\\CS259\\\\Group Assignment\\\\group proj\\\\training-set.csv", trainingData, trainingLabels);  
           *loadData*("G:\\\\Other computers\\\\My laptop\\\\Uni\\\\Year2\\\\CS259\\\\Group Assignment\\\\group proj\\\\testing-set.csv", testingData, testingLabels);  
       }  
       catch (IOException e) {  
           System.*out*.println("Error reading data files: " \+ e.getMessage());  
           return;  
       }

       // KNN  
       System.*out*.println("Model 1 knn Classifier:");

       // Compute accuracy on the testing set  
       int correctPredictions \= 0;

       int K \= 3; // Setting K to 3 for best accuracy

       // Loop through each testing sample  
       for (int i \= 0; i \< testingData.length; i++) {  
           // Get the predicted label using the KNN classifier  
           int predictedLabel \= *knnClassify*(trainingData, trainingLabels, testingData\[i\], K);

           // Compare the predicted label with the actual label  
           if (predictedLabel \== testingLabels\[i\]) {  
               correctPredictions++;  
           }  
       }

       double accuracy \= (double) correctPredictions / testingData.length \* 100;

       System.*out*.printf("A: %.2f%%\\n", accuracy);

       // Naive Bayes  
       System.*out*.println("\\nModel 2 Naive Bayes Classifier:");

       *calculateMeanVariance*(trainingData, trainingLabels);

       correctPredictions \= 0;  
       for (int i \= 0; i \< testingData.length; i++) {  
           int predictedLabel \= *predictNaiveBayes*(testingData\[i\]);  
           if (predictedLabel \== testingLabels\[i\]) {  
               correctPredictions++;  
           }  
       }  
       accuracy \= (double) correctPredictions / testingData.length \* 100;  
       System.*out*.printf("A: %.2f%%\\n", accuracy);  
   }  
}

