/*
 * Implementation of Nearest Neighbour classifier with DTW distance metric
 */
package MScMartinoCode;

import java.io.Serializable;
import timeseriesweka.classifiers.FastWWS.windowSearcher.FastWWS;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Martino Gonzales
 */
public class DTW1NN extends NNClassifiers implements Serializable  {

/////////////////////////////////////// Variable \\\\\\\\\\\\\\\\\\\\\\\\\\\    
    
    // Window Size
    private double windowSize;
    // Window flag
    private boolean windowF;
    
    
    
//////////////////////////////////// Constructor \\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    /**
     * Constructor 
     * @param windowF boolean variable to set flag for optimal windows search
     */
    public DTW1NN(boolean windowF, boolean standardiseF) {
        this.windowSize = 0; // set to 0 by default
        this.windowF = windowF;
        this.isStandardise = standardiseF;
        if (this.windowF) {
            this.clsID = "1NNDTW_W";
        } else {
            this.clsID = "1NNDTW";
        }
    }
    
    public double getWindowSize() {
        return this.windowSize;
    }
    
///////////////////////////////// Methods \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    @Override
    public void buildClassifier(Instances data) {
        
        if (this.isStandardise) {
            this.standardise = new Utilities.StandardiseDataset(data);
            this.trainData = new Instances(this.standardise.standardiseInstances(data));
        }
        else {
            this.trainData = new Instances(data);    
        }
        
        // Search for hte best warping windows
        if(this.windowF) {
            // Intitialise FastWWS instances. Not my class
            FastWWS windowSearcher = new FastWWS();
            try {
                // Build FastWWS
                windowSearcher.buildClassifier(this.trainData);
                // Assugb the best windows found
                this.windowSize = windowSearcher.getBestWin();
            } catch (Exception e) {
                System.err.println("ERROR : " + e.getMessage());
            }
        }
    }
    

    @Override
    public double distance(Instance first, Instance second,double abandonValue) throws ClassIndexMismatchException {

        // First we test that the class index is the last of the attributes 
        // in both of the Instance
        if (first.classIndex() != first.numAttributes() -1 || 
                second.classIndex() != second.numAttributes() -1) {
           throw new ClassIndexMismatchException("The class value is not the last of the attributes");
        }
        
        // Initialise variable needed
        double minDist;
        int n = first.numAttributes()-1;
        int m = second.numAttributes()-1;
        double[][] matrix = new double[n][m];
        
        if (this.windowF) {
            // In case it use a wapring windows
            // Get the windows
            int windowSize = (int) this.windowSize;

            // Set the boundry element of the windows to the max value
            int start, end;
            for (int i = 0; i < n; i++) {
                start = windowSize < i ? i - windowSize : 0;
                end   = /*added the i*/i + windowSize + 1 < m ? i + windowSize + 1 : m;
                for (int j = start; /*j < n && */j < end; j++) {
                    matrix[i][j] = Double.MAX_VALUE;
                }
            }
            
            // Initialise first element of the matrix
            matrix[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
            
            // If alrady exceeds the threshold then return the maximum value
            if (matrix[0][0] > abandonValue) {
                return Double.MAX_VALUE;
            }
            
            // Initialise first row of the matrix
            for (int i = 1; i < windowSize && i < n; i++) {
                double distance = (first.value(0) - second.value(i)) * (first.value(0) - second.value(i));
                matrix[0][i] = matrix[0][i-1] + distance;
            }
            
            // Initialise first column of the matrix
            for (int i = 1; i < windowSize && i < m; i++) {
                double distance = (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
                matrix[i][0] = distance + matrix[i-1][0];
            }
            // Warp the rest
            for (int i = 1; i < n; i++) {
                boolean tooBig = true;
                start = windowSize < i ? i - windowSize + 1 : 1;
                end = i+windowSize<m?i+windowSize:m;
                for (int j = start; j < end; j++) {
                    minDist = matrix[i][j-1];
                    if (matrix[i-1][j] < minDist) 
                        minDist = matrix[i-1][j];
                    if (matrix[i-1][j-1] < minDist) 
                        minDist = matrix[i-1][j-1];
                    double distance = (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                    matrix[i][j] = minDist + distance; 
                    if (tooBig && matrix[i][j]<abandonValue)
                        tooBig = false;
                }
                if (tooBig)
                    return Double.MAX_VALUE;
            }
            return matrix[n-1][m-1];
        }
        
            // Initialise first element of the matrix
            matrix[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
            
            // If alrady exceeds the threshold then return the maximum value
            if (matrix[0][0] > abandonValue) {
                return Double.MAX_VALUE;
            }
            
            // Initialise first row of the matrix
            for (int i = 1; i < n; i++) {
                double distance = (first.value(0) - second.value(i)) * (first.value(0) - second.value(i));
                matrix[0][i] = matrix[0][i-1] + distance;
            }
            
            // Initialise first column of the matrix
            for (int i = 1; i < m; i++) {
                double distance = (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
                matrix[i][0] = distance + matrix[i-1][0];
            }
            
            // Warp the rest
            boolean abandon = true;
            for (int i = 1; i < n; i++) {
                abandon = true;
                for (int j = 1; j < m; j++) {
                    minDist = matrix[i][j-1];
                    if (matrix[i-1][j] < minDist) {
                        minDist = matrix[i-1][j];
                    }
                    if (matrix[i-1][j-1] < minDist) {
                        minDist = matrix[i-1][j-1]; 
                    }
                    
                    double distance = (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                    matrix[i][j] = minDist + distance;
                    
                    if (matrix[i][j] > abandonValue) {
                        matrix[i][j] = Double.MAX_VALUE; 
                    } else {
                        abandon = false;  
                    }
                }

                if (abandon) {
                    break;   
                }
            }
                       
            if (abandon) {
                return Double.MAX_VALUE;
            }
            
           return matrix[n-1][m-1];
    }
}
