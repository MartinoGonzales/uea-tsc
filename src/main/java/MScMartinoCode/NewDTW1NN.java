/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import java.io.Serializable;
import java.time.Duration;
import java.time.Instant;
import timeseriesweka.classifiers.FastWWS.windowSearcher.FastWWS;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author fax14yxu
 */
public class NewDTW1NN extends NNClassifiers implements Serializable {

/////////////////////////////////////// Variable \\\\\\\\\\\\\\\\\\\\\\\\\\\    
    
    // Window Size
    private double [] windowSize;
    // Window flag
    private boolean windowF;
    
//////////////////////////////////// Constructor \\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    /**
     * Constructor 
     * @param windowF boolean variable to set flag for optimal windows search
     */
    public NewDTW1NN(boolean windowF, boolean standardiseF) {
        this.windowSize = null; // set to 0 by default
        this.windowF = windowF;
        this.isStandardise = standardiseF;
        if (this.windowF) {
            this.clsID = "New1NNDTW_W";
        } else {
            this.clsID = "New1NNDTW";
        }
    }
    
    public double [] getWindowsSize() {
        return this.windowSize;
    }
    
    public String getWindowsSizeString() {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < this.windowSize.length; i++) {
            str.append(this.windowSize[i] + " - ");
        }
        return str.toString();
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
            this.windowSize = new double[10];
            for (int feat = 0; feat < 10; feat++) {
                Instances tempData = this.featureSelection(feat, data);
                tempData.numAttributes();
                FastWWS windowSearcher = new FastWWS();
                try {
                    // Build FastWWS
                    //windowSearcher.buildClassifier(data);
                    windowSearcher.buildClassifier(tempData);

                    // Assugb the best windows found
                    this.windowSize[feat] = (windowSearcher.getBestWin() == 0) ? 1 : windowSearcher.getBestWin();
                } catch (Exception e) {
                    System.err.println("ERROR : " + e.getMessage());
                }
            }
            
//            System.out.println("Printing Windows Sizes");
//            for (int feat = 0; feat < this.windowSize.length; feat++) {
//                System.out.println("Feat " + feat + " : " + this.windowSize[feat]);
//            }
            
//            FastWWS windowSearcher = new FastWWS();
//            try {
//                // Build FastWWS
//                //windowSearcher.buildClassifier(data);
//                windowSearcher.buildClassifier(this.trainData);
//                // Assugb the best windows found
//                this.windowSize = windowSearcher.getBestWin() == 0 ? 1 : windowSearcher.getBestWin();//(windowSearcher.getBestWin() == 0) ? 1 : windowSearcher.getBestWin();
//            } catch (Exception e) {
//                System.err.println("ERROR : " + e.getMessage());
//            }
        }
    }
    
    @Override
    public double distance(Instance first, Instance second, double abandonValue) throws ClassIndexMismatchException {
        // First we test that the class index is the last of the attributes 
        // in both of the Instance
        if (first.classIndex() != first.numAttributes() -1 || 
                second.classIndex() != second.numAttributes() -1) {
           throw new ClassIndexMismatchException("The class value is not the last of the attributes");
        }
        
        double totDist = 0;
        for (int feat = 0; feat < 10; feat++) {
            Instance firstTemp = this.featureSelection(feat, first);
            Instance secondTemp = this.featureSelection(feat, second);

            totDist += this.featureDist(firstTemp, secondTemp, abandonValue, feat);
        }
        //return totDist;
        return totDist/10;
    }
    
    public double featureDist(Instance first, Instance second, double abandonValue, int featIndex) {
            double minDist; 
            int n = first.numAttributes()-1;
            int m = second.numAttributes()-1;
            double [][] distMatrix = new double[n][m];
            
            if (this.windowF) {
                int ws = (int) this.windowSize[featIndex];
                // Set the boundry element of distance matrix to max value
                int start,end;
                for (int i = 0; i < n; i++) {
                    start = ws < i ? i - ws : 0;
                    end   = /*added the i*/ i + ws + 1 < m ? i + ws + 1 : m;
                    for (int j = start; /*j < n &&*/ j < end ; j++) {
                        distMatrix[i][j] = Double.MAX_VALUE;
                    }
                }
                // Initialise first element of matrix
                distMatrix[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
                // If already exceeds the threshold then return the maximum value
                if (distMatrix[0][0] > abandonValue)
                    return Double.MAX_VALUE;
                // Initialise first row of matrix
                for (int i = 1; i < ws && i < n; i++) {
                    double dist = (first.value(0) - second.value(i)) * (first.value(0) - second.value(i));
                    distMatrix[0][i] = distMatrix[0][i-1] + dist; 
                }
                // Initialise first colum
                for (int i = 1; i < ws && i < m; i++) {
                    double dist = (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
                    distMatrix[i][0] = distMatrix[i-1][0] + dist;
                }
                // Warp the rest
                for (int i = 1; i < n; i++) {
                    boolean tooBig = true;
                    start = ws < i ? i - ws + 1 : 1;
                    end = i + ws < m ? i + ws : m;
                    for (int j = start; j < end; j++) {
                        minDist = distMatrix[i][j-1];
                        if (distMatrix[i-1][j] < minDist) 
                            minDist = distMatrix[i-1][j];
                        if (distMatrix[i-1][j-1] < minDist) 
                            minDist = distMatrix[i-1][j-1];
                        double distance = (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                        distMatrix[i][j] = minDist + distance; 
                        if (tooBig && distMatrix[i][j]<abandonValue)
                            tooBig = false;
                    }
                    if (tooBig)
                        return Double.MAX_VALUE;
                }
                return distMatrix[n-1][m-1];  
            }
            
            
            
            // Initialise first element of the matrix 
            distMatrix[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
            // If already exceeds the threshold then return the maximum value
            if (distMatrix[0][0] > abandonValue)
                return Double.MAX_VALUE;
            // Initialise first row of the matrix
            for (int i = 1; i < n; i++) {
                double dist = (first.value(0) - second.value(i)) * (first.value(0) - second.value(i));
                distMatrix[0][i] = distMatrix[0][i-1] + dist;
            }
            // Initialise first column of the matrix
            for (int i = 1; i < m; i++) {
                double dist = (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
                distMatrix[i][0] = distMatrix[i-1][0] + dist;
            }
            // Warp the rest
            boolean abandon = true;
            for (int i = 1; i < n; i++) {
                abandon = true;
                for (int j = 1; j < m; j++) {
                    minDist = distMatrix[i][j-1];
                    if (distMatrix[i-1][j] < minDist)
                        minDist = distMatrix[i-1][j];
                    if (distMatrix[i-1][j-1] < minDist)
                        minDist = distMatrix[i-1][j-1];
                    double dist = (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                    distMatrix[i][j] = minDist + dist;
                    if (distMatrix[i][j] > abandonValue)
                        distMatrix[i][j] = Double.MAX_VALUE;
                    else 
                        abandon = false;
                }
                if (abandon)
                    break;
            }
            if (abandon) 
                return Double.MAX_VALUE;
            return distMatrix[n-1][m-1];
    }
    
    // Method to return Instance with just a single features given the index 
    // of the features : from 0 to 9
    public Instance featureSelection(int featIndex, Instance inst) {
        // Divide by 10 as there are 10 features and +1 for the class value
        DenseInstance tempInst = new DenseInstance(inst.numAttributes()/10+1);
        // Add the relevant attribute to the temporary Instance
        for (int j = featIndex; j < inst.numAttributes()-1; j+=10) {
            tempInst.setValue(j/10, inst.value(j));
        }
        // Add class value 
        tempInst.setValue(tempInst.numAttributes()-1, inst.classValue()); 
        return tempInst;
        
    }
    
    public Instances featureSelection(int featIndex, Instances data) {
        Instances tempData = new Instances(data);
        int delCount = 0;
        for (int atrFeat = 0; atrFeat < data.numAttributes(); atrFeat++) {
            if (atrFeat == data.classIndex() || atrFeat == featIndex) {
                featIndex += 10;
            } else {
                tempData.deleteAttributeAt(atrFeat - delCount);
                delCount++;
            }
        }
        // loop to check it is working
//        for (int i = 0; i < tempData.numAttributes(); i++) {
//            System.out.println(tempData.attribute(i).name());
//        }
//        tempData.deleteAttributeAt(featIndex);
//        for (Instance inst : data) {
//            Instance tempInst = this.featureSelection(featIndex, inst);
//            tempData.add(tempInst);
//        }
        
        return tempData;
    }
    
    
}
