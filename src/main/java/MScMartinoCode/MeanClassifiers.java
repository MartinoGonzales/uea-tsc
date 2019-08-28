/*
 * Implementation of Mean classifier
 */
package MScMartinoCode;

import MScMartinoCode.Utilities.StandardiseDataset;
import java.io.Serializable;
import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Stats;
import weka.core.converters.ConverterUtils.DataSink;

/**
 *
 * @author Martino Gonzales
 */
public class MeanClassifiers extends StatsClassifier implements Serializable {
    
////////////////////////////////// Variables \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    // Matrix where means are going to be stored
    private double [][][] meansMatrix; 


////////////////////////////////// Constructor \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    
    
    public MeanClassifiers (int numberClasses, int noFeat, int noDays, boolean overallMeanF, boolean standardiseF) {
        if (overallMeanF)
            this.clsID = "MeanClass_Overall";
        else 
            this.clsID = "MeanClass";
        this.noFeat = noFeat;
        this.noTimeStamp = noDays;
        this.isOverall = overallMeanF;
        this.isStandardise = standardiseF;
        this.meansMatrix = new double[numberClasses][][];
        for (int c= 0; c < meansMatrix.length; c++) {
            if (isOverall) 
                this.meansMatrix[c] = new double[noFeat][1];
            else
                this.meansMatrix[c] = new double[noFeat][noDays];
        }
    }
    

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Instances temp = new Instances(data);
        if (this.isStandardise) {
            this.standardise = new StandardiseDataset(data);
            temp = new Instances(this.standardise.standardiseInstances(data));
        }
        
        // Iterate through each class and get subset of data
        // And then calculate the mean for each attributes over each day
        // and store it in matrix 
        for (int cls = 0; cls < temp.numClasses(); cls++) {
            Instances subSet = Utilities.subSet(temp, cls);
            //DataSink.write("\\\\ueahome4\\stusci3\\fax14yxu\\data\\Documents\\4th year\\Dissertation\\data\\Temp.arff", subSet);
            for (int atr = 0; atr < subSet.numAttributes()-1; atr++) {
                int featIndex = atr % this.noFeat;
                int dayIndex  = atr / this.noFeat;

                double mean = subSet.meanOrMode(atr);
                if (this.isOverall)
                    this.meansMatrix[cls][featIndex][0] += mean;
                else 
                    this.meansMatrix[cls][featIndex][dayIndex] = mean;
            }
        }
        if (this.isOverall) {
            for (int cls = 0; cls < temp.numClasses(); cls++) {
                for (int feat = 0; feat < this.meansMatrix[cls].length; feat++) {
                    this.meansMatrix[cls][feat][0] /= this.noTimeStamp;
                }
            }
        }
    }

    
        @Override
    public double classifyInstance(Instance instance) throws Exception {
        
        double [] meanDist = this.getClassDistances(instance);
        
        // Find the minimum distance to a class
        int minIndex = 0;
        for (int i = 1; i < meanDist.length; i++) {
            if (meanDist[i] < meanDist[minIndex])
                minIndex = i;
        }
        return minIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] getClassDistances(Instance inst) {
        DenseInstance tempInst = new DenseInstance(inst.numAttributes());
        // Add the relevant attribute to the temporary Instance
        for (int j = 0; j < inst.numAttributes()-1; j++) {
            tempInst.setValue(j, inst.value(j));
        }
        tempInst.setValue(inst.classIndex(), inst.classValue()); 
        if (this.isStandardise)
            tempInst = (DenseInstance) this.standardise.standardiseInstance(tempInst);
        
        // Iterate throught each Attribute of the instance
        double [] meanDist = new double[inst.numClasses()];
        for (int atr = 0; atr < inst.numAttributes()-1; atr++) {
            int featIndex = atr % this.noFeat;
            int dayIndex  = atr / this.noFeat;
            // For each attribute calculate the distance to each class
            for (int cls = 0; cls < inst.numClasses(); cls++) {
                double dist;
                if (this.isOverall)
                    dist = (tempInst.value(atr) - this.meansMatrix[cls][featIndex][0]) * (tempInst.value(atr) - this.meansMatrix[cls][featIndex][0]);
                else 
                    dist = (tempInst.value(atr) - this.meansMatrix[cls][featIndex][dayIndex]) * (tempInst.value(atr) - this.meansMatrix[cls][featIndex][dayIndex]);
                meanDist[cls] += dist;
            }
        }
        return meanDist;
    }
}
