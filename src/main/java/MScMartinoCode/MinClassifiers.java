/*
 * Implementation for Min classifier
 */
package MScMartinoCode;

import java.io.Serializable;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Stats;

/**
 *
 * @author Martino Gonzales
 */
public class MinClassifiers extends StatsClassifier implements Serializable {
    
    private double [][][] minMatrix;
   
    public MinClassifiers(int numberClasses, int noFeat, int noDays, boolean overallStatsF, boolean standardiseF) {
        if (overallStatsF)
            this.clsID = "MinClass_Overall";
        else 
            this.clsID = "MinClass";
        this.noFeat = noFeat;
        this.noTimeStamp = noDays;
        this.isOverall = overallStatsF;
        this.isStandardise = standardiseF;
        this.minMatrix = new double[numberClasses][][];
        for (int c= 0; c < minMatrix.length; c++) {
            if (isOverall) 
                this.minMatrix[c] = new double[noFeat][1];
            else
                this.minMatrix[c] = new double[noFeat][noDays];
        }
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        Instances temp = new Instances(data);
        if (this.isStandardise) {
            this.standardise = new Utilities.StandardiseDataset(data);
            temp = new Instances(this.standardise.standardiseInstances(data));
        }
        // Iterate through each class and get subset of the data 
        // And then get the minimum for each attributes on each timestamp
        for (int cls = 0; cls < temp.numClasses(); cls++) {
            Instances subSet = Utilities.subSet(temp, cls);
            
            for (int atr = 0; atr < subSet.numAttributes()-1; atr++) {
                int featIndex = atr % this.noFeat;
                int dayIndex  = atr / this.noFeat;
                AttributeStats atrStats =  subSet.attributeStats(atr);
                Stats stats = atrStats.numericStats;
                if (this.isOverall) {
                    if (dayIndex == 0)
                        this.minMatrix[cls][featIndex][0] = stats.min;
                    else 
                        if (stats.min < this.minMatrix[cls][featIndex][0]) 
                            this.minMatrix[cls][featIndex][0] = stats.min; 
                }
                else
                    this.minMatrix[cls][featIndex][dayIndex] = stats.min;
            }
        }
    }
    


    @Override
    public double classifyInstance(Instance instance) throws Exception {

        double [] dists = this.getClassDistances(instance);

        // Find the minimum distance to a class
        int minIndex = 0;
        for (int i = 1; i < dists.length; i++) {
            if (dists[i] < dists[minIndex])
                minIndex = i;
        }
        return minIndex;
    }
    
    public double [] getClassDistances(Instance inst) {
        DenseInstance tempInst = new DenseInstance(inst.numAttributes());
        // Add the relevant attribute to the temporary Instance
        for (int j = 0; j < inst.numAttributes()-1; j++) {
            tempInst.setValue(j, inst.value(j));
        }
        tempInst.setValue(inst.classIndex(), inst.classValue()); 
        
        if (this.isStandardise)
            tempInst = (DenseInstance) this.standardise.standardiseInstance(tempInst);
        
        double [] dists = new double[inst.numClasses()];
        // Iterate through each attribute
        for (int atr = 0; atr < tempInst.numAttributes()-1; atr++) {
            int featIndex = atr % this.noFeat;
            int dayIndex  = atr / this.noFeat;
            
            // For each attribute calculate distance to the min 
            for (int cls = 0; cls < inst.numClasses(); cls++) {
                double dist =0;
                if (this.isOverall)
                    dist = (tempInst.value(atr) - this.minMatrix[cls][featIndex][0]) * (tempInst.value(atr) - this.minMatrix[cls][featIndex][0]);
                else 
                    dist = (tempInst.value(atr) - this.minMatrix[cls][featIndex][dayIndex]) * (tempInst.value(atr) - this.minMatrix[cls][featIndex][dayIndex]);
                dists[cls] += dist;
            }
        }
        return dists;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
