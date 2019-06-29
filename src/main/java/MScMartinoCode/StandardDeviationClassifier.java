/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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
 * @author fax14yxu
 */
public class StandardDeviationClassifier extends StatsClassifier implements Serializable {

    private double [][][] stdMatrix;
    
    public StandardDeviationClassifier(int noClasses, int noFeat, int noDays, boolean overallStd, boolean standardiseF) {
        if (overallStd) 
            this.clsID = "StdC_Overall";
        else 
            this.clsID = "StdC";
        this.noFeat = noFeat;
        this.noTimeStamp = noDays;
        this.isOverall = overallStd;
        this.isStandardise = standardiseF;
        this.stdMatrix = new double[noClasses][][];
        for (int c = 0; c < this.stdMatrix.length; c++) {
            if(this.isOverall)
                this.stdMatrix[c] = new double[noFeat][1];
            else 
                this.stdMatrix[c] = new double[noFeat][noDays];
        }
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
        double [] stdDists = new double[inst.numClasses()];
        for (int atr = 0; atr < inst.numAttributes()-1; atr++) {
            int featIndex = atr % this.noFeat;
            int dayIndex  = atr / this.noFeat;
            
            for (int cls = 0; cls < inst.numClasses(); cls++) {
                double dist = 0;
                if (this.isOverall)
                    dist = (tempInst.value(atr) - this.stdMatrix[cls][featIndex][0]) * (tempInst.value(atr) - this.stdMatrix[cls][featIndex][0]);
                else 
                    dist = (tempInst.value(atr) - this.stdMatrix[cls][featIndex][dayIndex]) * (tempInst.value(atr) - this.stdMatrix[cls][featIndex][dayIndex]);
                stdDists[cls] += dist;
            }
        }
        return stdDists;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        Instances temp = new Instances(data);
        if (this.isStandardise) {
            this.standardise = new Utilities.StandardiseDataset(data);
            temp = new Instances(this.standardise.standardiseInstances(data));
        }
        
        // Iterate through each class and get subset of data
        // and then calculate the standardad deviation for each attributes over 
        // each day 
        for (int cls = 0; cls < temp.numClasses(); cls++) {
            Instances subSet = Utilities.subSet(temp, cls);

            for (int atr = 0; atr < subSet.numAttributes()-1; atr++) {
                int featIndex = atr % this.noFeat;
                int dayIndex  = atr / this.noFeat;
                AttributeStats atrStats = subSet.attributeStats(atr);
                Stats stats = atrStats.numericStats;
                
                if (this.isOverall) 
                    this.stdMatrix[cls][featIndex][0] += stats.stdDev;
                else 
                    this.stdMatrix[cls][featIndex][dayIndex] = stats.stdDev;
            }  
        }  
        if (this.isOverall) {
            for (int cls = 0; cls < temp.numClasses(); cls++) {
                for (int feat = 0; feat < this.stdMatrix[cls].length; feat++) {
                    this.stdMatrix[cls][feat][0] /= this.noTimeStamp;
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
//        if (this.isStandardise)
//            instance = this.standardise.standardiseInstance(instance);
//        
//        double [] stdDists = new double[instance.numClasses()];
//        for (int atr = 0; atr < instance.numAttributes()-1; atr++) {
//            int featIndex = atr % this.noFeat;
//            int dayIndex  = atr / this.noFeat;
//            
//            for (int cls = 0; cls < instance.numClasses(); cls++) {
//                double dist = 0;
//                if (this.isOverall)
//                    dist = (instance.value(atr) - this.stdMatrix[cls][featIndex][0]) * (instance.value(atr) - this.stdMatrix[cls][featIndex][0]);
//                else 
//                    dist = (instance.value(atr) - this.stdMatrix[cls][featIndex][dayIndex]) * (instance.value(atr) - this.stdMatrix[cls][featIndex][dayIndex]);
//                stdDists[cls] += dist;
//            }
//        }
        double [] stdDists = this.getClassDistances(instance);
        // Find the minimum distance to a class
        int minIndex = 0;
        for (int i = 1; i < stdDists.length; i++) {
            if (stdDists[i] < stdDists[minIndex])
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
    
}
