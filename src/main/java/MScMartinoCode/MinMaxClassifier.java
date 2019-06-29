/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import java.io.Serializable;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author fax14yxu
 */
public class MinMaxClassifier extends StatsClassifier implements Serializable {
    
    private MinClassifiers minC;
    private MaxClassifier maxC;
    private StatsClassifier [] ensamble;
    
    
    /**
     * Default classifiers that build same min and max classifiers with the 
     * same parameters
     * @param numberClasses
     * @param noFeat
     * @param noDays
     * @param overallStats 
     */
    public MinMaxClassifier(int numberClasses, int noFeat, int noDays, boolean overallStats,boolean standardiseF) {
        if (overallStats) 
            this.clsID = "MinMaxClass_Overall";
        else 
            this.clsID = "MinMaxClass";
        this.noFeat = noFeat;
        this.noTimeStamp = noDays;
        this.isOverall = overallStats;
        this.isStandardise = standardiseF;
        this.minC = new MinClassifiers(numberClasses, noFeat, noDays, overallStats,standardiseF);
        this.maxC = new MaxClassifier(numberClasses, noFeat, noDays, overallStats,standardiseF);
        this.ensamble = null;
    }
    
    public MinMaxClassifier(MinClassifiers minC, MaxClassifier maxC) {
        this.minC = minC;
        this.maxC = maxC;
        this.clsID = "OptMinMaxC";
        this.ensamble = null;
    }
    
    public MinMaxClassifier(StatsClassifier [] classifiers) {
        this.clsID = "FullMinMaxC";
        this.ensamble = new StatsClassifier[classifiers.length];
        for (int i = 0; i < this.ensamble.length; i++) {
            this.ensamble[i] = classifiers[i];
        }
    }
    

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (this.ensamble == null) {
            this.minC.buildClassifier(data);
            this.maxC.buildClassifier(data);
        } else {
            for (int i = 0; i < this.ensamble.length; i++) {
                this.ensamble[i].buildClassifier(data);
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {   
//        double [] temp = new double[instance.numClasses()];
//        if (this.ensamble == null) {
//            double [] minCdistances = this.minC.getClassDistances(instance);
//            double [] maxCdistances = this.maxC.getClassDistances(instance);
//           
//            for (int cls = 0; cls < temp.length; cls++) {
//                temp [cls] = minCdistances[cls] + maxCdistances[cls];
//            }   
//        } else {
//            for (int i = 0; i < this.ensamble.length; i++) {
//                double [] distances = this.ensamble[i].getClassDistances(instance);
//                for (int cls = 0; cls < temp.length; cls++) {
//                    temp [cls] += distances[cls];
//                }   
//            }
//        }
        double [] temp = this.getClassDistances(instance);
        // Find the minimum distance to a class
        int minIndex = 0;
        for (int i = 1; i < temp.length; i++) {
            if (temp[i] < temp[minIndex])
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

        double [] temp = new double[inst.numClasses()];
        if (this.ensamble == null) {
            double [] minCdistances = this.minC.getClassDistances(inst);
            double [] maxCdistances = this.maxC.getClassDistances(inst);
           
            for (int cls = 0; cls < temp.length; cls++) {
                temp [cls] = minCdistances[cls] + maxCdistances[cls];
            }   
        } else {
            for (int i = 0; i < this.ensamble.length; i++) {
                double [] distances = this.ensamble[i].getClassDistances(inst);
                for (int cls = 0; cls < temp.length; cls++) {
                    temp [cls] += distances[cls];
                }   
            }
        }
        return temp;
    }
    
}
