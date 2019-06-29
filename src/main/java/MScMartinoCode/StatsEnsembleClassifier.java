/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import java.io.Serializable;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author fax14yxu
 */
public class StatsEnsembleClassifier extends StatsClassifier implements Serializable{

    StatsClassifier [] ensemble;
    
    public StatsEnsembleClassifier(int noClasses, int noFeat, int noDays,
            boolean overallF, boolean standardiseF) {
        if (overallF) 
            this.clsID = "StatsEnsembleC_Overall";
        else 
            this.clsID = "StatsEnsembleC";
        this.noFeat = noFeat;
        this.noTimeStamp = noDays;
        this.isOverall = overallF;
        this.isStandardise = standardiseF;
        MeanClassifiers meanC = new MeanClassifiers(9, 10, 23,overallF,standardiseF);
        MinMaxClassifier minMaxC = new MinMaxClassifier(9, 10, 23,overallF,standardiseF);
        StandardDeviationClassifier stdC = new StandardDeviationClassifier(9, 10, 23,overallF,standardiseF);
        this.ensemble = new StatsClassifier[]{meanC,minMaxC,stdC};
    }
    
    public StatsEnsembleClassifier(int noClasses, int noFeat, int noDays, 
            boolean overallF, boolean standardiseF, StatsClassifier [] classifiers) {
        this.clsID = "OptimalStatsEnsembleC";
        this.noFeat = noFeat;
        this.noTimeStamp = noDays;
        this.isOverall = overallF;
        this.isStandardise = standardiseF;
        this.ensemble = classifiers;
    }
    
    @Override
    public double[] getClassDistances(Instance inst) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
//        Instances temp = new Instances(data);
//        if (this.isStandardise) {
//            this.standardise = new Utilities.StandardiseDataset(data);
//            temp = new Instances(this.standardise.standardiseInstances(data));
//        }
        // Iterate through each classifiers in the ensamble and build it
        for (StatsClassifier c : this.ensemble) {
            c.buildClassifier(data);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // Create array with the distances between instance and each class
        // by itarating through each classifier in the ensable and calculating 
        // its distances for each class
        double [] dists = new double[instance.numClasses()];
        for (StatsClassifier c : ensemble) {
            double [] temp = c.getClassDistances(instance);
            for (int i = 0; i < temp.length; i++) {
                dists[i] +=  temp[i];
            }
        }
        int minIndex = 0;
        for (int i = 1; i < dists.length; i++) {
            if (dists[i] < dists[minIndex])
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
