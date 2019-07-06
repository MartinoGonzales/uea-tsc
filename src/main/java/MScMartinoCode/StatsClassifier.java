/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 *
 * @author fax14yxu
 */
public abstract class StatsClassifier implements Classifier {
    
    protected String clsID;
    // Variable to store the number of features for the problem 
    protected int noFeat;
    // Variable to store the number of timestamps over which the features are 
    // taken
    protected int noTimeStamp;
    // Boolean Flag to check if overall of stats need to be used 
    protected boolean isOverall;
    protected Utilities.StandardiseDataset standardise;
    protected boolean isStandardise;
    
    public String getClsID() {
        return this.clsID;
    }
    
    
    public boolean getIsOverall() {
        return this.isOverall;
    }
    
    public boolean getIsStandardised() {
        return this.isStandardise;
    }
    
    public abstract double [] getClassDistances(Instance inst);
}