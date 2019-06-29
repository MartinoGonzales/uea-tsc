/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import MScMartinoCode.Utilities.StandardiseDataset;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author fax14yxu
 */
public abstract class NNClassifiers implements Classifier {

////////////////////////////////// Variables \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

protected Instances trainData;
protected String clsID; // classifier ID
protected Utilities.StandardiseDataset standardise;
protected boolean isStandardise;


///////////////////////////////// Methods \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    @Override
    public void buildClassifier(Instances data) {
        if (this.isStandardise) {
            this.standardise = new StandardiseDataset(data);
            this.trainData = new Instances(this.standardise.standardiseInstances(data));
        }
        else
            this.trainData = new Instances(data);
    }
    
    public boolean getStandardiseF() {
        return this.isStandardise;
    }
    
    /**
     * Method that return the distance between two instances. It use early 
     * abandon to speed up computation 
     * @param first the first Instance
     * @param second the second Instance
     * @param abandonValue the threshold for the early abandon
     * @return the distance between first and second
     * @throws MScMartinoCode.ClassIndexMismatchException in case class value is not
     *                                                      last attribute
     */
    public abstract double distance(Instance first, Instance second, double abandonValue) throws ClassIndexMismatchException;

    @Override
    public double classifyInstance(Instance inst) throws ClassIndexMismatchException{

        if (this.isStandardise)
            inst = this.standardise.standardiseInstance(inst);
        
        // Initialise the best distance to maximum value
        double bestDist = Double.MAX_VALUE;
        // Initialise array to store distances between given Instance and whole 
        // training set 
        double [] distResults = new double[this.trainData.size()];
        // Iterate through train set and calculate distances
        for (int i = 0; i < this.trainData.size(); i++) {
            // Calculate distance
            double dist = this.distance(inst, this.trainData.get(i), bestDist);
            // store distance
            distResults[i] = dist;
            // Update best distance 
            if (dist < bestDist)
                bestDist = dist;
        }
        
        // Find index of the lowest distance
        int indexSmallest = 0; 
        for (int i = 1; i < distResults.length; i++) 
            indexSmallest = (distResults[i] < distResults[indexSmallest]) ? i : indexSmallest;
        
        // Return predicted class
        return this.trainData.get(indexSmallest).classValue(); 
    }

    /**
     * Calculate the distribution for instance. 
     * Even if 1-NN is not a probabilistic classifier I implemented it anyway. 
     * The class predicted will be 1 all the other 0
     * @param instance the instance 
     * @return the distribution of predicted class for this instance
     * @throws Exception 
     */
    @Override
    public double[] distributionForInstance(Instance inst) throws ClassIndexMismatchException{
        Instance temp = new DenseInstance(inst);
        if (this.isStandardise)
            temp = this.standardise.standardiseInstance(inst);
        
        double distribution[] = new double[temp.numClasses()];
        int prediction = (int) this.classifyInstance(temp);
        distribution[prediction] = 1.0;
        return distribution;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
    
}
