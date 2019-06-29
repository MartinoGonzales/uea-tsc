/*
 * Class implementing a 1-Nearest Neighbour classifiers using Euclidean Distance
 * and an Early abandon mecanism 
 */
package MScMartinoCode;

import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author fax14yxu
 */
public class ED1NN extends NNClassifiers implements Serializable {
    
    ///////////////////////////////// Constructor \\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    /**
     * Default constructor. 
     */
    public ED1NN(boolean standardiseF) {
        this.clsID = "1NN_ED";
        this.isStandardise = standardiseF;
    }

    
    @Override
    public double distance(Instance first, Instance second, double abandonValue) throws ClassIndexMismatchException {
        // Calculate distance between two instances using Euclidean distance 
        
        // test if class value is the last attribute in the Instances
        if ((first.classIndex() != first.numAttributes()-1) || (second.classIndex() != second.numAttributes()-1))
            throw new ClassIndexMismatchException("The class value is not the last of the attributes");
    
        // Initialise distance
        double dist = 0;
        // Iterate through each attribute and calculate total dist between instances
        for (int i = 0; i < first.numAttributes() - 1; i++) {
            // In case partial sum exceed abandonValue return maximum distance possible
            if (dist > abandonValue) 
               return Double.MAX_VALUE;
            
            // Calculate ED
            dist += (first.value(i) - second.value(i)) * (first.value(i) - second.value(i));
        }
        // Return dist
        return dist;
    }

    
    
    
}
