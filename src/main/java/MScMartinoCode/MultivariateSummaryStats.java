/*
 * Code to create Multivariate summary statistics representation
 */
package MScMartinoCode;

import java.text.ParseException;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.filters.SummaryStats;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author Martino Gonzales
 */

// Global stats implemented so far mean, standardDeviation, skewness, kurtosis, min, max
public class MultivariateSummaryStats extends SimpleBatchFilter {
    
    private int numbStats = 6; 
    public void setNumbStats(int newValue) {numbStats = newValue;}
    
    

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {

        // Check that all the attributes are real valued
        // Since it is a multivariate format the first attribute will held the 
        // relational attribute (index 0) while the class index will be at index 1
        // First of all check that the class index is at the last index of the instance
        if (inputFormat.classIndex() != inputFormat.numAttributes()-1)
            throw new Exception("Class index is not at the end of the Instance");
        // Then get the relational attribute
        Attribute relAtr = inputFormat.attribute(0);
        // Iterate through attributes in the relational attributes and check they are numeric
        for (int i = 0; i < relAtr.relation(0).numAttributes(); i++) {
            if (!relAtr.relation(0).attribute(i).isNumeric())
                throw new Exception("Non Numeric attribute not allowed in SummaryStats");
        }
                
        // Create the ARFF format information 
        // In order to create relational attribute found useful to look at https://waikato.github.io/weka-wiki/creating_arff_file/
        // it seems you specify an instance with some attribute and use it as the relational attribute
        FastVector atts = new FastVector();
        FastVector attsRel = new FastVector();
        String source = inputFormat.relationName();
        // Create numeric attribute inside the relational attribute
        attsRel.addElement(new Attribute(source+"_mean"));
        attsRel.addElement(new Attribute(source+"_std"));
        attsRel.addElement(new Attribute(source+"_skewness"));
        attsRel.addElement(new Attribute(source+"_kurtosis"));
        attsRel.addElement(new Attribute(source+"_min"));
        attsRel.addElement(new Attribute(source+"_max"));
        // Create instance to store the relational attribute
        Instances dataRel = new Instances("FeaturesSummaryStats",attsRel,0);
        // Add it as an normal attribute
        atts.addElement(new Attribute("FeaturesSummaryStats",dataRel,0));
        
        // Set the class values
        if (inputFormat.classIndex() >= 0) {
            // Get the class values as a fast vector 
            Attribute target = inputFormat.attribute(inputFormat.classIndex());
            FastVector values = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) 
                values.addElement(target.value(i));
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),values));  
        }
        Instances result = new Instances("LCdata_SummaryStats",atts,inputFormat.numInstances());
        // Set the class index to enforce that is at the end of the attribute
        if (inputFormat.classIndex() >= 0)
            result.setClassIndex(result.numAttributes()-1);
        
        return result;
    }

    @Override
    protected Instances process(Instances inst) throws Exception {
        Instances output = determineOutputFormat(inst);
        
        // Iterate through Instances
        for (int i = 0; i < inst.numInstances(); i++) {
            // Get the Instances representing the relational attribute
            Instances relInst = inst.instance(i).relationalValue(0);
            // Get the class value of the instance
            int clsVal = (int) inst.instance(i).classValue();
            // Create instances where store the relational Attributes for 
            // each features
            Instances dataRel = new Instances(output.attribute(0).relation(),0);

            // Iterate through each instance in the relational Instances
            for (int j = 0; j < relInst.numInstances(); j++) {
                
                // Get the Instance 
                Instance temp = relInst.instance(j);
                
                // Start calculating the summary stats
                // mean, standardDeviation, skewness, kurtosis, min, max
                double [] summaryStats = new double[numbStats];
                
                // Get series 
                double [] seriesVal = temp.toDoubleArray();
                
                summaryStats[0] = 0; // mean
                summaryStats[1] = 0; // standardDeviation
                summaryStats[2] = 0; // skewness
                summaryStats[3] = 0; // kurtosis
                summaryStats[4] = Double.MAX_VALUE; // min
                summaryStats[5] = Double.MIN_VALUE; // max
                for (int val = 0; val < seriesVal.length; val++) {
                    summaryStats[0] += seriesVal[val];
                    if (seriesVal[val] < summaryStats[4])
                        summaryStats[4] = seriesVal[val];
                    if (seriesVal[val] > summaryStats[5])
                        summaryStats[5] = seriesVal[val]; 

                }
                summaryStats[0] /= seriesVal.length;
                if (i == 42 && summaryStats[0] == 0)
                    System.out.println("");
                
                for (int val = 0; val < seriesVal.length; val++) {
                    summaryStats[1] += (seriesVal[val] - summaryStats[0]) * (seriesVal[val] - summaryStats[0]);
                    summaryStats[2] += (seriesVal[val]-summaryStats[0])*(seriesVal[val]-summaryStats[0])*(seriesVal[val]-summaryStats[0]);
                    summaryStats[3] += (seriesVal[val]-summaryStats[0])*(seriesVal[val]-summaryStats[0])*(seriesVal[val]-summaryStats[0])*(seriesVal[val]-summaryStats[0]);
                }
                summaryStats[1] = summaryStats[1]/(seriesVal.length-1);
                summaryStats[1] = Math.sqrt(summaryStats[1]);
                if (summaryStats[1] != 0) {
                    summaryStats[2] = summaryStats[2]/(summaryStats[1]*summaryStats[1]*summaryStats[1]);
                    summaryStats[2] = summaryStats[2]/seriesVal.length;
                    summaryStats[3] = summaryStats[3]/(summaryStats[1]*summaryStats[1]*summaryStats[1]*summaryStats[1]);
                    summaryStats[3] = summaryStats[3]/seriesVal.length;
                } else {
                    double tempStd = 0.0000000001;
                    summaryStats[2] = summaryStats[2]/(tempStd*tempStd*tempStd);
                    summaryStats[2] = summaryStats[2]/seriesVal.length;
                    summaryStats[3] = summaryStats[3]/(tempStd*tempStd*tempStd*tempStd);
                    summaryStats[3] = summaryStats[3]/seriesVal.length;
                }
 
                // Add summary stats to the relational data
                dataRel.add(new DenseInstance(1.0,summaryStats));

            }
            // Create temporal array where to store the relational attributes
            // and the class value
            double [] temp = new double[output.numAttributes()];
            temp[0] = output.attribute(0).addRelation(dataRel);
            // add the class value
            temp[1] = clsVal;
            output.add(new DenseInstance(1.0,temp));
        }
        return output;
        
    }
    
	public static void main(String[] args) {
/**Debug code to test SummaryStats generation: **/

            try{
                // Instances data = Utilities.loadData("\\\\ueahome4\\stusci3\\fax14yxu\\data\\Documents\\4th year\\Dissertation\\data\\1000InstPerClass_LCdata\\1000LCdata_multivariate");
                Instances data = Utilities.loadData("C:\\Users\\Martino94\\Desktop\\Dissertation\\Data\\1000InstPerClass_LCdata\\1000LCdata_multivariate");
                MultivariateSummaryStats mst = new MultivariateSummaryStats();
                mst.setInputFormat(data);
                
                Instances filter=Filter.useFilter(data,mst);
                System.out.println(filter);
                
                // Save Instances to arr file 
                // DataSink.write("\\\\ueahome4\\stusci3\\fax14yxu\\data\\Documents\\4th year\\Dissertation\\data\\1000InstPerClass_LCdata\\1000LCdata_multivariate_SumStats2.arff", filter);
                DataSink.write("C:\\Users\\Martino94\\Desktop\\Dissertation\\Data\\1000InstPerClass_LCdata\\1000LCdata_multivariate_SumStats2.arff", filter);
            }
            catch(Exception e){
               System.out.println("Exception thrown ="+e);
               e.printStackTrace();
               
            }
        }
}
