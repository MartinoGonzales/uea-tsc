/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import timeseriesweka.filters.SummaryStats;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author fax14yxu
 */

// Global stats implemented so far mean, standardDeviation, skewness, kurtosis, min, max
public class MultivariateSummaryStats extends SimpleBatchFilter {
    
    private int numbMoments = 4; // What it is and for what is used??
    public void setNumMoments(int newValue) {numbMoments = newValue;}
    
    

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        // Check all the attributes are real valued
        // To check that all attribute are numeric for a multivariate format
        // the only way that I found is to extract a instance and iterate 
        // throught its attribute
        Instance temp = inputFormat.firstInstance();
        // Then retrieve the relational attribute, there is only one so return 
        // the one at 0
        Instances features = temp.relationalValue(0);
        // Get the first features an iterate through it to check that all 
        // attribute are numeric. 
        temp = features.firstInstance();
        for (int i = 0; i < temp.numAttributes(); i++) {
            // Not need to check for the class value since we are looking 
            // at a single feature an not the whole instance
            if (!temp.attribute(i).isNumeric())
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
        //System.out.println(dataRel);
        //Instances temp2 = new Instances("DIOCANE",atts,0);
        //System.out.println(temp2);
        
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
        System.out.println(result);
        
        return result;
    }

    @Override
    protected Instances process(Instances instances) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
	public static void main(String[] args) {
/**Debug code to test SummaryStats generation: **/
            System.out.println("DioCane");
		
            try{
                Instances data = Utilities.loadData("\\\\ueahome4\\stusci3\\fax14yxu\\data\\Documents\\4th year\\Dissertation\\data\\1000InstPerClass_LCdata\\1000LCdata_multivariate");
                MultivariateSummaryStats mst = new MultivariateSummaryStats();
                mst.determineOutputFormat(data);
//                Instances filter=new SummaryStats().process(test);
//               SummaryStats m=new SummaryStats();
//               m.setInputFormat(test);
//               Instances filter=Filter.useFilter(test,m);
//               System.out.println(filter);
            }
            catch(Exception e){
               System.out.println("Exception thrown ="+e);
               e.printStackTrace();
               
            }
        }
}
