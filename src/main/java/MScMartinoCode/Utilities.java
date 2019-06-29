/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import fileIO.OutFile;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author fax14yxu
 */
public class Utilities {
    
  public static Instances loadData(String dataPath) {
        Instances data = null;
        FileReader r;
        
        try {

            r = new FileReader(dataPath + ".arff");
            data = new Instances(r);
            data.setClassIndex(data.numAttributes()-1);
            
        } catch (IOException e) {
            
            System.out.println("Cannot open the file at path " + dataPath + "\n");
            System.out.println("Exception Thrown" + e.getMessage());
            System.exit(0);
            
        }
        return data;
    }
    /**
     * Method to print a confusion matrix to the console
     * @param matrix the confusion matrix to print 
     */
    public static void printConfusionMatrix(int [][] matrix, OutFile outF) {
        if (outF == null) {
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix.length; j++) {
                    System.out.print("[ " + matrix[i][j] + " ]\t");
                }
                System.out.println();
            }
        } else {
            for (int i = 0; i < matrix.length; i++) {
                StringBuilder str = new StringBuilder();
                for (int j = 0; j < matrix.length; j++) {
                    str.append(matrix[i][j] + ",");
                }
                outF.writeLine(str.toString());
            }
        }
    }
    
    public static Instances subSet(Instances data,int classIndex) {
        Instances temp = new Instances(data, 0);
        for (Instance inst : data) {
            if ((int)inst.classValue() == classIndex) 
                temp.add(inst);
        }
        return temp;
    }
    
    public static class StandardiseDataset {
      
      private double [] means;
      private double [] std;
      
      public StandardiseDataset(Instances data) {
          // Initialise Variable to store means of each attributes
          means = new double[data.numAttributes()-1];
          
          // calculate mean of each attributes
          for (Instance inst : data){
              for (int atr = 0; atr < inst.numAttributes()-1; atr++) {
                  means[atr] += inst.value(atr);
              }
          }
          for (int atr = 0; atr < data.numAttributes()-1; atr++) {
              means[atr] /= data.numInstances();
          }
          
          // Initialise variable to store standard deviation 
          std = new double[data.numAttributes()-1];
          // Calculate deviation 
          for (Instance inst : data) {
              for (int atr = 0; atr < inst.numAttributes()-1; atr++) {
                  std[atr] += ( (inst.value(atr) - means[atr]) * (inst.value(atr) - means[atr]) ) ;
              }
          }
          // Calculate the variance
          for (int atr = 0; atr < data.numAttributes()-1; atr++) {
              std[atr] /= data.numInstances();
          }
          // Calculate the standard deviation
          for (int atr = 0; atr < data.numAttributes()-1; atr++) {
              std[atr] = Math.sqrt(std[atr]);
          }
      }
      
      public Instances standardiseInstances(Instances data) {
          
          // Standardise the data
          Instances temp  = new Instances(data);
          int count = 0;
          for (Instance inst : data) {
              for (int atr = 0; atr < inst.numAttributes()-1; atr++) {
                    double standardiseValue = (inst.value(atr) - means[atr]) / std[atr];
                    temp.get(count).setValue(atr, standardiseValue);
              }
              count++;
          }
          return temp;
      }
      
      public Instance standardiseInstance(Instance inst) {
        Instance temp = inst;
        
        for (int atr = 0; atr < inst.numAttributes()-1; atr++) {
            
            double standardisedValue;
            
            //if (std[atr] == 0) {
            //    standardisedValue = 0;
                
            //} else {
                standardisedValue = (inst.value(atr) - means[atr]) / std[atr];
                
            //}
            
            temp.setValue(atr, standardisedValue);
        }
        return temp;
      }
      
  }
    
    // Method to convert univariate to multivariate format. 
    // The file is given as a csv file and will print their multivariate 
    // representation to a txt file. 
    public static void multivariateFormatConversion(String dataDir, String dataType, String fileName, int numbFeatures, int numbDates) {
  
      try {
            // Read the csv file
            BufferedReader br = new BufferedReader(new FileReader(dataDir + "\\" + dataType + "\\" + fileName + ".csv"));
            // Ignore the first line as there is the heading of the attribute
            br.readLine();
            
            // Create the text file where to write the multivariate format
            OutFile outF = null;
            outF = new OutFile(dataDir + "\\" + dataType + "\\" +  "multivariateFormat" + ".txt");
            
            // Start reading file line by line
            String line;
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] temp = line.split(",");
                
                // Start with ' since multivariate ARFF file instances start with '
                StringBuilder sb = new StringBuilder("'");
                for (int feat = 0; feat < numbFeatures; feat++) {
                    for (int date = 0; date < numbDates; date++) {
                        sb.append(temp[feat+(date*numbFeatures)]).append(",");
                    }
                    // Remove last ,
                    sb.deleteCharAt(sb.length() - 1);
                    // Add the \n since represent the start of next feature
                    sb.append("\\n");
                }
                // Remove the last \n
                //sb.deleteCharAt(sb.length() - 1);
                sb.delete(sb.length() - 2, sb.length());
                // Add the class value
                sb.append("',").append(temp[temp.length-1]);
                outF.writeLine(sb.toString());
            }
            outF.closeFile();
      } catch (FileNotFoundException ex) {
          System.out.println("ERROR : " + ex.getMessage());
      } catch (IOException ex) {
          System.out.println("ERROR : " + ex.getMessage());
      }
        
        
    }
}
