/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import evaluation.PerformanceMetric;
import java.io.FileReader;
import java.io.IOException;
import multivariate_timeseriesweka.classifiers.MultivariateShapeletTransformClassifier;
import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import multivariate_timeseriesweka.classifiers.NN_DTW_D;
import multivariate_timeseriesweka.classifiers.NN_DTW_I;
import multivariate_timeseriesweka.classifiers.NN_ED_D;
import multivariate_timeseriesweka.classifiers.NN_ED_I;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;



/**
 *
 * @author fax14yxu
 */
public class ProjectMain {
/**
     * @param args the command line arguments
    */
    public static void main(String[] args) throws Exception {
        
    /////////////////////////// Variables \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        // Dataset Dir
        String datasetDir = "";
        // Output Dir
        String outputDir = "";
        // Cluster flag
        boolean clusterF = false;
        // char to decide which data use
        char set;
        // Array with the dataset type used
        String [] dataSetTypes = null;
        String datasetName = null;
        // Standardisation flag
        boolean standardiseF = false;
        // output flag
        boolean outputF = false;
        // String to decide which experiments run 
        String experiment = null;
        // Integer representing exp ID
        int expId = 0;
        // Flag for windwos search
        boolean windowsSearchF = false;
        // Flag for using overall mean in MeanClassifier
        boolean overallStatsF = false;
        
        
        // DT variables
        boolean cvMinNumberPerNode;
        int minNumberStart = 0;
        int minNumberEnd = 0;
        boolean cvNumbFolds;
        int numbFoldsStart = 0;
        int numbFoldsEnd = 0;
        boolean pruningTreeF; 
        
        // Random Forest variable
        boolean cvTreeDepth;
        int startDepth, endDepth;
        boolean cvNumTrees;
        int startTrees, endTrees;
        boolean cvNumFeatures;
        
        // SVM variables
        boolean normaliseF; 
        boolean cvNumFoldsF;
        int startFold, endFold;
        boolean cvCF;
        boolean cvEpsilonF;
        
        // Multi layer perceptron variables
        boolean mlpNormaliseF;
        boolean mlpCvLearnRate;
        boolean mlpCvMomenumRate;
        boolean mlpCvEpoch;
        
        // Variable to convert univariate data to multivariate 
        int numbFeat = 10;
        int numbDate = 23;
        
        
        
        
        if (args.length>0) {
            clusterF = true;
            // Run on cluster
            System.out.println("\t -- GENERAL VARIABLE -- ");
            System.out.println(args.length);
            datasetDir = args[0];
            System.out.println("datasetDir : " + datasetDir);
            outputDir  = args[1];
            System.out.println("OutputDir : " + outputDir);
            set = args[2].charAt(0);
            System.out.println("Set : " + set);
            standardiseF = Boolean.parseBoolean(args[3]);
            System.out.println("standardiseF : " + standardiseF);
            outputF = Boolean.parseBoolean(args[4]);
            System.out.println("OutputF : " + outputF);
            experiment = args[5];
            System.out.println("Experiment : "  + experiment);
            datasetName = args[9];
            System.out.println("datasetName : " + datasetName);
            expId = Integer.parseInt(args[6]);
            System.out.println("ExperimentId : " + expId);
            System.out.println("\t -- DTW Variable -- ");
            windowsSearchF = Boolean.parseBoolean(args[7]);
            System.out.println("Window Search Flag : " + windowsSearchF);
            System.out.println("\t -- Statistical Classifier  Variable -- ");
            overallStatsF = Boolean.parseBoolean(args[8]);
            System.out.println("OverallStats : " + overallStatsF);
            System.out.println("\t -- DT  Variables -- ");
            cvMinNumberPerNode = Boolean.parseBoolean(args[10]);
            System.out.println("cvMinNumberPerNode : " + cvMinNumberPerNode);
            minNumberStart = Integer.parseInt(args[11]);
            minNumberEnd   = Integer.parseInt(args[12]);
            System.out.println("Range of min Number : " + minNumberStart + " - " + minNumberEnd);
            cvNumbFolds = Boolean.parseBoolean(args[13]);
            System.out.println("cvNumbFolds : " + cvNumbFolds);
            numbFoldsStart = Integer.parseInt(args[14]);
            numbFoldsEnd = Integer.parseInt(args[15]);
            System.out.println("Range of min Number : " + numbFoldsStart + " - " + numbFoldsEnd);
            pruningTreeF = Boolean.parseBoolean(args[16]);
            System.out.println("pruningTreeF : " + pruningTreeF);
            System.out.println("\t -- Random Forests  Variables -- ");
            cvTreeDepth = Boolean.parseBoolean(args[17]);
            System.out.println("cvTreeDepth : " + cvTreeDepth);
            startDepth = Integer.parseInt(args[18]);
            endDepth = Integer.parseInt(args[19]);
            System.out.println("Range for trees depth : " + startDepth + " - " + endDepth);
            cvNumTrees = Boolean.parseBoolean(args[20]);
            System.out.println("cvNumTrees : " + cvNumTrees);
            startTrees = Integer.parseInt(args[21]);
            endTrees = Integer.parseInt(args[22]);
            System.out.println("Range for trees depth : " + startTrees + " - " + endTrees);
            cvNumFeatures = Boolean.parseBoolean(args[23]);
            System.out.println("cvNumFeatures : " + cvNumFeatures);
            System.out.println("\t -- SVM  Variables -- ");
            normaliseF = Boolean.parseBoolean(args[24]);
            System.out.println("normaliseF : " + normaliseF);
            cvNumFoldsF = Boolean.parseBoolean(args[25]);
            startFold = Integer.parseInt(args[26]);
            endFold   = Integer.parseInt(args[27]);
            System.out.println("cvNumFoldsF : " + cvNumFoldsF);
            System.out.println("Range for number of folds : " + startFold + " - " + endFold);
            cvCF = Boolean.parseBoolean(args[28]);
            System.out.println("cvC : " + cvCF);
            cvEpsilonF = Boolean.parseBoolean(args[29]);
            System.out.println("cvEpsilonF : " + cvEpsilonF);
            System.out.println("\t -- MLP Variables -- ");
            mlpNormaliseF = Boolean.parseBoolean(args[30]);
            System.out.println("NormaliseF : " +  mlpNormaliseF);
            mlpCvLearnRate = Boolean.parseBoolean(args[31]);
            System.out.println("CvLearningRate : " + mlpCvLearnRate);
            mlpCvMomenumRate = Boolean.parseBoolean(args[32]);
            System.out.println("mlpCvMomenumRate : " + mlpCvMomenumRate);
            mlpCvEpoch = Boolean.parseBoolean(args[33]);
            System.out.println("mlpCvEpoch : " + mlpCvEpoch);
            
            System.out.println("");
            
        } else {
            // Run locally 
            datasetDir = "\\\\ueahome4\\stusci3\\fax14yxu\\data\\Documents\\4th year\\Dissertation\\Data\\";
            outputDir  = "\\\\ueahome4\\stusci3\\fax14yxu\\data\\Documents\\4th year\\Dissertation\\ResultsTemp\\";
            
            //datasetDir = "C:\\Users\\Martino94\\Desktop\\Dissertation\\Data\\";
            //outputDir  = "C:\\Users\\Martino94\\Desktop\\Dissertation\\ResultsTemp\\";
            set = '2';
            //datasetName = "LCdata";
            datasetName = "1000LCdata";
            //datasetName = "1000LCdata_multivariate";
            //datasetName = "1000LCdata_multivariate_SumStats2";
            //datasetName = "binaryLCdata";
            standardiseF = false;
            outputF = true;
            experiment = "NewNNDTW";
            expId = 0;
            windowsSearchF = false;
            overallStatsF = false;
            
            // DT variables
            cvMinNumberPerNode = false;
            minNumberStart = 1;
            minNumberEnd   = 100;
            cvNumbFolds = false;
            numbFoldsStart = 1;
            numbFoldsEnd = 25;
            pruningTreeF = false;
            
            // Random Forest 
            cvTreeDepth = false;
            startDepth = 0;
            endDepth = 20;
            cvNumTrees = false;
            startTrees = 1;
            endTrees = 20;
            cvNumFeatures = true;
            
            // SVM variables
            normaliseF = false;
            cvNumFoldsF = false;
            startFold = 1;
            endFold = 25;
            cvCF = false;
            cvEpsilonF = false;
            
            // Multi layer perceptron variables
            mlpNormaliseF = false;
            mlpCvLearnRate = false;
            mlpCvMomenumRate = false;
            mlpCvEpoch = false;
            
            // Rotation forest variable
            
            
        }       
      
        
        switch(set) {
            case '1' :
                dataSetTypes = new String[]{"LCproblem"};
                break;
            case '2' :
                dataSetTypes = new String[]{"1000InstPerClass_LCdata"};
                break;
            case '3' :
                dataSetTypes = new String[]{"BinaryLC"};
                break;
            default : 
                throw new Exception("No dataset types was selected");
        }
        
        switch (experiment) {
            case "NNED" :
            case "NNDTW":
            case "NewNNDTW":
                NNexperiment(datasetDir,outputDir,dataSetTypes,datasetName,standardiseF,clusterF,outputF,experiment,expId,windowsSearchF);
                break;
            case "MeansC" :
            case "MinC" :
            case "MaxC" :
            case "MinMaxC" :
            case "OptMinMaxC" :
            case "FullMinMaxC" :
            case "StdC" :
            case "StatsEnsembleC" :
            case "OptStatsEnsembleC" :
                StatisticClassifiers(datasetDir,outputDir,dataSetTypes,datasetName,standardiseF,clusterF,outputF,experiment,expId,overallStatsF);
                break;
            case "DT" :
                DTClassifier(datasetDir,outputDir,dataSetTypes,datasetName,
                        standardiseF,clusterF,outputF,experiment,expId,
                        cvMinNumberPerNode,minNumberStart,minNumberEnd,
                        cvNumbFolds,numbFoldsStart,numbFoldsEnd,pruningTreeF);
                break;
            case "RandForest" :
                RandomForestClassifier(datasetDir, outputDir, dataSetTypes, 
                        datasetName, standardiseF, clusterF, outputF, experiment,
                        expId,cvTreeDepth,startDepth,endDepth,
                        cvNumTrees, startTrees, endTrees, cvNumFeatures);
                break;
            case "SVM" :
                SMOClassifier(datasetDir, outputDir, dataSetTypes,
                        datasetName, standardiseF, clusterF, outputF,experiment,
                        expId, normaliseF,cvNumFoldsF, startFold, endFold,
                        cvCF, cvEpsilonF);
                break;
            case "MLP" :
                MLPClassifier(datasetDir, outputDir, dataSetTypes, datasetName,
                        standardiseF, clusterF, outputF, experiment, expId,
                        mlpNormaliseF,mlpCvLearnRate,mlpCvMomenumRate,mlpCvEpoch);
                break;
            case "RotationForest" :
                RotationForestExps(datasetDir,outputDir,dataSetTypes,datasetName,
                        standardiseF,clusterF,outputF,expId);
                break;
            case "multivariateConv" :
                Utilities.multivariateFormatConversion(datasetDir,dataSetTypes[0],"1000InstPerCls_LCdata", numbFeat, numbDate);
                break;
            case "NN_ED_D" :
            case "NN_ED_I" :
            case "NN_DTW_A" :
            case "NN_DTW_D" :
            case "NN_DTW_I" :
            case "MultShapeletTransformC" : 
                multivariateExps(datasetDir,outputDir,dataSetTypes,datasetName,
                        standardiseF,clusterF,outputF,experiment,expId);
                break;
            default :
                throw new Exception("Experiments specified not found");
                       
        }
    }
    
    public static void multivariateExps(String datasetDir, String outputDir,
            String [] datasetTypes,String datasetName, boolean standardiseF,
            boolean clusterF, boolean outputF, String cId, int expId) throws Exception {
        
        for (int i = 0; i < datasetTypes.length; i++) {

            AbstractClassifier c = null;
            if (cId.compareTo("NN_ED_D") == 0) 
                c = new NN_ED_D();
            else if (cId.compareTo("NN_ED_I") == 0) 
                c = new NN_ED_I();
            else if (cId.compareTo("NN_DTW_A") == 0)
                c = new NN_DTW_A();
            else if (cId.compareTo("NN_DTW_D") == 0)
                c= new NN_DTW_D();
            else if (cId.compareTo("NN_DTW_I") == 0)
                c = new NN_DTW_I();
            else if (cId.compareTo("MultShapeletTransformC") == 0)
                c = new MultivariateShapeletTransformClassifier();
            
            Experiments exp = new Experiments(datasetDir, datasetTypes[i],datasetName,outputDir,clusterF,outputF);
            exp.samplingExp(c, cId, expId, standardiseF);
            
        }
 


    }

    /**
     * Method to run experiments with the Nearest Neighbour classifiers
     * @param datasetDir the string representing directory of the datasets
     * @param outputDir string representing directory where store output
     * @param datasetType string representing the type of dataset to use
     * @param standardiseF flag to check if standardisation is on
     * @param cluster flag to check if run on cluster
     * @param outputF flag to check if print output to file 
     * @param expType which experiment type to run (NNED or NNDTW)
     * @param windSearchF
     */
    public static void NNexperiment(String datasetDir, String outputDir, String [] datasetType,
            String datasetName,boolean standardiseF, boolean cluster, 
            boolean outputF, String expType, int expId, boolean windSearchF) throws Exception {
        
        // Iterate through each dataset given
        for (int i = 0; i < datasetType.length; i++) {
            
            NNClassifiers c = null;
            if (expType.compareTo("NNED")==0) {
                c = new ED1NN(standardiseF);
            } else if (expType.compareTo("NNDTW")==0) {
                c  = new DTW1NN(windSearchF,standardiseF);
            } else if (expType.compareTo("NewNNDTW")==0) {
                c  = new NewDTW1NN(windSearchF,standardiseF);
            } else {
                throw new Exception("NN classifier specified is not implemented");
            }
            
            Experiments exp = new Experiments(datasetDir, datasetType[i],datasetName,outputDir, cluster,outputF);
            //exp.runExp(c,expId);
//            exp.samplingExp(c, expId);
            
            if (datasetType.length > 1) {
                exp.runExp(c,expId);
            } else 
                exp.samplingExp(c, expId);
            
        }
        
    }

    private static void StatisticClassifiers(String datasetDir, String outputDir,
            String[] dataSetTypes, String datasetName, boolean standardiseF, boolean clusterF, 
            boolean outputF, String experiment, int expId, boolean overallStatsF) throws ClassIndexMismatchException, Exception {
        
        // Iterate through each dataset given
        for (int i = 0; i < dataSetTypes.length; i++) {
            
            // Hard coded the number of classes, the number of features and
            // the number of days
            StatsClassifier c = null;
            if (experiment.compareTo("MeansC")==0)
                c = new MeanClassifiers(9, 10, 23,overallStatsF,standardiseF);
            else if(experiment.compareTo("MaxC")==0)
                c = new MaxClassifier(9,10,23,overallStatsF,standardiseF);
            else if (experiment.compareTo("MinC") == 0) 
                c = new MinClassifiers(9, 10, 23, overallStatsF,standardiseF);
            else if (experiment.compareTo("MinMaxC") == 0)
                c = new MinMaxClassifier(9, 10, 23, overallStatsF,standardiseF);
            else if (experiment.compareTo("OptMinMaxC")==0) {
                MinClassifiers minC = new MinClassifiers(9, 10, 23, true, true);
                MaxClassifier  maxC = new MaxClassifier(9, 10, 23, false, true );
                c = new MinMaxClassifier(minC, maxC);
            } else if (experiment.compareTo("FullMinMaxC")==0) {
                MinClassifiers minC = new MinClassifiers(9, 10, 23, false, false);
                MinClassifiers minC2 = new MinClassifiers(9, 10, 23, false, true);
                MinClassifiers minC3 = new MinClassifiers(9, 10, 23, true, false);
                MinClassifiers minC4 = new MinClassifiers(9, 10, 23, true, true);
                MaxClassifier maxC  = new MaxClassifier(9, 10, 23, false, false);
                MaxClassifier maxC2 = new MaxClassifier(9, 10, 23, false, true);
                MaxClassifier maxC3 = new MaxClassifier(9, 10, 23, true, false);
                MaxClassifier maxC4 = new MaxClassifier(9, 10, 23, true, true);
                StatsClassifier [] ensamble = new StatsClassifier[]{minC,minC2,minC3,minC4,maxC,maxC2,maxC3,maxC4};
                c = new MinMaxClassifier(ensamble);
            } else if (experiment.compareTo("StdC")==0) 
                c = new StandardDeviationClassifier(9, 10, 23, overallStatsF, standardiseF);
            else if (experiment.compareTo("StatsEnsembleC")==0) 
                c = new StatsEnsembleClassifier(9, 10, 23, overallStatsF, standardiseF);
            else if (experiment.compareTo("OptStatsEnsembleC")==0) {
                MeanClassifiers meanC = new MeanClassifiers(9, 10, 23, false, true);
                MinMaxClassifier minMaxC = new MinMaxClassifier(9, 10, 23, false, true);
                StandardDeviationClassifier stdC = new StandardDeviationClassifier(9, 10, 23, false, true);
                c = new StatsEnsembleClassifier(9, 10, 23, overallStatsF, standardiseF, new StatsClassifier[]{meanC,minMaxC,stdC});//,minMaxC,stdC});
            }
            Experiments exp = new Experiments(datasetDir, dataSetTypes[i],datasetName,outputDir, clusterF, outputF);
            //exp.runExp(c,expId);
//            exp.samplingExp(c, expId);
            

            exp.samplingExp(c, expId);
            
        }
    }

    private static void DTClassifier(String datasetDir, String outputDir, 
            String[] dataSetTypes, String datasetName, 
            boolean standardiseF, boolean clusterF, boolean outputF, 
            String experiment, int expId, boolean cvMinNumberPerNode,
            int minNumberStart, int minNumberEnd, boolean cvNumbFolds,
            int numbFoldsStart,int numbFoldsEnd, boolean pruningTreeF) throws Exception {
        
        for (int i = 0; i <  dataSetTypes.length; i++) {
            J48 dt = new J48();
            dt.setUnpruned(pruningTreeF);
            Experiments exp = new Experiments(datasetDir, dataSetTypes[i],datasetName,outputDir, clusterF,outputF);
            exp.samplingExp(dt, expId, standardiseF,cvMinNumberPerNode,minNumberStart,minNumberEnd,cvNumbFolds,numbFoldsStart,numbFoldsEnd);
        }
    
    } 
    
    private static void RandomForestClassifier(String datasetDir, String outputDir,
            String [] datasetType, String datasetName, boolean standardiseF,
            boolean clusterF, boolean outputF, String experiment, int expId,
            boolean cvTreeDepth, int startDepth, int endDepth,
            boolean cvNumTrees, int startTrees, int endTrees,
            boolean cvNumFeatures) throws Exception {
        for (int i = 0; i <  datasetType.length; i++) {
            RandomForest rf = new RandomForest();
            Experiments exp = new Experiments(datasetDir, datasetType[i],datasetName,outputDir, clusterF,outputF);
            exp.samplingExp(rf, expId, standardiseF, cvTreeDepth, startDepth, endDepth,cvNumTrees, startTrees, endTrees, cvNumFeatures);
        }
    }
    
    private static void SMOClassifier(String datasetDir, String outputDir,
            String [] datasetType, String datasetName, boolean standardiseF,
            boolean clusterF, boolean outputF, String experiment, int expId,
            boolean normaliseF, boolean cvNumFoldsF, int startFold, int endFold,
            boolean cvC,boolean cvEpsilonF) throws Exception {
        for (int i = 0; i < datasetType.length; i++) { 
            SMO svm = new SMO();
            svm.getFilterType();
            if (normaliseF)
                 svm.setFilterType(new SelectedTag(SMO.FILTER_NORMALIZE, SMO.TAGS_FILTER)); 
            else 
                svm.setFilterType(new SelectedTag(SMO.FILTER_NONE, SMO.TAGS_FILTER));
            Experiments exp = new Experiments(datasetDir,datasetType[i],datasetName,outputDir,clusterF,outputF);
            exp.samplingExp(svm,expId,standardiseF,cvNumFoldsF,startFold,endFold,cvC,cvEpsilonF);
        }
    }
    
    private static void MLPClassifier(String datasetDir, String outputDir,
            String [] datasetType, String datasetName, boolean standardiseF,
            boolean clusterF, boolean outputF, String experiment, int expId,
            boolean normaliseF, boolean mlpCvLearnRate, boolean mlpCvMomenumRate,
            boolean mlpCvEpoch) throws Exception {
        for (int i = 0; i < datasetType.length; i++) {
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setNormalizeAttributes(normaliseF);
                
            Experiments exp = new Experiments(datasetDir, datasetType[i], datasetName, outputDir, clusterF, outputF);
            exp.samplingExp(mlp, expId, standardiseF, mlpCvLearnRate, mlpCvMomenumRate,mlpCvEpoch);
        }
    }

    private static void RotationForestExps(String datasetDir, String outputDir, 
            String[] dataSetTypes, String datasetName, boolean standardiseF, 
            boolean clusterF, boolean outputF, int expId) throws Exception {
        
        for (int i = 0; i < dataSetTypes.length; i++) {
            RotationForest c = new RotationForest();
            Experiments exp = new Experiments(datasetDir,dataSetTypes[i],datasetName,outputDir,clusterF,outputF);
            exp.samplingExp(c, expId,standardiseF);
        }
    }
}