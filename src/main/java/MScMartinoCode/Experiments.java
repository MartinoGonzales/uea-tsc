/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MScMartinoCode;

import fileIO.OutFile;
import java.io.File;
import java.time.Duration;
import java.time.Instant;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
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
public class Experiments {
    
///////////////////////////// Variables \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    // Dir of the dataset
    private String datasetDir;
    // Name of the dataset
    private String dataType;
    private String datasetName; 
    // Dir of where save output
    private String outputDir;
    // Flag to check if run locally or in cluster
    private boolean isCluster;
    // Flag to check if output on console of file
    private boolean isOutputFile;
    
//////////////////////////// Constructor \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    /**
     * Constructor to initialise an Experiments instance given its parameters
     * @param dataDir 
     * @param datasetType
     * @param outDir
     * @param clusterF 
     */
    public Experiments(String dataDir, String datasetType, String datasetName, String outDir, 
            boolean clusterF, boolean outputF) {
        this.datasetDir  = dataDir;
        this.dataType = datasetType;
        this.datasetName = datasetName;
        this.outputDir   = outDir;
        this.isCluster   = clusterF;
        this.isOutputFile = outputF;
    }
    
//////////////////////////////////// Methods \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    public void runExp(NNClassifiers c, int expId) throws ClassIndexMismatchException {
        
        // Load the train and test data
        Instances train = null;
        Instances test  = null;
        if (this.isCluster) {
            // Run on cluster
            train = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/LC_train");
            test  = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/LC_test");
        } else {
            // Run locally
            train = Utilities.loadData(this.datasetDir + "//" + this.dataType + "//LC_train");
            test  = Utilities.loadData(this.datasetDir + "//" + this.dataType + "//LC_test");
        }
        
        
        if (this.isOutputFile) {
            // Print output to file 
            
            // Create and initialise the output file
            File file = null;
            OutFile outF = null;
            
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.dataType);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.dataType + "/" + c.clsID + "_" + expId + ".csv");
            } else {
                // Run locally 
                file = new File(this.outputDir + "\\" + this.dataType);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.dataType + "\\" + c.clsID + "_" + expId + ".csv" );
            }
            
            outF.writeLine(this.dataType + "," + c.clsID + ",test");
            outF.writeLine("noParams");
            
            // Build classifiers and take time 
            Instant start = Instant.now();
            c.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            
            StringBuilder strResults = new StringBuilder();
            
            // Classify test set
            int correctGuess = 0;
            start = Instant.now();
            for (Instance inst : test) {
                double guess = c.classifyInstance(inst);
                if (guess == inst.classValue())
                    correctGuess++;
                
//                // Append class value and predicted one to file 
//                strResults.append(inst.classValue()).append(",").append(guess).append(",,");
//                // Get the distribution for instance and append it to file
//                double [] distribution = c.distributionForInstance(inst);
//                for (int i = 0; i < distribution.length; i++) {
//                    strResults.append(distribution[i]).append(",");
//                }
//                
//                strResults.append("\n");
            }
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Write everything to file 
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.closeFile();
            
            
        } else {
            // Print output to console
            System.out.println("Data Type used : " + this.dataType);
            System.out.println("Classifier ID : " + c.clsID);
            
            System.out.print("Building classifier ...");
            Instant start = Instant.now();
            c.buildClassifier(train);
            Instant end = Instant.now();
            System.out.println("done");
            
            // calculate time taken to build classifier
            long timeTakenBuild = Duration.between(start, end).toMillis();
            
            // Classify test set
            int correctCount = 0;
            System.out.print("Start classifing new data ...");
            start = Instant.now();
            for (Instance inst : test) {
                double guess = c.classifyInstance(inst);
                if (guess == inst.classValue())
                    correctCount++;
            }
            end = Instant.now();
            System.out.println("done");
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            System.out.println("Accuracy : " + (double)correctCount/(double)test.size());
            System.out.println("Time taken to build classifier : " + timeTakenBuild);
            System.out.println("Time taken to classify new data : " + timeTakenClassify);
            System.out.println("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n");
        }
    }
    
    public void samplingExp(NNClassifiers c, int expId) throws ClassIndexMismatchException {
        
        if (this.isOutputFile) {
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" + this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" + this.datasetName);

            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            System.out.println("done");
            String standDir = "NoStandardised";
            if (c.getStandardiseF())
                standDir = "Standardised";
            File file = null; 
            OutFile outF = null;
            OutFile temp1 = null;
            OutFile temp2 = null;
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + c.clsID + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + c.clsID + "/" + standDir + "/" + c.clsID + "_" + expId + ".csv");
                temp1 = new OutFile(this.outputDir + "/" + this.datasetName + "/" + c.clsID + "/" + standDir + "/" + c.clsID + "temp1_" + expId + ".csv");
                temp2 = new OutFile(this.outputDir + "/" + this.datasetName + "/" + c.clsID + "/" + standDir + "/" + c.clsID + "temp2_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + c.clsID + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + c.clsID + "\\" + standDir + "\\" + c.clsID + "_" + expId + ".csv");
                temp1 = new OutFile(this.outputDir + "/" + this.datasetName + "/" + c.clsID + "/" + standDir + "/" + c.clsID + "temp1_" + expId + ".csv");
                temp2 = new OutFile(this.outputDir + "/" + this.datasetName + "/" + c.clsID + "/" + standDir + "/" + c.clsID + "temp2_" + expId + ".csv");
            }
            
            outF.writeLine(this.dataType + "," + c.clsID + ",test");
            outF.writeLine("noParams");
            
            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            c.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();
//            for (Instance inst : test) {
////                System.out.println(tempCount);
////                tempCount++;
//                double guess = c.classifyInstance(inst);
//                if (guess == inst.classValue())
//                    correctGuess++;
//                
////                // Append class value and predicted one to file 
////                strResults.append(inst.classValue()).append(",").append(guess).append(",,");
////                // Get the distribution for instance and append it to file
////                double [] distribution = c.distributionForInstance(inst);
////                for (int i = 0; i < distribution.length; i++) {
////                    strResults.append(distribution[i]).append(",");
////                }
////                
////                strResults.append("\n");
//            }

            for (int inst = 0; inst < test.numInstances(); inst++) {
                temp1.writeDouble(test.get(inst).classValue());
                temp1.writeString(",");
                
                //System.out.println(inst);
                guesses[inst] = (int) c.classifyInstance(test.get(inst));
                temp2.writeDouble(guesses[inst]);
                temp2.writeString(",");
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            
            // Write everything to file 
            if (c.clsID.compareTo("1NNDTW_W")==0) 
                outF.writeLine("Window Size : " + ((DTW1NN)c).getWindowSize());
            else if (c.clsID.compareTo("New1NNDTW_W")==0) 
                outF.writeLine("Window Size : " + ((NewDTW1NN)c).getWindowsSizeString());
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
            temp1.closeFile();
            temp2.closeFile();
        } else {
            // To Do 
        }
    }
    
    public void samplingExp(StatsClassifier c, int expId) throws ClassIndexMismatchException, Exception {

        if (this.isOutputFile) {
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" +  this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" +  this.datasetName);
                
            
            // standardise data
//            if (this.isStandardise) {
//                data = Utilities.StandardiseDataset.standardiseInstances(data);
//            }
//            
            
            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            System.out.println("done");
            String standDir = "NoStandardised";
            if (c.getIsStandardised())
                standDir = "Standardised";
            File file = null; 
            OutFile outF = null;
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + c.getClsID() + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + c.getClsID() + "/" + standDir + "/" + c.getClsID() + "_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + c.getClsID() + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + c.getClsID() + "\\" + standDir + "\\" + c.getClsID() + "_" + expId + ".csv");
            }
            

            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            c.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();
//            for (Instance inst : test) {
////                System.out.println(tempCount);
////                tempCount++;
//                double guess = c.classifyInstance(inst);
//                if (guess == inst.classValue())
//                    correctGuess++;
//                
////                // Append class value and predicted one to file 
////                strResults.append(inst.classValue()).append(",").append(guess).append(",,");
////                // Get the distribution for instance and append it to file
////                double [] distribution = c.distributionForInstance(inst);
////                for (int i = 0; i < distribution.length; i++) {
////                    strResults.append(distribution[i]).append(",");
////                }
////                
////                strResults.append("\n");
//            }

            for (int inst = 0; inst < test.numInstances(); inst++) {
                guesses[inst] = (int) c.classifyInstance(test.get(inst));
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            
            // Write everything to file 
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
            
        } else {
            // To Do 
        }
    }
    
    public void samplingExp(J48 dt, int expId,boolean standardiseF,
            boolean cvMinNumberPerNode, int minNumberStart, int minNumberEnd,
            boolean cvNumbFolds, int numbFoldsStart,int numbFoldsEnd) throws Exception {
        String standDir = (standardiseF) ? "Standardise" : "NoStandardised";
        String cId = "DT";
        if (cvMinNumberPerNode)
            cId += "_cvMinNoNode";
        if (cvNumbFolds)
            cId += "_cvNoFolds";
        if (dt.getUnpruned()) 
            cId += "_Unpruned";
        if (this.isOutputFile) {
            
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" +  this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" +  this.datasetName);
                
            
                
//            if (standardiseF) {
//                standDir = "Standardised";
//                Utilities.StandardiseDataset standInst = new Utilities.StandardiseDataset(data);
//                data = standInst.standardiseInstances(data);
//            }
//            
            
            //Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            //Instances train = temp[0];
            //Instances test  = temp[1];
            System.out.println("done");
 
            File file = null; 
            OutFile outF = null;
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + cId  + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_" + expId + ".csv");
            }
            

            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            // I put standardisation here since in other experiments 
            // it was done inside the classifiers
            if (standardiseF) {
                Utilities.StandardiseDataset standInst = new Utilities.StandardiseDataset(data);
                data = standInst.standardiseInstances(data);
            }
            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            // Carry out a 10-fold cross validation  to set best min number of 
            // object per node if required
            if (cvMinNumberPerNode) {
                dt.setMinNumObj(dtCvMinNumberPerNode(dt,minNumberStart,minNumberEnd,train) + minNumberStart);  
            }
            if (!dt.getUnpruned() && cvNumbFolds) {
                dt.setNumFolds(dtCvNumbOfFolds(dt,numbFoldsStart,numbFoldsEnd,train) + minNumberStart); 
            }
            
            dt.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();

            for (int inst = 0; inst < test.numInstances(); inst++) {
                guesses[inst] = (int) dt.classifyInstance(test.get(inst));
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            
            // Write everything to file 
            if (cvMinNumberPerNode)
                outF.writeLine("MinNumberPerNode," + dt.getMinNumObj());
            if (cvNumbFolds)
                outF.writeLine("Number of Folds," + dt.getNumFolds());
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
            
        } else {
            // To Do 
        }
    }
    
    public void samplingExp(RandomForest rf, int expId, boolean standardiseF,
            boolean cvTreeDepth, int startDepth, int endDepth,
            boolean cvNumTrees,int startTrees, int endTrees, boolean cvNumFeatures) throws Exception {
        String standDir = (standardiseF) ? "Standardise" : "NoStandardised";
        String cId = "RandomForest";
        if (cvTreeDepth)
            cId += "_cvTreeDepth";
        if (cvNumTrees)
            cId += "_cvNumTrees";
        if (cvNumFeatures)
            cId += "_cvNumFeatures";
        
        if (this.isOutputFile) {
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" +  this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" +  this.datasetName);

            System.out.println("done");
 
            File file = null; 
            OutFile outF = null;
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + cId  + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_" + expId + ".csv");
            }
            
            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            // I put standardisation here since in other experiments 
            // it was done inside the classifiers
            if (standardiseF) {
                Utilities.StandardiseDataset standInst = new Utilities.StandardiseDataset(data);
                data = standInst.standardiseInstances(data);
            }
            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            // Carry out a 10-fold cross validation  to set best min number of 
            // object per node if required
            // For the maximum Depth of trees
            if (cvTreeDepth) {
                rf.setMaxDepth(rfCvTreeDepth(rf,startDepth,endDepth,train) + startDepth);  
            }
            if (cvNumTrees)
                rf.setNumTrees(rfCvNumTrees(rf,startTrees,endTrees,train) + startTrees);
            if (cvNumFeatures)
                rf.setNumFeatures(rfCvNumFeatures(rf,train));

            rf.buildClassifier(train);
            
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();

            for (int inst = 0; inst < test.numInstances(); inst++) {
                guesses[inst] = (int) rf.classifyInstance(test.get(inst));
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            
            if (cvTreeDepth)
                outF.writeLine("Max Depth tree," + rf.getMaxDepth());
            if (cvNumTrees)
                outF.writeLine("Number of trees," + rf.getNumTrees());
            if (cvNumFeatures)
                outF.writeLine("Number of features," + rf.getNumFeatures());
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
        }

    }
    
    void samplingExp(SMO svm, int expId, boolean standardiseF,
            boolean cvNumFoldsF, int startFold, int endFold, boolean cvC,
            boolean cvEpsilonF) throws Exception {
        String standDir = (standardiseF) ? "Standardise" : "NoStandardised";
        String cId = "SVM";
        
        
        if (svm.getFilterType().toString().compareTo("0") == 0)
            cId += "_Normalized";
        if (cvNumFoldsF)
            cId += "_cvNumFolds";
        if (cvC)
            cId += "_cvC";
        if (cvEpsilonF)
            cId += "_cvEpsilon";
        
        System.out.println("Running Classifier : " + cId);

        
        if (this.isOutputFile) {
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" +  this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" +  this.datasetName);

            System.out.println("done");
 
            File file = null; 
            OutFile outF = null;
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + cId  + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_" + expId + ".csv");
            }
            
            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            // I put standardisation here since in other experiments 
            // it was done inside the classifiers
            if (standardiseF) {
                Utilities.StandardiseDataset standInst = new Utilities.StandardiseDataset(data);
                data = standInst.standardiseInstances(data);
            }
            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            // Carry out a 10-fold cross validation  to set best min number of 
            // object per node if required
            if (cvNumFoldsF)
                svm.setNumFolds(svmCvNumFolds(svm,startFold, endFold, train) + startFold);
            if (cvC)
                svm.setC(svmCvC(svm,train));
            if (cvEpsilonF)
                svm.setEpsilon(svmCvEpsilon(svm,train));

            
            svm.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();

            for (int inst = 0; inst < test.numInstances(); inst++) {
                System.out.println(inst);
                guesses[inst] = (int) svm.classifyInstance(test.get(inst));
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            
            if (cvNumFoldsF)
                outF.writeLine("Number of Folds," + svm.getNumFolds());
            if (cvC)
                outF.writeLine("C value," + svm.getC());
            if (cvEpsilonF)
                outF.writeLine("Epsilon value," + svm.getEpsilon() );
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
        }
    }
    
    
    void samplingExp(MultilayerPerceptron mlp, int expId, boolean standardiseF,
            boolean cvLearnRate, boolean cvMomenumRate, boolean cvEpoch) throws Exception {
        String standDir = (standardiseF) ? "Standardise" : "NoStandardised";
        String cId = "MLP";
        if (mlp.getNormalizeAttributes())
            cId += "_NormaliseAtts";
        if (cvLearnRate)
            cId += "_cvLearnRate";
        if (cvMomenumRate)
            cId += "_cvMomenumRate";
        if (cvEpoch) 
            cId += "_cvEpoch";
        
        System.out.println("Running Classifier : " + cId);
        
        if (this.isOutputFile) {
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" +  this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" +  this.datasetName);

            System.out.println("done");
 
            File file = null; 
            OutFile outF = null;
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + cId  + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_" + expId + ".csv");
            }
            
            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            // I put standardisation here since in other experiments 
            // it was done inside the classifiers
            if (standardiseF) {
                Utilities.StandardiseDataset standInst = new Utilities.StandardiseDataset(data);
                data = standInst.standardiseInstances(data);
            }
            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            // Carry out a 10-fold cross validation  to set best min number of 
            // object per node if required
            if (cvLearnRate)
                mlp.setLearningRate(mlpCvLearnRate(mlp,train));
            if (cvMomenumRate)
                mlp.setMomentum(mlpCvMomentumRate(mlp,train));
            if (cvEpoch) 
                mlp.setTrainingTime(mlpCvEpochs(mlp,train));
            
            mlp.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();

            for (int inst = 0; inst < test.numInstances(); inst++) {
                System.out.println(inst);
                guesses[inst] = (int) mlp.classifyInstance(test.get(inst));
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            
            if (cvLearnRate)
                outF.writeLine("Learing Rate," + mlp.getLearningRate());
            if (cvMomenumRate)
                outF.writeLine("Momentum rate," + mlp.getMomentum());
            if (cvEpoch) 
                outF.writeLine("Epoch," + mlp.getTrainingTime());
            
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
        }
    }
    
    void samplingExp(RotationForest c, int expId, boolean standardiseF) throws Exception {
        String standDir = (standardiseF) ? "Standardise" : "NoStandardised";
        String cId = "RotationForest";

        
        System.out.println("Running Classifier : " + cId);
        if (this.isOutputFile) {
            // Load the data
            System.out.print("Loading data...");
            Instances data = null;
            if (this.isCluster) 
                data = Utilities.loadData(this.datasetDir + "/" + this.dataType + "/" +  this.datasetName);
            else
                data = Utilities.loadData(this.datasetDir + "\\" + this.dataType + "\\" +  this.datasetName);

            System.out.println("done");
 
            File file = null; 
            OutFile outF = null;
            OutFile classLabels = null;
            OutFile predictedLabels = null; 
            if (this.isCluster) {
                file = new File(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_" + expId + ".csv");
                classLabels = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_cLabels_" + expId + ".csv");
                predictedLabels = new OutFile(this.outputDir + "/" + this.datasetName + "/" + cId + "/" + standDir + "/" + cId + "_prediLabels_" + expId + ".csv");
            } else {
                file = new File(this.outputDir + "\\" + this.datasetName + "\\" + cId  + "\\" + standDir);
                file.mkdirs();
                outF = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_" + expId + ".csv");
                classLabels = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_cLabels_" + expId + ".csv");
                predictedLabels = new OutFile(this.outputDir + "\\" + this.datasetName + "\\" + cId + "\\" + standDir + "\\" + cId + "_prediLabels_" + expId + ".csv");
            }
            
            // Build classifiers and take time 
            System.out.print("Building Classifier...");
            Instant start = Instant.now();
            // I put standardisation here since in other experiments 
            // it was done inside the classifiers
            if (standardiseF) {
                Utilities.StandardiseDataset standInst = new Utilities.StandardiseDataset(data);
                data = standInst.standardiseInstances(data);
            }
            Instances [] temp = InstanceTools.resampleInstances(data, expId, 0.5);
            Instances train = temp[0];
            Instances test  = temp[1];
            // Carry out a 10-fold cross validation  to set best min number of 
            // object per node if required

            c.buildClassifier(train);
            Instant end = Instant.now();
            long timeTakenBuild = Duration.between(start, end).toMillis();
            System.out.println("done");
            StringBuilder strResults = new StringBuilder();
            // Classify test set
//            int tempCount = 0;
            System.out.print("Start classification...");
            int correctGuess = 0;  
            int [] guesses = new int[test.numInstances()];
            start = Instant.now();

            for (int inst = 0; inst < test.numInstances(); inst++) {
                System.out.println(inst);
                classLabels.writeDouble(test.get(inst).classValue());
                classLabels.writeString(",");
                
                guesses[inst] = (int) c.classifyInstance(test.get(inst));
                
                predictedLabels.writeDouble(guesses[inst]);
                predictedLabels.writeString(",");
                
                if (guesses[inst] == (int) test.get(inst).classValue()) 
                    correctGuess++;
            }
            System.out.println("done");
            end = Instant.now();
            long timeTakenClassify = Duration.between(start, end).toMillis();
            
            // Create confusion matrix
            int [][] confMatrix = createConfusionMatrix(train, guesses);
            double Fscore = calculateFscore(confMatrix);
            outF.writeLine("Accuracy," + (double)correctGuess/(double)test.size());
            outF.writeLine("F-Score," + Fscore);
            outF.writeLine("Time Taken Build," + timeTakenBuild);
            outF.writeLine("Time Taken Classify," + timeTakenClassify);
            outF.writeLine("Confusion Matrix");
            Utilities.printConfusionMatrix(confMatrix,outF);
            outF.closeFile();
        }
    }
    
    
    public static int[][] createConfusionMatrix(Instances train, int [] guesses){
        int [][] matrix = new int[train.numClasses()][train.numClasses()];
        for (int i = 0; i < train.numInstances(); i++) {
            matrix[(int)train.get(i).classValue()][guesses[i]]++;
        }
        return matrix;
    }
    
    public static double calculateFscore(int [][] confMatrix) {
        // Start by calculating the overall precision for each class
        double [] precisions = new double[confMatrix.length];
        for (int col = 0; col < precisions.length; col++) {
            double truePositive = confMatrix[col][col];
            double falsePositive = 0;
            for (int row = 0; row < confMatrix.length; row++) {
                if (!(col == row))
                    falsePositive += confMatrix[row][col];
            }
            precisions[col] = truePositive / (truePositive + falsePositive);
        }
        double overallPrecision = 0;
        for (int i = 0; i < precisions.length; i++) {
            overallPrecision += precisions[i];
        }
        overallPrecision /= precisions.length;
        
        // Calculate the recall 
        double [] recalls = new double[confMatrix.length];
        for (int row = 0; row < recalls.length; row++) {
            double truePositive = confMatrix[row][row];
            double falseNegative = 0;
            for (int col = 0; col < confMatrix.length; col++) {
                if (!(row==col))
                    falseNegative += confMatrix[row][col];  
            }
            recalls[row] = truePositive / (truePositive + falseNegative);
        }
        double overallRecall = 0;
        for (int i = 0; i < recalls.length; i++) {
            overallRecall += recalls[i];
        }
        overallRecall /= recalls.length;
        
        // calculate and return overall Fscore
        return 2 * ((overallPrecision * overallRecall) / (overallPrecision + overallRecall));
    }
    
    public int dtCvMinNumberPerNode(J48 dt, int minNumberStart, int minNumberEnd, Instances train) throws Exception {
        double [] cvMinNumberAcc = new double[minNumberEnd - minNumberStart + 1];
        // Loop to iterate from Start form End
        for (int noPerNode = minNumberStart; noPerNode <= minNumberEnd; noPerNode++) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);
                // Set minNumbPerNode
                dt.setMinNumObj(noPerNode);
                // Build classifier
                dt.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = dt.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                cvMinNumberAcc[noPerNode-minNumberStart] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            cvMinNumberAcc[noPerNode-minNumberStart] /= 10;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < cvMinNumberAcc.length; i++) {
            if (cvMinNumberAcc[i] > cvMinNumberAcc[indexMax])
                indexMax = i;
        }
        return indexMax;
    }

    public int dtCvNumbOfFolds(J48 dt, int numbFoldsStart,int numbFoldsEnd, Instances train) throws Exception {
        double [] cvNumbFoldsAcc = new double[numbFoldsEnd - numbFoldsStart + 1];
        //Loop to iterate from Start form End
        for (int noFolds = numbFoldsStart; noFolds <= numbFoldsEnd; noFolds++) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);
                // Set number of folds
                dt.setNumFolds(noFolds);
                // Build classifier
                dt.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = dt.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                cvNumbFoldsAcc[noFolds-numbFoldsStart] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            cvNumbFoldsAcc[noFolds-numbFoldsStart] /= 10;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < cvNumbFoldsAcc.length; i++) {
            if (cvNumbFoldsAcc[i] > cvNumbFoldsAcc[indexMax])
                indexMax = i;
        }
        return indexMax;
    }

    private int rfCvTreeDepth(RandomForest rf,int startDepth, int endDepth, Instances train) throws Exception {
        double [] cvAccResults = new double[endDepth - startDepth + 1];
        //Loop to iterate from Start form End
        for (int depthLvl = startDepth; depthLvl <= endDepth; depthLvl++) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);  
                // Set number of folds
                rf.setMaxDepth(depthLvl);
                // Build classifier
                rf.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = rf.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                cvAccResults[depthLvl-startDepth] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            cvAccResults[depthLvl-startDepth] /= 10;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < cvAccResults.length; i++) {
            if (cvAccResults[i] > cvAccResults[indexMax])
                indexMax = i;
        }
        return indexMax;
    }
    
    public int rfCvNumTrees(RandomForest rf,int start, int end, Instances train) throws Exception {
        double [] cvAccResults = new double[end - start + 1];
        //Loop to iterate from Start form End
        for (int numTrees = start; numTrees <= end; numTrees++) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold); 
                // Set number of folds
                rf.setNumTrees(numTrees);
                // Build classifier
                rf.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = rf.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                cvAccResults[numTrees-start] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            cvAccResults[numTrees-start] /= 10;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < cvAccResults.length; i++) {
            if (cvAccResults[i] > cvAccResults[indexMax])
                indexMax = i;
        }
        return indexMax;
    }
    
    public int rfCvNumFeatures(RandomForest rf,Instances train) throws Exception {
        double [] cvAccResults = new double[train.numAttributes()];
        //Loop to iterate from Start form End
        for (int numFeatures = 0; numFeatures <= train.numAttributes()-1; numFeatures++) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);
                // Set number of folds
                rf.setNumFeatures(numFeatures);
                // Build classifier
                rf.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = rf.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                cvAccResults[numFeatures] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            cvAccResults[numFeatures] /= 10;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < cvAccResults.length; i++) {
            if (cvAccResults[i] > cvAccResults[indexMax])
                indexMax = i;
        }
        return indexMax;
    }

    public int svmCvNumFolds(SMO svm, int start, int end, Instances train) throws Exception {
        double [] cvNumbFoldsAcc = new double[end - start + 1];
        //Loop to iterate from Start form End
        for (int noFolds = start; noFolds <= end; noFolds++) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);
                // Set number of folds
                svm.setNumFolds(noFolds);
                // Build classifier
                svm.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = svm.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                cvNumbFoldsAcc[noFolds-start] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            cvNumbFoldsAcc[noFolds-start] /= 10;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < cvNumbFoldsAcc.length; i++) {
            if (cvNumbFoldsAcc[i] > cvNumbFoldsAcc[indexMax])
                indexMax = i;
        }
        return indexMax;
    }
    
    public double svmCvC(SMO svm, Instances train) throws Exception {
        // Given that we seach from 0.1 to 10 there are 100 values in it 
        // if we increment it every 0.1
        double [] accResults = new double[100];
        int index = 0;
        for (double tempC = 0.1; tempC <= 10; tempC += 0.1) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);
                svm.setC(tempC);
                // Build classifier
                svm.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = svm.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                accResults[index] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            accResults[index] /= 10;
            index++;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < accResults.length; i++) {
            if (accResults[i] > accResults[indexMax])
                indexMax = i;
        }
        return (0.1*indexMax)+0.1;
    }
    
    public double svmCvEpsilon(SMO svm, Instances train) throws Exception {
        // Given that we start form 1*10^-12 and get to 10 incrementing it by 10
        // there are 14 values
        double [] accResults = new double[14];
        int index = 0;
        for (double tempEpsilon = 1.0e-12; tempEpsilon <= 10; tempEpsilon *= 10) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);
                svm.setEpsilon(tempEpsilon);
                // Build classifier
                svm.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = svm.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                accResults[index] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            accResults[index] /= 10;
            index++;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < accResults.length; i++) {
            if (accResults[i] > accResults[indexMax])
                indexMax = i;
        }
        int temp = indexMax-12;
        String temp2 = "1.0e" + (indexMax-12);
        double temp3 = Double.parseDouble(temp2);
        return temp3;
    }

    public double mlpCvLearnRate(MultilayerPerceptron mlp, Instances train) throws Exception {
        // Given that we start form 0.1 and get to 1 each time incrementing 
        // 0.1 we will have 10 accuracies
        double [] accResults = new double[10];
        int index = 0;
        for (double tempRate = 0.1; tempRate <= 1; tempRate += 0.1) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);

                mlp.setLearningRate(tempRate);
                // Build classifier
                mlp.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = mlp.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                accResults[index] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            accResults[index] /= 10;
            index++;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < accResults.length; i++) {
            if (accResults[i] > accResults[indexMax])
                indexMax = i;
        }
        return (indexMax/10)+0.1;
    }
    
    public double mlpCvMomentumRate(MultilayerPerceptron mlp, Instances train) throws Exception {
        // Given that we start form 0.1 and get to 1 each time incrementing 
        // 0.1 we will have 10 accuracies
        double [] accResults = new double[10];
        int index = 0;
        for (double tempRate = 0.1; tempRate <= 1; tempRate += 0.1) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);

                mlp.setMomentum(tempRate);
                // Build classifier
                mlp.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = mlp.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                accResults[index] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            accResults[index] /= 10;
            index++;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < accResults.length; i++) {
            if (accResults[i] > accResults[indexMax])
                indexMax = i;
        }
        return (indexMax/10)+0.1;
    }
    
    public int mlpCvEpochs(MultilayerPerceptron mlp, Instances train) throws Exception {
        // Given that we start form 0.1 and get to 1 each time incrementing 
        // 0.1 we will have 10 accuracies
        double [] accResults = new double[10];
        int index = 0;
        for (int tempEpoch = 100; tempEpoch <= 1000; tempEpoch += 100) {
            // Run 10FoldCrossValidation
            for (int fold = 0; fold < 10; fold++) {
                Instances tempTrain = train.trainCV(10, fold);
                Instances tempTest  = train.testCV(10, fold);

                mlp.setTrainingTime(tempEpoch);
                // Build classifier
                mlp.buildClassifier(tempTrain);
                // Classify data
                int correctGuess = 0;
                for (Instance inst : tempTest) {
                    double pred = mlp.classifyInstance(inst);
                    if (pred == inst.classValue())
                        correctGuess++;
                }
                // Get accuracy 
                accResults[index] += (double) correctGuess/tempTest.size();
            }
            // Average acc over the 10 folds
            accResults[index] /= 10;
            index++;
        }
        // Find higher accuracy 
        int indexMax = 0;
        for (int i = 1; i < accResults.length; i++) {
            if (accResults[i] > accResults[indexMax])
                indexMax = i;
        }
        return (100 * indexMax)+100;
    }

}
