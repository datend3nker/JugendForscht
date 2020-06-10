using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using System.Diagnostics;
using Microsoft.ML.Trainers.LightGbm;
using System.Text.Json;

namespace Csharp_machieneLearning
{
    class Program
    {

        public static IDictionary<string, dynamic> Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Stopwatch stw = new Stopwatch();
            stw.Start();
            IDataView predictions = model.Transform(splitTestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(transformOutput.Label), scoreColumnName: nameof(QuestionPrediction.Score), probabilityColumnName: nameof(QuestionPrediction.Probability), predictedLabelColumnName: "PredictedLabel");
            stw.Stop();
            #region consol output
            Console.WriteLine($"Evaluating the given model                      {stw.ElapsedMilliseconds / 1000f}s");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"  Accuracy:             {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:                  {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  Auprc:                {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"  F1Score:              {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:              {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:     {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision:    {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:       {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision:    {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:       {metrics.NegativeRecall:0.##}");
            Console.WriteLine();
            #endregion
            IDictionary<string, dynamic> output = new Dictionary<string, dynamic>(){
                {"Accuracy", metrics.Accuracy},
                {"Auc", metrics.AreaUnderRocCurve},
                {"Auprc", metrics.AreaUnderPrecisionRecallCurve},
                {"F1Score", metrics.F1Score},
                {"LogLoss", metrics.LogLoss},
                {"LogLossReduction", metrics.LogLossReduction},
                {"PositivePrecision", metrics.PositivePrecision},
                {"PositiveRecall", metrics.PositiveRecall},
                {"NegativePrecision", metrics.NegativePrecision},
                {"NegativeRecall", metrics.NegativeRecall},
            };
            return output;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView traindata, IEstimator<ITransformer> pipeline, string modelname, IDictionary<string, IEstimator<ITransformer>> estimator, Stopwatch stw)
        {
            ITransformer model;
            try
            {
                stw.Start();
                model = estimator[modelname].Fit(traindata);
                stw.Stop();
                Console.WriteLine($"Finished training model         {stw.ElapsedMilliseconds / 1000f}s");
                return model;
            }
            catch(KeyNotFoundException)
            {
                Console.Write($"The program has stoped because KeyNotFoundException");
                Console.WriteLine("The modelname might be spelled incorrectly");
                Console.WriteLine();
                Console.WriteLine("Press any key to end the program");
                Console.ReadKey();
                System.Environment.Exit(1);
                return null;
            }  
        }

        public static void TrainMultiModel(MLContext mlContext, IDataView file, IDictionary<string, IEstimator<ITransformer>> estimator, Stopwatch stw)
        {
            var data = mlContext.Data.TrainTestSplit(file, testFraction:0.2, seed: 42);
            foreach (KeyValuePair<string, IEstimator<ITransformer>> item in estimator)
            {
                stw.Restart();
                var model = item.Value.Fit(data.TrainSet);
                stw.Stop();
                Console.WriteLine($"Start training of {item.Key}    {stw.ElapsedMilliseconds / 1000f}s");
                Evaluate(mlContext, model, data.TestSet);
            }
        }

        public static void CrossValidation(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView data, IEstimator<ITransformer> model, int f)
                {
                    IDataView transData = pipeline.Fit(data).Transform(data);
                    var cvResults = mlContext.BinaryClassification.CrossValidate(
                                    data: transData,
                                    estimator: model,
                                    numberOfFolds: f);
                    Console.WriteLine("Result of crossValidation");
                    foreach (var r in cvResults)
                        Console.WriteLine($"  Fold: {r.Fold}, AUC: {r.Metrics.AreaUnderRocCurve}, Accuracy: {r.Metrics.Accuracy}");
                    Console.WriteLine($"   Average AUC: {cvResults.Average(r => r.Metrics.AreaUnderRocCurve)}");
                    Console.WriteLine();
                }

        public static void PreviewData(IDataView data, int numberofitems)
                {
                    var preview = data.Preview(numberofitems);

                    foreach (var row in preview.RowView)
                    {
                        foreach (var column in row.Values)
                        {
                            Console.WriteLine(column);
                        }
                        Console.WriteLine();
                        Console.WriteLine();
                    }
                }

        public static void AutoML(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView file, Progress<RunDetail<BinaryClassificationMetrics>> progress, BinaryExperimentSettings settings)
                {
                    var transdata = pipeline.Fit(file).Transform(file);
                    ExperimentResult<BinaryClassificationMetrics> experimentResult = mlContext.Auto()
                       .CreateBinaryClassificationExperiment(settings)
                       .Execute(trainData: transdata, labelColumnName: nameof(transformOutput.Label), progressHandler: progress);

                    Console.WriteLine();
                    Console.WriteLine($"Trainername- {experimentResult.BestRun.TrainerName}");
                    Console.WriteLine($"Accuracy- {experimentResult.BestRun.ValidationMetrics.Accuracy}");
                    Console.WriteLine($"AreaUnderRocCurve- {experimentResult.BestRun.ValidationMetrics.AreaUnderRocCurve}");
                    Console.WriteLine();
                    var model = experimentResult.BestRun.Model as TransformerChain<ITransformer>;
                    //mlContext.Model.Save(model, transdata.Schema, @"C:\Users\ludwi\source\repos\JugendForscht");
                }

        public static void Sweeper(MLContext mlContext, IDataView file, IEstimator<ITransformer> pipeline, string modelname, IDictionary<string, IEstimator<ITransformer>> estimator, Stopwatch stw)
        {
            //Diese Optionen sind nur für den LightGBM
            IDictionary<string, dynamic> Result = new Dictionary<string, dynamic>();
            int trainingStage = 0;
            double[] LearningRate = new double[] { 1, 0.5, 0.25, 0.1, 0.001, 0.0001, 0.00001, 0.000001 };
            int[] NumberOfIterations = new int[] { 10, 20, 50, 75, 100, 150, 200, 300 };
            double[] Sigmoid = new double[] { 1.1, 1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001, 0.0001 };
            bool[] UnbalancedSets = new bool[] { true, false };

            var Model = estimator[modelname];
            var data = mlContext.Data.TrainTestSplit(file, testFraction: 0.2, seed: 42);

            foreach (double learningrate in LearningRate)
            {
                foreach (int numberofiterations in NumberOfIterations)
                {
                    foreach (double sigmoid in Sigmoid)
                    {
                        foreach (bool unbalancedsets in UnbalancedSets)
                        {
                            trainingStage++;
                            var options = new LightGbmBinaryTrainer.Options
                            {
                                LearningRate = learningrate,
                                NumberOfIterations = numberofiterations,
                                Sigmoid = sigmoid,
                                UnbalancedSets = unbalancedsets,
                                Verbose = true,
                                Silent = false,
                            };
                            stw.Restart();
                            var model = Model.Fit(data.TrainSet);
                            stw.Stop();
                            Console.WriteLine($"Finished training {modelname} with parameters:  {stw.ElapsedMilliseconds / 1000}s");
                            Console.WriteLine($"  LearningRate:             {learningrate}");
                            Console.WriteLine($"  NumberOfIterations:       {numberofiterations}");
                            Console.WriteLine($"  Sigmoid:                  {sigmoid}");
                            Console.WriteLine($"  UnbalancedSets:           {unbalancedsets}");
                            Console.WriteLine();
                            #region Dictionary for logging data
                            IDictionary<string, dynamic> param = new Dictionary<string, dynamic>(){
                                { "LearningRate", learningrate},
                                { "NumberOfIterations", numberofiterations },
                                {"Sigmoid", sigmoid },
                                {"UnbalancedSets",  unbalancedsets},

                            };
                            
                            IDictionary<string, dynamic> leistung = Evaluate(mlContext: mlContext, model: model, data.TestSet);

                            IDictionary<string, Dictionary<string, dynamic>> leistung_param = new Dictionary<string, Dictionary<string, dynamic>>()
                            {
                                {"Leistung", (Dictionary<string, dynamic>)leistung },
                                {"Parameter", (Dictionary<string, dynamic>)param }
                            };

                            IDictionary<string, Dictionary<string, Dictionary<string, dynamic>>> output = new Dictionary<string, Dictionary<string, Dictionary<string, dynamic>>>()
                            {
                                {modelname, (Dictionary<string, Dictionary<string, dynamic>>)leistung_param }
                            };
                            #endregion
                            string json = JsonSerializer.Serialize(output);
                            Console.WriteLine(json);
                        }
                    }
                }
            }
        }
            
        
    
        

        static void Main(string[] args)
        {

            #region creating all objects needed
            MLContext mlContext = new MLContext();
            Stopwatch stw = new Stopwatch();
            BinaryExperimentSettings settings = new BinaryExperimentSettings();
            Progress<RunDetail<BinaryClassificationMetrics>> progress = new Progress<RunDetail<BinaryClassificationMetrics>>(p =>
                                    {
                                        if (p.ValidationMetrics != null)
                                        {
                                            Console.WriteLine($"Current result - {p.TrainerName}, {p.ValidationMetrics.Accuracy}, {p.ValidationMetrics.AreaUnderRocCurve}");
                                        }
                                    });

            #region Transformer pipeline
            Action<QuestionPairs, transformOutput> mapping = (input, output) => { output.Label = input.is_duplicate.Equals("1") ? true : false; };
            IEstimator<ITransformer> pipeline = mlContext.Transforms.CustomMapping(mapping, contractName: null)
                            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "question1", outputColumnName: "question1Featurized"))
                            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "question2", outputColumnName: "question2Featurized"))
                            .Append(mlContext.Transforms.Concatenate("Features", "question1Featurized", "question2Featurized"))
                            .Append(mlContext.Transforms.DropColumns("question1", "question2", "is_duplicate"))
                            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features"))
                            .AppendCacheCheckpoint(mlContext);
            #endregion
            
            #region Dictionary of models
                        IDictionary<string, IEstimator<ITransformer>> estimator = new Dictionary<string, IEstimator<ITransformer>>();

                        estimator.Add("AveragedPerceptronTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("SdcaLogisticRegressionBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("SdcaNonCalibratedBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("SymbolicSgdLogisticRegressionBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("LbfgsLogisticRegressionBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("LightGbmBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("FastTreeBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("FastForestBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        //estimator.Add("GamBinaryTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.Gam(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("FieldAwareFactorizationMachineTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
                        estimator.Add("LinearSvmTrainer", pipeline.Append(mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: nameof(transformOutput.Label), featureColumnName: "Features")));
            #endregion

            #region AutoML settings
            settings.MaxExperimentTimeInSeconds = 60;
            settings.Trainers.Clear();
            settings.Trainers.Add(BinaryClassificationTrainer.LightGbm);
            #endregion

            #endregion
            stw.Start();
            IDataView file = mlContext.Data.LoadFromTextFile<QuestionPairs>(@".\questions.csv", separatorChar: ',', hasHeader: true, allowQuoting: true);
            var data = mlContext.Data.TrainTestSplit(file, testFraction: 0.2, seed: 42);
            stw.Stop();
            Console.WriteLine($"Finished loading dataset                        {stw.ElapsedMilliseconds / 1000f}s");

            #region Examples
            /*
            var model = BuildAndTrainModel(mlContext: mlContext, traindata: data.TrainSet, pipeline: pipeline, modelname: "FastTreeBinaryTrainer", estimator: estimator, stw: stw);
            Evaluate(mlContext: mlContext, model: model, splitTestSet: data.TestSet);
            //=====================================================================================================================================================================================
            AutoML(mlContext: mlContext, pipeline: pipeline, file: file, progress: progress, settings: settings);
            //=====================================================================================================================================================================================
            TrainMultiModel(mlContext: mlContext, file: file, estimator: estimator, stw: stw);
            //=====================================================================================================================================================================================
            */

            Sweeper(mlContext: mlContext, file: file, pipeline: pipeline, modelname: "LightGbmBinaryTrainer", estimator: estimator, stw: stw);
            #endregion


            Console.WriteLine("Press any key to end the program");
            Console.ReadKey();
        }
    }
}