using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Text;
using Microsoft.ML.Data;

namespace Csharp_machieneLearning
{
    public class QuestionPairs
    {
        [LoadColumn(3)]
        public string question1 { get; set; }

        [LoadColumn(4)]
        public string question2 { get; set; }

        [LoadColumn(5)]
        public string is_duplicate { get; set; }

    }
    public class QuestionPrediction : QuestionPairs
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
    public class transformOutput
    {
        public bool Label { get; set; }
    }
}
