using Microsoft.ML;
using Microsoft.ML.Data;

public class HousingData
{
    [LoadColumn(0)]
    public float SquareFeet { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }

}

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

public class Program
{
    // Entry point for the ML.NET application 
    static void Main(string[] args)
    {
        string filePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "housing-data.csv");

        if (File.Exists(filePath))
        {
            var lines = File.ReadAllLines(filePath);
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }
        else
        {
            Console.WriteLine("File not found.");
        }

        MLContext mlContext = new MLContext();
        IDataView dataView = mlContext.Data.LoadFromTextFile<HousingData>(filePath, separatorChar: ',', hasHeader: true);
        string[] featuresColumn = { "SquareFeet", "Bedrooms" };
        string labelColumn = "Price";
        // Fast Tree is a regression trainer that uses decision trees for prediction.
        var pipeline = mlContext.Transforms.Concatenate("Features", featuresColumn)
            .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));
        // Fit the model to the data
        var model = pipeline.Fit(dataView);
        var prediction = model.Transform(dataView);
        var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);
        Console.WriteLine($"R-squared: {metrics.RSquared}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
    }
}