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

    [LoadColumn(3)]
    public float Neighborhood { get; set; }
}

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

public class TransformedHousingData
{
    public float SquareFeet { get; set; }
    public float Bedrooms { get; set; }
    public float Price { get; set; }
    public float[] Features { get; set; }
    public float[] Neighborhood { get; set; }
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
        var dataPipeline = mlContext.Transforms.Conversion.ConvertType("SquareFeet", outputKind: DataKind.Single)
            .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
            .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));
        var transformedData = dataPipeline.Fit(dataView).Transform(dataView);
        var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TransformedHousingData>(transformedData, reuseRowObject: false).ToList();
        foreach (var item in transformedDataEnumerable)
        {
            Console.WriteLine($"SquareFeet: {item.SquareFeet}, Bedrooms: {item.Bedrooms}, Price: {item.Price}, Features: [{string.Join(", ", item.Features)}], Neighborhood: [{string.Join(", ", item.Neighborhood)}]");
        }

    }
}