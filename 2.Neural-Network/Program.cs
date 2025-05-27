public class NeuralNetwork
{
    private double[,] weights;

    enum OPERATION
    {
        Multiply, Add, Subtract
    }

    public NeuralNetwork()
    {
        Random randomNumber = new Random(1);

        int noOfInputNodes = 3;
        int noOfOutputNodes = 1;

        weights = new double[noOfInputNodes, noOfOutputNodes];

        for (int i = 0; i < noOfInputNodes; i++)
        {
            for (int j = 0; j < noOfOutputNodes; j++)
            {
                weights[i, j] = 2 *  randomNumber.NextDouble() - 1;
            }
        }
    }

    private double[,] Activate(double[,] matrix, bool isDerivative)
    {
        int noOfRows = matrix.GetLength(0);
        int noOfColumns = matrix.GetLength(1);

        double[,] result = new double[noOfRows, noOfColumns];

        for (int row = 0; row < noOfRows; row++)
        {
            for (int column = 0; column < noOfColumns; column++)
            {
                double sigmoidOutput = result[row, column] = 1 / (1 + Math.Exp(-matrix[row, column]));
                double derivativeSigmoidOutput = result[row, column] = matrix[row, column] * (1 - matrix[row, column]);
                result[row, column] = isDerivative ? derivativeSigmoidOutput : sigmoidOutput;   
            }
        }

        return result;
    }

    public void Train(double[,] trainingInputs, double[,] trainingOutputs, int noOfIterations)
    {
        for (int iteration = 0; iteration < noOfIterations; iteration++)
        {
            double[,] output = Think(trainingInputs);
            double[,] error = PerformOperation(trainingOutputs, output, OPERATION.Subtract);
            double[,] adjustment = DotProduct(Transpose(trainingInputs), PerformOperation(error, Activate(output, true), OPERATION.Multiply));
            weights = PerformOperation(weights, adjustment, OPERATION.Add);
        }
    }

    private double[,] DotProduct(double[,] matrix1, double[,] matrix2)
    {
        int noOfRowsInMatrix1 = matrix1.GetLength(0);
        int noOfColumnsInMatrix1 = matrix1.GetLength(1);
        int noOfRowsInMatrix2 = matrix2.GetLength(0);
        int noOfColumnsInMatrix2 = matrix2.GetLength(1);

        double[,] result = new double[noOfColumnsInMatrix1, noOfColumnsInMatrix2];

        for (int rowInMatrix1 = 0; rowInMatrix1 < noOfRowsInMatrix1; rowInMatrix1++)
        {
            for (int columnInMatrix2 = 0; columnInMatrix2 < noOfColumnsInMatrix2; columnInMatrix2++)
            {
                double sum = 0;

                for (int columnInMatrix1 = 0; columnInMatrix1 < noOfColumnsInMatrix1; columnInMatrix1++)
                {
                    sum += matrix1[rowInMatrix1, columnInMatrix1] * matrix2[columnInMatrix1, columnInMatrix2];
                }
                result[rowInMatrix1, columnInMatrix2] = sum;
            }
        }
        return result;
    }

    public double[,] Think(double[,] inputs) {
        return Activate(DotProduct(inputs, weights), false);
    }

    private double[,] PerformOperation(double[,] matrix1, double[,] matrix2, OPERATION operation)
    {
        int noOfRows = matrix1.GetLength(0);
        int noOfColumns = matrix1.GetLength(1);

        double[,] result = new double[noOfRows, noOfColumns];

        for (int row = 0; row < noOfRows; row++)
        {
            for (int column = 0; column < noOfColumns; column++)
            {
                switch (operation)
                {
                    case OPERATION.Multiply:
                        result[row, column] = matrix1[row, column] * matrix2[row, column];
                        break;
                    case OPERATION.Add:
                        result[row, column] = matrix1[row, column] + matrix2[row, column];
                        break;
                    case OPERATION.Subtract:
                        result[row, column] = matrix1[row, column] - matrix2[row, column];
                        break;
                }
            }
        }
        return result;
    }

    private double[,] Transpose (double[,] matrix)
    {
        return matrix.Cast<double>().ToArray().Transpose(matrix.GetLength(0), matrix.GetLength(1));
    }

    static void Main(string[] args)
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        double[,] trainingInputs = new double[,]
        {
            { 0, 0, 0 },
            { 1, 1, 1 },
            { 1, 0, 0 }
        };
        double[,] trainingOutputs = new double[,]
        {
            { 0 },
            { 1 },
            { 1 }
        };

        neuralNetwork.Train(trainingInputs, trainingOutputs, 1000);

        double[,] testInput = new double[,]
        {
            // testInput SET - 1 
            { 0, 1, 0 }, // Output - 1
            { 0, 0, 0 }, // Output - 0
            { 0, 0, 1 }, // Output - 1

            // testInput SET - 2 
            // { 0, 0, 1 }, // Output - 1
            // { 1, 0, 1 }, // Output - 1
            // { 1, 1, 1 }  // Output - 1
        };

        double[,] output = neuralNetwork.Think(testInput);
        PrintMatrix(output);
    }

    static void PrintMatrix(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int columns = matrix.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                Console.Write(Math.Round(matrix[row, column]) + " ");
            }
            Console.WriteLine();
        }
    }
}

public static class Extensions 
{
    public static double[,] Transpose(this double[] array, int noOfRows, int noOfColumns)
    {
        double[,] transposedMatrix = new double[noOfColumns, noOfRows];

        for (int row = 0; row < noOfRows; row++)
        {
            for (int column = 0; column < noOfColumns; column++)
            {
                transposedMatrix[column, row] = array[row * column + column];
            }
        }
        return transposedMatrix;
    } 
}