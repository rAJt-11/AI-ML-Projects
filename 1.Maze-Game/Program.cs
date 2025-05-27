using TorchSharp;

// Maze Setup :
int[,] maze1 = {
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
    { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
    { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
    { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
    { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
    { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
    { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
    { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
};


// Implementing a Q-learning algorithm in C# using TorchSharp to train an agent to navigate a maze from a start point to a goal.

// The maze is a 2D grid.

// 0 represents -- WALL
// 1 represents -- PATH
// 2 represents -- GOAL

// The agent starts at (11, 5) and tries to reach the goal at (0, 5).

// Q-learning is used to train the agent via rewards for each state-action pair.

// Q-learning FORMULA :

// Q(s, a) = Q(s, a) + learningRate * (reward + discountFactor * max(Q(s', a')) - Q(s, a))

//    s: current state(cell)
//    a: action taken
//    s': new state (after action)
//    reward: immediate reward for moving to s'
//    Q(s, a): current Q-value
//    Q(s', a'): best possible Q-value in new state


const string UP = "up";
const string DOWN = "down";
const string LEFT = "left";
const string RIGHT = "right";

string[] actions = { UP, DOWN, LEFT, RIGHT };

int[,] rewards;

const int WALL_REWARD_VALUE = -500; // Represents a wall in the maze, which the agent cannot pass through.
const int FLOOR_REWARD_VALUE = -10; // Represents a floor in the maze, which the agent can walk on.
const int GOAL_REWARD_VALUE = 500; // Represents the goal in the maze, which the agent aims to reach.

void setupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);

    rewards = new int[mazeRows, mazeColumns];
    for (int i = 0; i < mazeRows; i++)
    {
        for (int j = 0; j < mazeColumns; j++)
        {
            if (maze[i, j] == 0)
                rewards[i, j] = wallValue;
            else if (maze[i, j] == 1)
                rewards[i, j] = floorValue;
            else if (maze[i, j] == 2)
                rewards[i, j] = goalValue;
        }
    }
}

torch.Tensor qValues;

void setupQValues(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);
    // For every cell [i, j], you track 4 Q-values (one per action).
    qValues = torch.zeros(mazeRows, mazeColumns, 4);
}

bool hasHitWallOrEndOfMaze(int currentRow, int currentColumn, int floorValue)
{
    return rewards[currentRow, currentColumn] != floorValue;
}

long determineNextAction(int currentRow, int currentColumn, float epsilon)
{
    Random random = new Random();
    double randomBetween0and1 = random.NextDouble();
    long nextAction = randomBetween0and1 < epsilon ? torch.argmax(qValues[currentRow, currentColumn]).item<long>() : random.Next(4);
    return nextAction;
}


(int, int) moveOneSpace(int[,] maze, int currentRow, int currentColumn, long currentAction)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);

    int nextRow = currentRow;
    int nextColumn = currentColumn;

    if (actions[currentAction] == UP && currentRow > 0)
    {
        nextRow--;
    }
    else if (actions[currentAction] == DOWN && currentRow < mazeRows - 1)
    {
        nextRow++;
    }
    else if (actions[currentAction] == LEFT && currentColumn > 0)
    {
        nextColumn--;
    }
    else if (actions[currentAction] == RIGHT && currentColumn < mazeColumns  - 1)
    {
        nextColumn++;
    }
    return (nextRow, nextColumn);
}

void trainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, float episodes)
{
    for (int episode = 0; episode < episodes; episode++)
    {
        Console.WriteLine($"-----Starting episode " + episode + "-----");
        int currentRow = 11;
        int currentColumn  = 5;

        while(!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            long currentAction = determineNextAction(currentRow, currentColumn, epsilon);
            int previousRow = currentRow;
            int previousColumn = currentColumn;
            (int, int) nextMove = moveOneSpace(maze, currentRow, currentColumn, currentAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            float reward = rewards[currentRow, currentColumn];
            float previousQValue = qValues[previousRow, previousColumn, currentAction].item<float>();
            float temporalDifference = reward + (discountFactor * torch.max(qValues[currentRow, currentColumn])).item<float>() - previousQValue;
            float nextQValue = previousQValue + (learningRate * temporalDifference);
            qValues[previousRow, previousColumn, currentAction] = nextQValue;
        }

        Console.WriteLine($"-----Finished episode " + episode + "-----");
    }

    Console.WriteLine($"-----Completed Training-----");
} 

List<int[]> navigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
{
    List<int[]> path = new List<int[]>();
    if (  hasHitWallOrEndOfMaze(startRow, startColumn, floorValue))
    {
        return [];
    }
    else
    {
        int currentRow = startRow;
        int currentColumn = startColumn;
        path = new List<int[]> { new int[] { currentRow, currentColumn } };
        while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            int nextAction = (int) determineNextAction(currentRow, currentColumn, 1.0f);
            (int, int) nextMove = moveOneSpace(maze, currentRow, currentColumn, nextAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            if (rewards[currentRow, currentColumn] != wallValue)
            {
                path.Add(new int[] { currentRow, currentColumn });
            }
            else
            {
                continue;
            }
        }
    }

    int moveCount = 1;
    for(int i = 0; i < path.Count; i++)
    {
        Console.Write("Move " + moveCount + ": (");
        foreach (int coordinate in path[i])
        {
            Console.Write(" " + coordinate);
        }
        Console.Write(" )");
        Console.WriteLine();
        moveCount++;
    }
    return path;
}

const float EPSILON = 0.95f;
const float DISCOUNT_FACTOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const float EPISODES = 1500;
const int START_ROW = 11;
const int START_COLUMN = 5;

setupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);
setupQValues(maze1);
trainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES); 
navigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);