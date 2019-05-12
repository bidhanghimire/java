package NeuralNetClasses;

import data.NeuralDataSet;
import learn.DeltaRule;
import learn.LearningAlgorithm;
import mathClasses.HyperTan;
import mathClasses.Sigmoid
import mathClasses.Linear;
import mathClasses.RandomNumberGenerator;
import mathClasses.IActivationFunction;

/**
* NeuralNetTest classes
* This class trains the neurons and test data are tested using delta rule method
* MAin class
* Sigmoid function is used as activation function in this class
**/
public class NeuralNetTest {

    public static void main(String[] args){

        RandomNumberGenerator.seed=0;

        int numberOfInputs=1;
        int numberOfOutputs=1;

        Linear outputAcFnc = new Linear(0.9);
        IActivationFunction sgAcFnc = new Sigmoid(1.0); //Activation function used is sigmoid
        System.out.println("Creating Neural Network...");
        NeuralNet nn = new NeuralNet(numberOfInputs,numberOfOutputs, sgAcFnc);
                //outputAcFnc);

        System.out.println("Neural Network created!");
        System.out.println(nn.toString());
        /**
         * The first column is the input, x values and the second column is the result of the function.
         * Rather than calculating and putting the values the function itself is used, so that it will work for changed inputs.
         */

        Double[][] dataSet = {

                {12.0 , fncTest(30.5) }
              , {12.0 , fncTest(30.5) }
              , {15.0 , fncTest(23.3) }
              , {9.0 , fncTest(37.7) }
              , {5.0 , fncTest(47.3) }
              , {3.0 , fncTest(52.1) }
              , {4.0 , fncTest(49.7) }
              , {11.0 , fncTest(32.9) }
              , {10.0 , fncTest(35.3) }
                    , {11.0 , fncTest(32.9) }
              , {6.0 , fncTest(44.9) }
              , {9.0 , fncTest(37.7) }
              , {3.0 , fncTest(52.1) }
              , {6.0 , fncTest(44.9) }
              , {12.0 , fncTest(30.5) }
              , {3.0 , fncTest(52.1) }
              , {11.0 , fncTest(32.9) }
                    , {11.0 , fncTest(32.9) }
              , {13.0 , fncTest(28.1) }
              , {6.0 , fncTest(44.9) }
              , {9.0 , fncTest(37.7) }
              , {7.0 , fncTest(42.5) }
              , {2.0 , fncTest(54.5) }
              , {10.0 , fncTest(35.3) }
              , {14.0 , fncTest(25.7) }
                    , {3.0 , fncTest(52.1) }
              , {0.0 , fncTest(59.3) }
              , {0.0 , fncTest(59.3) }
              , {6.0 , fncTest(44.9) }

            };


        int[] inputColumns = {0};
        int[] outputColumns = {1};

        NeuralDataSet neuralDataSet = new NeuralDataSet(dataSet,inputColumns,outputColumns);

        System.out.println("Dataset created");
        neuralDataSet.printInput();
        neuralDataSet.printTargetOutput(); // This is the output created by applying the function to the values in the first column

        System.out.println("Getting the first output of the neural network");

        DeltaRule deltaRule = new DeltaRule(nn,neuralDataSet
                ,LearningAlgorithm.LearningMode.ONLINE);   // online mode applies to each record, rathen that the whole set
       /**
        * Set the values of learning rate, epochs or number of generations to train.
        */
        deltaRule.printTraining = true;
        deltaRule.setLearningRate(0.3);
        deltaRule.setMaxEpochs(200);
        deltaRule.setGeneralErrorMeasurement(DeltaRule.ErrorMeasurement.SimpleError);
        deltaRule.setOverallErrorMeasurement(DeltaRule.ErrorMeasurement.MSE);
        deltaRule.setMinOverallError(0.001);

        try
        {
            deltaRule.forward();
            neuralDataSet.printNeuralOutput();

            Double weight = nn.getOutputLayer().getWeight(0, 0);
            Double bias = nn.getOutputLayer().getWeight(1, 0);

            System.out.println("Initial weight:"+String.valueOf(weight));
            System.out.println("Initial bias:"+String.valueOf(bias));

            System.out.println("training begins");

            /**
             * The train method goes in a loop till the MaxEpoch is reached or Error is reduced to the target.
             * The train() method calls forward which calculates and updates the errors. Based on the the weights
             * are adjusted by deltWeight.
             */
            deltaRule.train();

            System.out.println("training ends");
            if(deltaRule.getMinOverallError() >= deltaRule.getOverallGeneralError())
            {
                System.out.println("Training succesful!");
            }
            else
            {
                System.out.println("Training was unsuccesful");
            }
            System.out.println("Overall Error:"
                        +String.valueOf(deltaRule.getOverallGeneralError()));
            System.out.println("Min Overall Error:"
                        +String.valueOf(deltaRule.getMinOverallError()));
            System.out.println("Epochs of training:"
                        +String.valueOf(deltaRule.getEpoch()));

            System.out.println("Target Outputs:");
            neuralDataSet.printTargetOutput();

            System.out.println("Neural Output after training:");
            deltaRule.forward();
            neuralDataSet.printNeuralOutput();

            weight = nn.getOutputLayer().getWeight(0, 0);
            bias = nn.getOutputLayer().getWeight(1, 0);

            System.out.println("Weight found:"+String.valueOf(weight));
            System.out.println("Bias found:"+String.valueOf(bias));

            Double[][] testDataSet ={
            {6.0 , fncTest(44.9)}
        ,   {8.0 , fncTest(40.1)}
        ,   {5.0 , fncTest(47.3)}
        ,   {0.0 , fncTest(59.3)}
        ,   {10.0 , fncTest(40.1)}
        ,   {10.0 , fncTest(35.3)}
        ,   {14.0 , fncTest(25.7)}
        ,   {8.0 , fncTest(40.1)}
        ,   {6.0 , fncTest(44.9)}
        ,   {5.0 , fncTest(47.3)}
        ,   {12.0 , fncTest(30.5)}
        ,   {1.0 , fncTest(56.9)}
        ,   {9.0 , fncTest(37.7)}
        ,   {7.0 , fncTest(42.5)}
        ,   {13.0 , fncTest(28.1)}
        ,   {2.0 , fncTest(52.1)}
        ,   {2.0 , fncTest(54.5)}
        ,   {12.0 , fncTest(30.5)}
        ,   {13.0 , fncTest(28.1)}
        ,   {7.0 , fncTest(42.5)}
        ,   {0.0 , fncTest(59.3)}
        ,   {3.0 , fncTest(52.1)}
        ,   {10.0 , fncTest(35.3)}
        ,   {5.0 , fncTest(47.3)}
        ,   {13.0 , fncTest(28.1)}
                ,   {11.0 , fncTest(32.9)}
        ,   {7.0 , fncTest(42.5)}
                ,   {10.0 , fncTest(35.3)}

        };

            NeuralDataSet testDataSet = new NeuralDataSet(testDataSet, inputColumns, outputColumns);

            deltaRule.setTestingDataSet(testDataSet);
            deltaRule.test();
            testDataSet.printNeuralOutput();
        }
        catch(NeuralException ne)
        {

        }


    }

    public static double fncTest(double x)
    {
       return 59.3 - 2.4 * x;
    }

}
