package ece.cpen502;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNet {

    final double[] bias_arr = {1.0}; // The input for each neurons bias weight
    final double error_threshold = 0.05;

    int numOutput;
    int numInputs;
    int numHiddenLayerNeurons;
    double learningRate;
    double momentumValue;
    double sigmoidLB;
    double sigmoidUB;

    double[][] v_inputToHidden;
    double[][] w_hiddenToOutput;

//    double[][] xorPatterns = {{0,0}, {0,1}, {1,0}, {1,1}};
//    double[] xorExpectedOutput = {0,1,1,0};
    double[][] xorPatterns = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
    double[] xorExpectedOutput = {-1,1,1,-1};

    double[] zDotProduct;
    double[] zActivation;

    //Compute dot product and activation at each output layer neuron
    double[] yDotProduct;
    double[] yActivation;

    double[][] v_inputToHidden_corrTermPrev;
    double[][] w_hiddenToOutput_corrTermPrev;

    // Constructor
    public NeuralNet(int argNumOutput,
                     int argNumInputs,
                     int argNumHidden,
                     float argLearningRate,
                     float argMomentumTerm,
                     double argA,
                     double argB)
    {

        this.numOutput = argNumOutput;
        this.numInputs = argNumInputs;
        this.numHiddenLayerNeurons = argNumHidden;
        this.learningRate = argLearningRate;
        this.momentumValue = argMomentumTerm;
        this.sigmoidLB = argA;
        this.sigmoidUB = argB;

        this.initialize();

    }

    private void initialize(){

        int num_attempts = 100;
        int max_iteration = 100;
        int[] maxIterationEachAttempt = new int[num_attempts];

        double[][] track_mse_and_max_attempt = new double[num_attempts][max_iteration+1];
        for(int attemptInd = 0; attemptInd < num_attempts; attemptInd++){

            double[] meanSquaredError = new double[max_iteration];
            int max_used_iteration = 0;

            // Initialize the weights
            this.v_inputToHidden = this.initializeWeights(this.numInputs, this.numHiddenLayerNeurons);
            this.w_hiddenToOutput = this.initializeWeights(this.numHiddenLayerNeurons, this.numOutput);

            // Initialize the weight correction terms (changes) to zero (no correction term for the first iteration)
            v_inputToHidden_corrTermPrev = new double[this.numInputs+1][this.numHiddenLayerNeurons];
            w_hiddenToOutput_corrTermPrev = new double[this.numHiddenLayerNeurons+1][this.numOutput];

            // Declare some class variables
            this.zDotProduct = new double[this.numHiddenLayerNeurons];
            this.zActivation = new double[this.numHiddenLayerNeurons];
            this.yDotProduct = new double[this.numOutput];
            this.yActivation = new double[this.numOutput];

            for(int iteration_index = 0; iteration_index < max_iteration; iteration_index++){

                double[] forward_prop_output = new double[this.xorPatterns.length];

                for(int xor_index = 0; xor_index < this.xorPatterns.length; xor_index++){

                    double[] xor_input = this.xorPatterns[xor_index];
                    double[] correct_y = {this.xorExpectedOutput[xor_index]};

                    double[] xorInputWithBias = concatenate(xor_input, this.bias_arr);
                    double[] ret = forward_propagation(xorInputWithBias);
                    forward_prop_output[xor_index] = ret[0];

                    TwoArrays bck_ret = back_propagation( this.yActivation, correct_y, this.yDotProduct, this.zActivation, this.zDotProduct, xorInputWithBias);
//                System.out.println("v_inputToHidden");
//                print2d(bck_ret.vWeights);
//                System.out.println("w_hiddenToOutput");
//                print2d(bck_ret.wWeights);

//                System.out.println("--- Iteration num "+xor_index+" is complete ---");
                }

                // calculate MSE
                for(int xorInd = 0; xorInd < this.xorPatterns.length; xorInd++){
                    meanSquaredError[iteration_index] += (this.xorExpectedOutput[xorInd] - forward_prop_output[xorInd]) * (this.xorExpectedOutput[xorInd] - forward_prop_output[xorInd]);
                }
                double mse = 0.5*meanSquaredError[iteration_index];
                meanSquaredError[iteration_index] = mse;

                max_used_iteration = iteration_index;
                if(mse < this.error_threshold){
                    System.out.println("Threshold reached. Breaking the loop. Iteration count: "+iteration_index+". MSE:"+mse+" Iteration index:"+iteration_index);
                    break;
                }

            }

            maxIterationEachAttempt[attemptInd] = max_used_iteration;

            double[] max_used_tr_arr = {max_used_iteration};
            double[] meanSquaredErrorFinal = concatenate(max_used_tr_arr, meanSquaredError);
            track_mse_and_max_attempt[attemptInd] = meanSquaredErrorFinal;
      }

        // Writing to file
        try{
            String output_str = prepare_2d_arr_for_matlab_import(track_mse_and_max_attempt);
            save("TrackMSEMAXiterationBipolarWithMomentum.txt", output_str);
        }
        catch (Exception e){
            System.out.println(e.getMessage());
        }

    }

    private double[] forward_propagation(double[] xorInputWithBias){

        //Compute dot product and activation at each hidden layer neuron
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            this.zDotProduct[j] = dotProduct(xorInputWithBias, get_col_from_2d_arr(this.v_inputToHidden, j));
            this.zActivation[j] = customSigmoid(this.zDotProduct[j], this.sigmoidLB, this.sigmoidUB);
        }

        //Concatenate bias to the hidden layer output
        double[] zOutputWithBias = concatenate(zActivation, this.bias_arr);

        //Compute dot product and activation at each output layer neuron
        for (int k = 0; k < this.numOutput; k++){
            this.yDotProduct[k] = dotProduct(zOutputWithBias, get_col_from_2d_arr(this.w_hiddenToOutput, k));
            this.yActivation[k] = customSigmoid(this.yDotProduct[k], this.sigmoidLB, this.sigmoidUB);
        }

        return yActivation;
    }

    // Compute dot product between the inputs (including bias) and their weights
    public double dotProduct(double[] inputs, double[] weights)
    {
        double dotProd = 0.0;
        assert(inputs.length == weights.length);
        for (int i = 0; i < inputs.length; ++i) {
            dotProd += inputs[i] * weights[i];
        }
        return dotProd;
    }

    /**
    * This method does backward error propagation.
    * @param yOutput The activation of the output
    * @param yOutputCorrect The correct output for the given inputs
    * @param yDotProd The dot product at the output neurons (from forward propagation)
    * @param zActiv The activation function output of the neurons at the hidden layer (from forward propagation)
    * @param zDotProduct The dot product at the neurons at the hidden layer (from forward propagation)
    * @param xorInputWithBias The XOR inputs with the bias neuron
    * @return The updated weights from input to hidden and from hidden to output
    */
    private TwoArrays back_propagation(double[] yOutput, double[] yOutputCorrect, double[] yDotProd, double[] zActiv, double[] zDotProduct, double[] xorInputWithBias){

        double[] zOutputWithBias = concatenate(zActiv, this.bias_arr);

        // Compute delta for each output neuron
        double[] yOutputDelta = new double[this.numOutput];
        for (int k = 0; k < this.numOutput; k++){
            yOutputDelta[k] = (yOutputCorrect[k] - yOutput[k]) * customSigmoidDerivative(yDotProd[k], this.sigmoidLB, this.sigmoidUB);
        }

        // Compute weight correction terms for the weights from hidden to output
        double[][] w_hiddenToOutput_corrTerm = new double[this.numHiddenLayerNeurons+1][this.numOutput];
        for (int k = 0; k < this.numOutput; k++){
            for (int j = 0; j <= this.numHiddenLayerNeurons; j++){
                w_hiddenToOutput_corrTerm[j][k] = this.learningRate * yOutputDelta[k] * zOutputWithBias[j] + this.momentumValue * this.w_hiddenToOutput_corrTermPrev[j][k];
                // Update weights correction term at this iteration to be used as prevCorrTerm in the next iteration
                this.w_hiddenToOutput_corrTermPrev[j][k] = w_hiddenToOutput_corrTerm[j][k];
            }
        }

        // Update weights from hidden to output
        for (int k = 0; k < this.numOutput; k++){
            for (int j = 0; j <= this.numHiddenLayerNeurons; j++){
                this.w_hiddenToOutput[j][k] = this.w_hiddenToOutput[j][k] + w_hiddenToOutput_corrTerm[j][k];
            }
        }

        // Compute dot product of delta inputs (from the output layer) for each hidden layer neuron
        double[] deltaInputHidden_dotProd = new double[this.numHiddenLayerNeurons];
        for (int j=0; j < this.numHiddenLayerNeurons; j++){
            deltaInputHidden_dotProd[j] = dotProduct(yOutputDelta,  this.w_hiddenToOutput[j]);
        }

        // Compute delta error for each hidden layer neuron
        double[] deltaErrorHidden = new double[this.numHiddenLayerNeurons];
        for (int j=0; j < this.numHiddenLayerNeurons; j++) {
            deltaErrorHidden[j] = deltaInputHidden_dotProd[j] * customSigmoidDerivative(zDotProduct[j], this.sigmoidLB, this.sigmoidUB);
        }

        // Compute weight correction terms for the weights from input layer to hidden layer
        double[][] v_inputToHidden_corrTerm = new double[this.numInputs+1][this.numHiddenLayerNeurons];
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            for (int i = 0; i <= this.numInputs; i++){
                v_inputToHidden_corrTerm[i][j] = this.learningRate*deltaErrorHidden[j] * xorInputWithBias[i] + this.momentumValue * this.v_inputToHidden_corrTermPrev[i][j];
                // Update weights correction term at this iteration to be used as prevCorrTerm in the next iteration
                this.v_inputToHidden_corrTermPrev[i][j] = v_inputToHidden_corrTerm[i][j];
            }
        }

        // Update weights from input layer to hidden layer
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            for (int i = 0; i <= this.numInputs; i++){
                this.v_inputToHidden[i][j] = this.v_inputToHidden[i][j] + v_inputToHidden_corrTerm[i][j];
            }
        }
        return new TwoArrays(this.v_inputToHidden, this.w_hiddenToOutput);
    }

    /**
    * This method implements a general sigmoid with asymptotes bounded by (a,b)
    * @param x The input
    * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
    */
    public double customSigmoid(double x, double a, double b) {
        double gamma = b - a;
        double eta = -a;
        return gamma * (1.0 / (1.0 + (double) Math.exp(-x))) - eta;
    }

    public double customSigmoidDerivative(double x, double a, double b){
        double gamma = b - a;
        double eta = -a;
        // Compute custom sigmoid
        double customSigmoidValue =  customSigmoid(x, a, b);
        // Compute derivative of custom sigmoid
        return 1/gamma * (eta + customSigmoidValue) * (gamma - eta - customSigmoidValue);
    }

    /**
    * Initialize the weights to random values.
    * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
    * Like wise for hidden units. For say 2 hidden units which are stored in an array.
    * [0] & [1] are the hidden & [2] the bias.
    * We also initialise the last weight change arrays. This is to implement the alpha term.
    */
    public double[][] initializeWeights(int numInputs, int numHiddenLayer) {
        // Initialize weights as random variables in [-0.5, 0.5], including weight for the bias
        double minWeight = -0.5f;
        double maxWeight = 0.5f;
        Random rand = new Random();

        double[][] v_ij = new double[numInputs+1][numHiddenLayer];
        for (int j = 0; j < numHiddenLayer; j++){
            for (int i = 0; i <= numInputs; i++){
                v_ij[i][j] = minWeight + rand.nextFloat() * (maxWeight - minWeight);
            }
        }

        return v_ij;
    }

    private String prepare_2d_arr_for_matlab_import(double[][] arr){

        String output = "[";
        int index = 0;
        for (double[] x : arr)
        {
            output += Arrays.toString(x)+";";

            if(index+1 < arr.length ){
                output += System.lineSeparator();
            }

            index++;
        }
        output += "]";
        return output;

    }

    private void save(String file_path, String content) throws Exception
    {
        PrintStream out = new PrintStream(new FileOutputStream(file_path));
        out.print(content);
    }

    private double[] get_col_from_2d_arr(double[][] arr, int col_loc){

        List<Double> dbl_list = new ArrayList<Double>();
        for (double[] doubles : arr) {
            for (int col_index = 0; col_index < arr[0].length; col_index++) {
                if (col_index == col_loc) {
                    dbl_list.add(doubles[col_index]);
                }
            }
        }
        return dbl_list.stream().mapToDouble(i->i).toArray();

    }

    public double[] concatenate(double[] array1, double[] array2) {
        double[] array1and2 = new double[array1.length + array2.length];
        System.arraycopy(array1, 0, array1and2, 0, array1.length);
        System.arraycopy(array2, 0, array1and2, array1.length, array2.length);
        return array1and2;
    }

    public void print2d(double[][] arr){
        System.out.println("Printing 2D arr");
        for (double[] x : arr)
        {
            for (double y : x)
            {
                System.out.print(y + " ");
            }
            System.out.println();
        }
    }

    public static class TwoArrays {
        public final double[][] vWeights;
        public final double[][] wWeights;
        public TwoArrays(double[][] A, double[][] B) {
            this.vWeights = A;
            this.wWeights = B;
        }
    }

}
