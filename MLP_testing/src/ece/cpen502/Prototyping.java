package ece.cpen502;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Prototyping {

    public static void main(String[] args){
    random_test();
    }

    public static void random_test() {
        int numInputs = 2;
        int umHiddenLayerNeurons = 4;
        int numOutput = 1;
        float learningRate = 0.2f;
        float momentumValue = 0.9f;
        double sigmoidLB = -1.0;
        double sigmoidUB = 1.0;
        NeuralNet nn = new NeuralNet(numOutput, numInputs, umHiddenLayerNeurons, learningRate, momentumValue, sigmoidLB, sigmoidUB);
    }

    //        try{
//            read_file();
//        }
//        catch (Exception e){
//            System.out.println(e);
//        }

    private static void read_file() throws Exception{

        Scanner input = new Scanner(new File("rmse_error.txt"));

        String arr_str = "";
        while (input.hasNext()){
            arr_str = arr_str+""+input.next();
        }

        String[] modified_nbr = arr_str.split(",");
        Double[] dbl_arr = new Double[modified_nbr.length];
        for(int i = 0; i< modified_nbr.length; i++){
            String tmp_str = modified_nbr[i];
            tmp_str = tmp_str.replaceAll("[^0-9\\.]", "");
            dbl_arr[i] = Double.parseDouble(tmp_str);
        }

        System.out.println(Arrays.toString(dbl_arr));

    }

    private static void  write_to_file(String file_path, String content) throws Exception
    {
        PrintStream out = new PrintStream(new FileOutputStream(file_path));
        out.print(content);
    }

}
