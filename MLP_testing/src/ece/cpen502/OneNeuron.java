package ece.cpen502;

public class OneNeuron {

    float weight[];

    // Constructor
    public OneNeuron(float[] weight){

        this.weight = weight;
    }

    public float[] getWeight() {
        return weight;
    }

    public void setWeight(float[] weight) {
        this.weight = weight;
    }
}
