package task1;

public class Main {
    public static void main(String[] args) {
        // setup
        int m = 100;
        int origin = -100;
        int bound = 100;

        // compute matrix vector product
        MatrixVector matVec = new MatrixVector(m, origin, bound);
        double seqTime = matVec.timeSequential();

        System.out.println("Sequential matrix-vector product took " + seqTime + " ms.");
    }

}
