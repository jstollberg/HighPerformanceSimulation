
import mpi.*;
import java.util.Random;

public class AllReduceMPJExample {

    public static void main(String[] args) {
        allreduce(args);
        reduce(args);
    }

    private static void reduce(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        if (rank == 0) {
        	
            // eigene 'Berechnung'
            int randomNumber = makeRandomNumber();
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + ">: "
                    + randomNumber);

            // Reduce randomNumber
            int[] sendBuff = new int[1];
            sendBuff[0] = randomNumber;
            int[] recvBuff = new int[1];
            MPI.COMM_WORLD.Reduce(sendBuff, 0, recvBuff, 0, 1, MPI.INT,
                    MPI.SUM, 0);
            randomNumber = recvBuff[0];
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + "> nach Kommunikation: "
                    + recvBuff[0]);
            System.out.println("Summe: " + randomNumber);
        } else {
            int randomNumber = makeRandomNumber();
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + ">: " + randomNumber);
            int[] sendBuff = new int[1];
            sendBuff[0] = randomNumber;
            int[] recvBuff = new int[1];
            MPI.COMM_WORLD.Reduce(sendBuff, 0, recvBuff, 0, 1, MPI.INT,
                    MPI.SUM, 0);
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + "> nach Kommunikation: " + recvBuff[0]);
        }
        MPI.Finalize();
    }

    private static void allreduce(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        if (rank == 0) {
        	
            // eigene 'Berechnung'
            int randomNumber = makeRandomNumber();
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + ">: "
                    + randomNumber);

            // Allreduce randomNumber
            int[] sendBuff = new int[1];
            sendBuff[0] = randomNumber;
            int[] recvBuff = new int[1];
            MPI.COMM_WORLD.Allreduce(sendBuff, 0, recvBuff, 0, 1, MPI.INT,
                    MPI.SUM);
            randomNumber = recvBuff[0];
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + "> nach Kommunikation: "
                    + recvBuff[0]);
            System.out.println("Summe: " + randomNumber);
        } else {
            int randomNumber = makeRandomNumber();
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + ">: " + randomNumber);
            int[] sendBuff = new int[1];
            sendBuff[0] = randomNumber;
            int[] recvBuff = new int[1];
            MPI.COMM_WORLD.Allreduce(sendBuff, 0, recvBuff, 0, 1, MPI.INT,
                    MPI.SUM);
            System.out.println("Zahl von <" + (rank + 1) + "/" + size + "> nach Kommunikation: " + recvBuff[0]);
        }
        MPI.Finalize();
    }

    public static int makeRandomNumber() {
        Random randomizer = new Random();
        return randomizer.nextInt(100);
    }
}
