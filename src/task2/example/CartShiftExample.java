import mpi.Cartcomm;
import mpi.MPI;
import mpi.ShiftParms;

public class CartShiftExample {

    public static void main(String[] args) throws InterruptedException {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();

        int dims[] = new int[]{2, 2};							// number of processes in each dimension
        boolean periods[] = new boolean[]{true, true};			// if grid is periodic, in each dimension 
        boolean reorder = true;									// if ranking may be reordered
        int coords[] = new int[2];
        
        // data:
        int[] sendrecvBuff = new int[]{123+rank, 456+rank, 789+rank, 101+rank};
              
        Cartcomm comm = MPI.COMM_WORLD.Create_cart(dims, periods, reorder);
        coords = comm.Coords(rank);
        
        // Shift: 0 --> up/down  1 ---> left/right
        switch (rank) {
            case 0: // Master
                System.out.println("I am the Master and thats the data:");
                for (int i = 0; i < dims[0]; i++) {
                    for (int j = i + 0; j < dims[1] + i; j++) {
                        System.out.print(sendrecvBuff[i + j] + " | ");
                    }
                    System.out.println();
                }
                System.out.println("-------------------------------------");
            default: // all
            	Thread.sleep(1000);
            	System.out.println("[" + rank + "]: " + "My coordinates are: [" + coords[0] + ", " + coords[1] + "] and my data is: " + sendrecvBuff[coords[0]*dims[0] + coords[1]]);
            	ShiftParms parms = comm.Shift(1, 1); // --> 1 nach rechts
                int src = parms.rank_source;
                int dest = parms.rank_dest; 
                System.out.println("[" + rank + "]: " + "Sendrecv src: " + src + " dest: " + dest + " data: " + sendrecvBuff[coords[0]*dims[0] + coords[1]]);               
                MPI.COMM_WORLD.Sendrecv_replace(sendrecvBuff, coords[0]*dims[0] + coords[1], 1, MPI.INT, dest, 555 , src, 555);
                System.out.println("[" + rank + "]: " + "My coordinates are: [" + coords[0] + ", " + coords[1] + "] and my data is now: " + sendrecvBuff[coords[0]*dims[0] + coords[1]]);
                break;
        }

        MPI.Finalize();
    }
}

