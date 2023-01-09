import mpi.*;

public class HelloWorldMPI {

	 public static void main(String[] args) {

		 MPI.Init(args);

	        int rank = MPI.COMM_WORLD.Rank();
	        int size = MPI.COMM_WORLD.Size();
	        
	        

	        // root process (master):
	        if (rank == 0) {
	        	
	        	System.out.println("Name: " + MPI.Get_processor_name());
	        	System.out.println("NUM_OF_PROCESSORS: " + MPI.NUM_OF_PROCESSORS);
	        	System.out.println("HOST: " + MPI.HOST);
	        	System.out.println("SIZE: " + MPI.COMM_WORLD.Size());
	        	System.out.println("RANK: " + MPI.COMM_WORLD.Rank());
	        	
	            System.out.println("Hello World! I am number <"+ (rank + 1) + "/" + size + "> *I am the root process.* \n");
	        }
	        // the other processes:
	        else {
	        	
	            System.out.println("Hello World! I am number <"+ (rank + 1) + "/" + size + ">\n");
	        }
	        
	        MPI.Finalize();
				 
	    }
}

